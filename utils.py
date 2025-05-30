import re
import json
import random
import torch
import difflib

from code_evaluation import CodeEval

class Collator(object):
    def __call__(self, batch):
        return batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self._load_data(path)

    def __len__(self):
        return len(self.data)

    def _load_data(self, path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fin:
                self.data = [json.loads(line) for line in fin]
        elif path.endswith(".json"):
            with open(path, "r", encoding='utf-8') as fin:
                self.data = json.load(fin)
        return self.data

    def __getitem__(self, index):
        return self.data[index]

def code_process(text, test_list=None):
    pattern1 = r'^(.*?)\[DONE\]'
    pattern2 = r'\[CORRECT\].*?\[DONE\]'
    pattern3 = r'\[BEGIN\].*?\[DONE\]'
 
    match = re.search(pattern1, text, re.DOTALL)
    if not match:
        match = re.search(pattern2, text, re.DOTALL)
        if not match:
            match = re.search(pattern3, text, re.DOTALL)
    if match:
        extracted_code = match.group(0).strip()
        text = extracted_code.replace("```python\n", "").replace("\n```", "").replace("[BEGIN]", "").replace("[DONE]", "").replace("[CORRECT]", "")
    return text



def make_fewshot(dataset):
    prompt = ""
    for data in dataset:
        test_list = '\n'.join(data['test_list'])
        prompt += f"""You are an expert Python programmer, and here is your task: {data['text']} Your code should pass these tests:

{test_list.strip()}

[BEGIN]
{data['code']}
[DONE]\n\n"""
    return prompt.lstrip()


def make_prompt(batch, num_turn, responses_for_prompt=None, fewshot=None):
    prompts = []

    for i, data in enumerate(batch):
        prompt = ""
        if num_turn == 1 and fewshot is not None:
            prompt = fewshot

        test_list = '\n'.join(data['test_list'])
        prompt += f"""You are an expert Python programmer, and here is your task: {data['text']} Your code should pass these tests:

{test_list.strip()}

[BEGIN]
"""
        if num_turn > 1:
            prompt += f"{responses_for_prompt[i][-1]}\n[DONE]"
            if fewshot is not None:
                prompt += "\n\nThere might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n\n[BEGIN]\n"
            else:
                prompt += "\n\nThere might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n\n[CORRECT]\n"

        prompts.append(prompt)
    return prompts


def calcul_reward(correct):
    return float(correct.split("_")[1])


def correction_reward(num_turns, corrects, gamma=0.5):
    rewards = 0.0
    prev_reward = 0.0
    turn = num_turns
    for correct in corrects:
        curr_reward = calcul_reward(correct)
        if gamma == 0.0:
            rewards = curr_reward - prev_reward
        else:
            rewards += (gamma ** (turn - 1)) * (curr_reward - prev_reward)
        prev_reward = curr_reward
        turn -= 1
        if turn == 0:
            return rewards


def get_reward(tokenizer, query_responses, test_lists, num_turns, corrects=None, responses_for_prompt=None):
    if not isinstance(query_responses[0], str) and tokenizer is not None:
        query_responses = tokenizer.batch_decode(
            query_responses,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    total_responses_for_prompt = []
    total_correct = []
    total_reward = []

    assert len(query_responses) == len(test_lists)
    for i, (query_response, test_list) in enumerate(zip(query_responses, test_lists)):
        query_response = code_process(query_response.strip())
        query_response = query_response.strip()
        correct_count = 0

        if num_turns == 1:
            total_responses_for_prompt.append([query_response])

            for testcase in test_list:
                try:
                    pass_at_k, result = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                    if float(pass_at_k["pass@1"]) == 1.0:
                        correct_count += 1
                except:
                    pass
            total_correct.append([f"pass_{correct_count}_{len(test_list)}"])
            rewards = correction_reward(num_turns, total_correct[-1])

        elif num_turns > 1:
            assert corrects is not None
            responses_for_prompt[i].append(query_response)
            total_responses_for_prompt.append(responses_for_prompt[i])

            for testcase in test_list:
                try:
                    pass_at_k, result = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                    if float(pass_at_k["pass@1"]) == 1.0:
                        correct_count += 1
                except:
                    pass

            corrects[i].append(f"pass_{correct_count}_{len(test_list)}")
            total_correct.append(corrects[i])
            rewards = correction_reward(num_turns, corrects[i], gamma=0.5)
        total_reward.append(rewards)

    return torch.Tensor(total_reward), total_correct, total_responses_for_prompt


def get_reward_one_reward(tokenizer, query_responses, test_lists, num_turns, responses_for_prompt, gamma=0.5):
    query_responses = tokenizer.batch_decode(
        query_responses,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    total_reward = []
    for i, (query_response, test_list, response_for_prompt) in enumerate(zip(query_responses, test_lists, responses_for_prompt)):
        prev_query_response = code_process(response_for_prompt[num_turns - 2].strip())
        query_response = code_process(query_response.strip())
        prev_query_response = prev_query_response.strip()
        query_response = query_response.strip()

        prev_correct_count = 0
        curr_correct_count = 0
        for testcase in test_list:
            try:
                prev_pass_at_k, _ = CodeEval.compute(references=[testcase], predictions=[[prev_query_response]], k=[1])
                if float(prev_pass_at_k["pass@1"]) == 1.0:
                    prev_correct_count += 1
            except:
                pass
            try:
                curr_pass_at_k, _ = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                if float(curr_pass_at_k["pass@1"]) == 1.0:
                    curr_correct_count += 1
            except:
                pass

        corrects = []
        corrects.append(f"pass_{prev_correct_count}_{len(test_list)}")
        corrects.append(f"pass_{curr_correct_count}_{len(test_list)}")
        rewards = correction_reward(num_turns, corrects, gamma=gamma)
        total_reward.append(rewards)

    return torch.Tensor(total_reward)


def calculate_improve(prev, curr):
    inco_co = 0
    co_inco = 0
    inco_inco = 0
    co_co = 0
    prev_correct_count = int(prev.split('_')[1])
    curr_correct_count = int(curr.split('_')[1])
    total_test_count = int(curr.split('_')[2])

    if prev_correct_count < curr_correct_count:
        improvement = "better"
    elif prev_correct_count > curr_correct_count:
        improvement = "worse"
    else:
        improvement = "same"

    if prev_correct_count == total_test_count:
        if curr_correct_count != total_test_count:
            co_inco = 1
        else:
            co_co = 1
    if prev_correct_count != total_test_count:
        if curr_correct_count == total_test_count:
            inco_co = 1
        else:
            inco_inco = 1

    return improvement, inco_co, co_inco, inco_inco, co_co



def make_prompt_humaneval(batch, num_turn, responses_for_prompt=None):
    prompts = []
    for i, data in enumerate(batch):
        test_list = '\n'.join(data['test_list'])
        prompt = f"""You are an expert Python programmer, and here is your task: Write a python function {data['entry_point']}. Your code should pass these tests:

{test_list.strip()}

[BEGIN]
"""
        if num_turn > 1:
            prompt += f"{responses_for_prompt[i][-1].rstrip()}\n[DONE]\n\nThere might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n\n[CORRECT]\n"
        prompts.append(prompt)
    return prompts


def get_reward_humaneval(tokenizer, query_responses, test_lists, num_turns, corrects=None, responses_for_prompt=None):
    total_responses_for_prompt = []
    total_correct = []

    assert len(query_responses) == len(test_lists)
    for i, (query_response, test_list) in enumerate(zip(query_responses, test_lists)):
        query_response = code_process(query_response)
        correct_count = 0
        if num_turns == 1:
            total_responses_for_prompt.append([query_response])

            for testcase in test_list:
                try:
                    pass_at_k, result = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                    if float(pass_at_k["pass@1"]) == 1.0:
                        correct_count += 1
                except:
                    pass
            total_correct.append([f"pass_{correct_count}_{len(test_list)}"])

        elif num_turns > 1:
            assert corrects is not None
            responses_for_prompt[i].append(query_response)
            total_responses_for_prompt.append(responses_for_prompt[i])

            for testcase in test_list:
                try:
                    pass_at_k, result = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                    if float(pass_at_k["pass@1"]) == 1.0:
                        correct_count += 1
                except:
                    pass
            corrects[i].append(f"pass_{correct_count}_{len(test_list)}")
            total_correct.append(corrects[i])

    return total_correct, total_responses_for_prompt


def get_reward_score(tokenizer, query_responses, test_lists, num_turns, responses_for_prompt, gamma=0.5):
    query_responses = tokenizer.batch_decode(
        query_responses,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    total_reward = []
    for i, (query_response, test_list, response_for_prompt) in enumerate(zip(query_responses, test_lists, responses_for_prompt)):
        prev_query_response = code_process(response_for_prompt.strip())
        query_response = code_process(query_response.strip())
        prev_query_response = prev_query_response.strip()
        query_response = query_response.strip()

        prev_correct_count = 0
        curr_correct_count = 0
        for testcase in test_list:
            try:
                prev_pass_at_k, _ = CodeEval.compute(references=[testcase], predictions=[[prev_query_response]], k=[1])
                if float(prev_pass_at_k["pass@1"]) == 1.0:
                    prev_correct_count += 1
            except:
                pass
            try:
                curr_pass_at_k, _ = CodeEval.compute(references=[testcase], predictions=[[query_response]], k=[1])
                if float(curr_pass_at_k["pass@1"]) == 1.0:
                    curr_correct_count += 1
            except:
                pass

        if curr_correct_count == len(test_list):
            curr_correct_count = 1
        else:
            curr_correct_count = 0
        
        if prev_correct_count == len(test_list):
            prev_correct_count = 1
        else:
            prev_correct_count = 0

        total_reward.append(curr_correct_count)

    return torch.Tensor(total_reward)



def make_prompt_score(batch, num_turn, responses_for_prompt=None, fewshot=None):
    prompts = []

    for i, data in enumerate(batch):
        if num_turn == 1 and fewshot is not None:
            prompt = fewshot
        else:
            prompt = "" 
        test_list = '\n'.join(data['test_list'])
        prompt += f"""You are an expert Python programmer, and here is your task: {data['text']} Your code should pass these tests:

{test_list.strip()}

[BEGIN]
"""
        if num_turn > 1:
            prompt += f"{responses_for_prompt[i].rstrip()}\n[DONE]"
            prompt += "\n\nThere might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n\n[CORRECT]\n"

        prompts.append(prompt)
    return prompts
