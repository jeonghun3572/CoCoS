import torch

from cocos.evaluation import CodeEval
from cocos.prompts import code_process


def _compute_pass_count(query_response, test_list):
    """Run test cases against a code response and return the number of passing tests."""
    correct_count = 0
    for testcase in test_list:
        try:
            pass_at_k, _ = CodeEval.compute(
                references=[testcase], predictions=[[query_response]], k=[1]
            )
            if float(pass_at_k["pass@1"]) == 1.0:
                correct_count += 1
        except Exception:
            pass
    return correct_count


def calcul_reward(correct):
    return float(correct.split("_")[1])


def correction_reward(num_turns, corrects, gamma=0.5):
    """Compute discounted correction reward across turns."""
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
    """Compute rewards for multi-turn evaluation (used during inference)."""
    if not isinstance(query_responses[0], str) and tokenizer is not None:
        query_responses = tokenizer.batch_decode(
            query_responses, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )

    total_responses_for_prompt = []
    total_correct = []
    total_reward = []

    assert len(query_responses) == len(test_lists)
    for i, (query_response, test_list) in enumerate(zip(query_responses, test_lists)):
        query_response = code_process(query_response.strip()).strip()
        correct_count = _compute_pass_count(query_response, test_list)

        if num_turns == 1:
            total_responses_for_prompt.append([query_response])
            total_correct.append([f"pass_{correct_count}_{len(test_list)}"])
            rewards = correction_reward(num_turns, total_correct[-1])

        elif num_turns > 1:
            assert corrects is not None
            responses_for_prompt[i].append(query_response)
            total_responses_for_prompt.append(responses_for_prompt[i])
            corrects[i].append(f"pass_{correct_count}_{len(test_list)}")
            total_correct.append(corrects[i])
            rewards = correction_reward(num_turns, corrects[i], gamma=0.5)

        total_reward.append(rewards)

    return torch.Tensor(total_reward), total_correct, total_responses_for_prompt


def get_reward_one_reward(tokenizer, query_responses, test_lists, num_turns, responses_for_prompt, gamma=0.5):
    """Compute correction reward for CoCoS RL training (single reward signal)."""
    query_responses = tokenizer.batch_decode(
        query_responses, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )

    total_reward = []
    for i, (query_response, test_list, response_for_prompt) in enumerate(
        zip(query_responses, test_lists, responses_for_prompt)
    ):
        prev_query_response = code_process(response_for_prompt[num_turns - 2].strip()).strip()
        query_response = code_process(query_response.strip()).strip()

        prev_correct_count = _compute_pass_count(prev_query_response, test_list)
        curr_correct_count = _compute_pass_count(query_response, test_list)

        corrects = [
            f"pass_{prev_correct_count}_{len(test_list)}",
            f"pass_{curr_correct_count}_{len(test_list)}",
        ]
        total_reward.append(correction_reward(num_turns, corrects, gamma=gamma))

    return torch.Tensor(total_reward)


def calculate_improve(prev, curr):
    """Compare two turns and classify the improvement direction."""
    prev_correct_count = int(prev.split('_')[1])
    curr_correct_count = int(curr.split('_')[1])
    total_test_count = int(curr.split('_')[2])

    if prev_correct_count < curr_correct_count:
        improvement = "better"
    elif prev_correct_count > curr_correct_count:
        improvement = "worse"
    else:
        improvement = "same"

    inco_co, co_inco, inco_inco, co_co = 0, 0, 0, 0
    if prev_correct_count == total_test_count:
        co_co = 1 if curr_correct_count == total_test_count else 0
        co_inco = 1 - co_co
    else:
        inco_co = 1 if curr_correct_count == total_test_count else 0
        inco_inco = 1 - inco_co

    return improvement, inco_co, co_inco, inco_inco, co_co


def get_reward_humaneval(tokenizer, query_responses, test_lists, num_turns, corrects=None, responses_for_prompt=None):
    """Compute rewards for HumanEval evaluation."""
    total_responses_for_prompt = []
    total_correct = []

    assert len(query_responses) == len(test_lists)
    for i, (query_response, test_list) in enumerate(zip(query_responses, test_lists)):
        query_response = code_process(query_response)
        correct_count = _compute_pass_count(query_response, test_list)

        if num_turns == 1:
            total_responses_for_prompt.append([query_response])
            total_correct.append([f"pass_{correct_count}_{len(test_list)}"])

        elif num_turns > 1:
            assert corrects is not None
            responses_for_prompt[i].append(query_response)
            total_responses_for_prompt.append(responses_for_prompt[i])
            corrects[i].append(f"pass_{correct_count}_{len(test_list)}")
            total_correct.append(corrects[i])

    return total_correct, total_responses_for_prompt


def get_reward_score(tokenizer, query_responses, test_lists, num_turns, responses_for_prompt, gamma=0.5):
    """Compute binary reward for SCoRe baseline training."""
    query_responses = tokenizer.batch_decode(
        query_responses, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )

    total_reward = []
    for i, (query_response, test_list, response_for_prompt) in enumerate(
        zip(query_responses, test_lists, responses_for_prompt)
    ):
        prev_query_response = code_process(response_for_prompt.strip()).strip()
        query_response = code_process(query_response.strip()).strip()

        prev_correct_count = _compute_pass_count(prev_query_response, test_list)
        curr_correct_count = _compute_pass_count(query_response, test_list)

        curr_correct_count = 1 if curr_correct_count == len(test_list) else 0
        prev_correct_count = 1 if prev_correct_count == len(test_list) else 0

        total_reward.append(curr_correct_count)

    return torch.Tensor(total_reward)
