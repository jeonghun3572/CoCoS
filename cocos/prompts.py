import re

CORRECTION_INSTRUCTION = (
    "\n\nThere might be an error in the code above because of lack of understanding "
    "of the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final correct Python program!\n\n"
)


def code_process(text, test_list=None):
    """Extract code from model output by matching [BEGIN]...[DONE] / [CORRECT]...[DONE] markers."""
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
        text = (
            extracted_code
            .replace("```python\n", "").replace("\n```", "")
            .replace("[BEGIN]", "").replace("[DONE]", "").replace("[CORRECT]", "")
        )
    return text


def make_fewshot(dataset):
    """Build a few-shot prompt from a dataset of examples."""
    prompt = ""
    for data in dataset:
        test_list = '\n'.join(data['test_list'])
        prompt += (
            f"You are an expert Python programmer, and here is your task: {data['text']} "
            f"Your code should pass these tests:\n\n{test_list.strip()}\n\n"
            f"[BEGIN]\n{data['code']}\n[DONE]\n\n"
        )
    return prompt.lstrip()


def _build_task_header(data):
    test_list = '\n'.join(data['test_list'])
    return (
        f"You are an expert Python programmer, and here is your task: {data['text']} "
        f"Your code should pass these tests:\n\n{test_list.strip()}\n\n[BEGIN]\n"
    )


def make_prompt(batch, num_turn, responses_for_prompt=None, fewshot=None):
    """Build prompts for CoCoS training/inference."""
    prompts = []
    for i, data in enumerate(batch):
        prompt = fewshot if (num_turn == 1 and fewshot is not None) else ""
        prompt += _build_task_header(data)

        if num_turn > 1:
            prompt += f"{responses_for_prompt[i][-1]}\n[DONE]"
            if fewshot is not None:
                prompt += CORRECTION_INSTRUCTION + "[BEGIN]\n"
            else:
                prompt += CORRECTION_INSTRUCTION + "[CORRECT]\n"

        prompts.append(prompt)
    return prompts


def make_prompt_humaneval(batch, num_turn, responses_for_prompt=None):
    """Build prompts for HumanEval evaluation."""
    prompts = []
    for i, data in enumerate(batch):
        test_list = '\n'.join(data['test_list'])
        prompt = (
            f"You are an expert Python programmer, and here is your task: "
            f"Write a python function {data['entry_point']}. Your code should pass these tests:\n\n"
            f"{test_list.strip()}\n\n[BEGIN]\n"
        )
        if num_turn > 1:
            prompt += (
                f"{responses_for_prompt[i][-1].rstrip()}\n[DONE]"
                + CORRECTION_INSTRUCTION + "[CORRECT]\n"
            )
        prompts.append(prompt)
    return prompts


def make_prompt_score(batch, num_turn, responses_for_prompt=None, fewshot=None):
    """Build prompts for SCoRe baseline training/inference."""
    prompts = []
    for i, data in enumerate(batch):
        prompt = fewshot if (num_turn == 1 and fewshot is not None) else ""
        prompt += _build_task_header(data)

        if num_turn > 1:
            prompt += (
                f"{responses_for_prompt[i].rstrip()}\n[DONE]"
                + CORRECTION_INSTRUCTION + "[CORRECT]\n"
            )
        prompts.append(prompt)
    return prompts
