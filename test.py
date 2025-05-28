import os
import torch
import json
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from utils import (
    make_fewshot,
    make_prompt,
    get_reward,
    calculate_improve,
    Collator,
    Dataset
)

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = Dataset(args.test_data)
    fewshot_dataset = Dataset(args.fewshot_data) if args.fewshot_data else None
    fewshot = make_fewshot(fewshot_dataset) if fewshot_dataset else None
    data_collator = Collator()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map="auto",
    ).eval()

    model_id = args.model_name_or_path.lower()
    if "llama" in model_id:
        model.config.pad_token_id = 128004
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    elif "qwen" in model_id:
        model.config.pad_token_id = 151643
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    elif "deepseek" in model_id:
        model.config.pad_token_id = 32014
        tokenizer.pad_token = "<|end▁of▁sentence|>"
        tokenizer.pad_token_id = 32014
    model.resize_token_embeddings(len(tokenizer))

    max_new_tokens = 512
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    data_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=data_collator,
        drop_last=False,
    )

    total = []
    eval_total = []
    corrects, responses_for_prompt = None, None
    for batch in tqdm(data_loader):
        output_batch_turn = []
        for num_turn in range(1, args.num_turns + 1):
            prompts = make_prompt(batch, num_turn, args.task, responses_for_prompt, fewshot)
            prompts = tokenizer.batch_encode_plus(
                prompts,
                padding='longest',
                max_length=8192,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False,
            ).to(model.device)

            outputs = model.generate(**prompts, generation_config=generation_config)
            outputs = tokenizer.batch_decode(outputs[:, prompts['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            test_lists = [data['test_list'] for data in batch]
            _, corrects, responses_for_prompt = get_reward(None, outputs, test_lists, num_turn, corrects, responses_for_prompt)
            output_batch_turn.append(outputs)

        assert len(output_batch_turn) == args.num_turns
        for i, data in enumerate(batch):
            temp = data.copy()
            output_per_batch = []
            eval_results_per_batch = {}
            for j in range(args.num_turns):
                eval_results_per_batch[f"acc_at_{j+1}"] = 0
                output_per_batch.append(output_batch_turn[j][i])

            for j in range(args.num_turns):
                if corrects[i][j] == f"pass_{len(data['test_list'])}_{len(data['test_list'])}":
                    eval_results_per_batch[f"acc_at_{j+1}"] = 1
                if j > 0:
                    eval_results_per_batch[f"better_at_{j+1}"] = 0
                    eval_results_per_batch[f"worsen_at_{j+1}"] = 0

                    result, inco_co, co_inco, inco_inco, co_co = calculate_improve(prev=corrects[i][j-1], curr=corrects[i][j])
                    if result == "better":
                        eval_results_per_batch[f"better_at_{j+1}"] = 1
                    elif result == "worse":
                        eval_results_per_batch[f"worsen_at_{j+1}"] = 1
                    eval_results_per_batch[f"inco-co_at_{j+1}"] = inco_co
                    eval_results_per_batch[f"co-inco_at_{j+1}"] = co_inco
                    eval_results_per_batch[f"inco-inco_at_{j+1}"] = inco_inco
                    eval_results_per_batch[f"co-co_at_{j+1}"] = co_co

            temp['response'] = output_per_batch
            temp['evaluate'] = eval_results_per_batch
            total.append(temp)
            eval_total.append(eval_results_per_batch)


    parts = args.model_name_or_path.split('/')
    log_name = f"{parts[-2]}-{parts[-1]}"
    # log_name = f"{args.model_name_or_path.split('/')[-1]}"
    with open(f"{args.output_dir}/{log_name}.jsonl", "w", encoding="utf-8") as f:
        for data in total:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Calculate the average of each key in eval_total
    assert len(eval_total) == len(test_dataset)
    avg_results = {}
    keys = eval_total[0].keys()
    for key in keys:
        avg_results[f"{key}"] = round(sum(d[key] for d in eval_total) / len(test_dataset), 3)

    # Save the average results to a JSON file
    with open(f"{args.output_dir}/{log_name}.log", "w", encoding="utf-8") as f:
        for key, value in avg_results.items():
            f.write(f"{key}: {value}\n")

    print("Done!")
    print(f"Files are saved in {args.output_dir}/{log_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with SFTTrainer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Size of the tensor parallelism")
    parser.add_argument("--num_turns", type=int, default=2, help="Number of turns")
    parser.add_argument("--task", type=str, default="code", choices=["code", "math"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--fewshot_data", type=str, default=None, help="Path to the few-shot data file")
    args = parser.parse_args()

    main(args)
