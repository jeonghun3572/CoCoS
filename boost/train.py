import os
import argparse

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTConfig

from cocos import configure_padding, BOOST_TOKEN_CONFIG
from boost.collator import BoostCollator
from boost.trainer import BoostTrainer


def main(args):
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "right"

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map={'': device_string},
    )
    model.config.use_cache = False

    configure_padding(model, tokenizer, args.model_name_or_path)

    model_id = args.model_name_or_path.lower()
    begin_token_id, correct_token_id = None, None
    for key, config in BOOST_TOKEN_CONFIG.items():
        if key in model_id:
            begin_token_id = config["begin_token_id"]
            correct_token_id = config["correct_token_id"]
            break

    max_seq_len = 8192
    data_collator = BoostCollator(
        response_template=begin_token_id,
        response_template_2=correct_token_id,
        tokenizer=tokenizer,
    )
    args.gradient_accumulation_steps = (
        args.global_batch_size // args.per_device_train_batch_size // torch.cuda.device_count()
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        run_name=args.wandb_run_name,
        report_to=args.report_to,
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        dataset_num_proc=30,
        max_seq_length=max_seq_len,
        save_safetensors=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        eval_on_start=False,
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"{example['prompt'][i].lstrip()}\n{example['completion'][i]}"
            output_texts.append(text.strip())
        return output_texts

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")

    trainer = BoostTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        formatting_func=formatting_prompts_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    if args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Boost model")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data file")
    parser.add_argument("--eval_steps", type=float, default=0.1, help="Number of steps between evaluations")
    parser.add_argument("--global_batch_size", type=int, default=256, help="Global batch size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Per-device evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name of the W&B run")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model identifier from HuggingFace")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # DeepSpeed
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str, default="")

    args = parser.parse_args()
    main(args)
