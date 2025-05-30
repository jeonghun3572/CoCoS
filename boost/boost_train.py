import os
import wandb
import torch
import argparse

from datasets import load_dataset
from accelerate import PartialState

from sft_trainer import BoostTrainer
from trl import SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from boost_collator import BoostCollator


def main(args):
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING']="1"

    num_gpus = torch.cuda.device_count()
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
        device_map={'':device_string},
    )
    model.config.use_cache = False

    model_id = args.model_name_or_path.lower()
    if "llama" in model_id:
        model.config.pad_token_id = 128004
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
        response_template_ids = [33722, 16841]
        response_template_ids_2 = [44604, 878, 45940]

    elif "qwen" in model_id:
        model.config.pad_token_id = 151643
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
        response_template_ids = [32622, 16436]
        response_template_ids_2 = [43504, 868, 44840]
    
    elif "deepseek" in model_id:
        model.config.pad_token_id = 32014
        tokenizer.pad_token = "<|end▁of▁sentence|>"
        tokenizer.pad_token_id = 32014
        response_template_ids = [58, 29509, 60]
        response_template_ids_2 = [58, 34, 1692, 25661, 60]

    model.resize_token_embeddings(len(tokenizer))
    max_seq_len = 8192
    data_collator = BoostCollator(
        response_template=response_template_ids,
        response_template_2=response_template_ids_2,
        tokenizer=tokenizer
    )
    args.gradient_accumulation_steps = args.global_batch_size // args.per_device_train_batch_size // torch.cuda.device_count()

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
        gradient_checkpointing_kwargs={'use_reentrant':False},
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
    parser = argparse.ArgumentParser(description="Train a model with SFTTrainer")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data file")
    parser.add_argument("--eval_steps", type=float, default=0.1, help="Number of steps between evaluations")
    parser.add_argument("--global_batch_size", type=int, default=256, help="Batch size (including gradient accumulation, multi-gpu training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The initial learning rate for Adam")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="The scheduler type to use", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name of the W&B run")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model identifier to load from huggingface.co/models")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # DeepSpeed
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str, default="")

    args = parser.parse_args()

    main(args)