import os
import wandb
import torch
import argparse

from datasets import load_dataset
from accelerate import PartialState

from trl import SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from boost_trainer import BoostTrainer
from boost_collator import BoostCollator


def main(args):
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING']="1"

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

    ## begin_token_id = [BEGIN
    ## correct_token_id = [CORRECT
    if "llama" in model_id:
        model.config.pad_token_id = 128004
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
        begin_token_id = [33722, 16841]
        correct_token_id = [44604, 878, 45940]

    elif "qwen" in model_id:
        model.config.pad_token_id = 151643
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
        begin_token_id = [32622, 16436]
        correct_token_id = [43504, 868, 44840]

    elif "deepseek" in model_id:
        model.config.pad_token_id = 32014
        tokenizer.pad_token = "<|end▁of▁sentence|>"
        tokenizer.pad_token_id = 32014
        begin_token_id = [58, 29509, 60]
        correct_token_id = [58, 34, 1692, 25661, 60]

    else:
        raise ValueError(f"Unsupported model: {args.model_name_or_path}. Add the pad token and the token ids of [BEGIN and [CORRECT for this model.")

    model.resize_token_embeddings(len(tokenizer))
    data_collator = BoostCollator(
        response_template=begin_token_id,
        response_template_2=correct_token_id,
        tokenizer=tokenizer
    )
    gradient_accumulation_steps = args.global_batch_size // args.per_device_train_batch_size // torch.cuda.device_count()

    training_args = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        bf16=True,
        deepspeed=args.deepspeed if args.deepspeed else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=args.max_steps,
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
        dataset_num_proc=args.dataset_num_proc,
        max_seq_length=args.max_seq_len,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    if args.report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Boost model")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output-dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--train-data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to the evaluation data file")
    parser.add_argument("--eval-steps", type=float, default=0.1, help="Number of steps between evaluations")
    parser.add_argument("--global-batch-size", type=int, default=256, help="Batch size (including gradient accumulation, multi-gpu training)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="The initial learning rate for Adam")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="The scheduler type to use", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay if we apply some")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total number of training steps to perform")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--dataset-num-proc", type=int, default=30, help="Number of processes for dataset tokenization")
    parser.add_argument("--early-stopping-patience", type=int, default=1, help="Early stopping patience")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Name of the W&B run")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Model identifier to load from huggingface.co/models")
    parser.add_argument("--report-to", type=str, default="tensorboard")

    # DeepSpeed launcher passes --local_rank
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str, default="", help="Path to the DeepSpeed config (e.g. deepspeed_zero2.json)")

    args = parser.parse_args()

    main(args)
