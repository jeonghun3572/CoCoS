import os
import argparse

import trl
import torch
import wandb
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM

from cocos import configure_padding
from cocos.config import CoCoSConfig
from cocos.data import Collator, Dataset
from baselines.score.trainer import SCoReTrainer


def main(args):
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    num_gpus = PartialState().num_processes
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"

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

    max_seq_len = 8192
    max_new_tokens = 512
    args.learning_rate = 1e-5
    args.global_batch_size = 128
    args.total_episodes = 1500 * args.global_batch_size

    args.gradient_accumulation_steps = args.global_batch_size // args.per_device_train_batch_size // num_gpus
    training_args = CoCoSConfig(
        output_dir=args.output_dir,
        do_train=True,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        total_episodes=args.total_episodes,
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        run_name=args.wandb_run_name,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        num_sample_generations=0,
        kl_coef=args.kl_coef,
        num_ppo_epochs=1,
        num_mini_batches=1,
        rloo_k=args.rloo_k,
        dataset_num_proc=num_gpus,
        remove_unused_columns=False,
        max_seq_len=max_seq_len,
        response_length=max_new_tokens,
        save_safetensors=False,
        num_turns=args.num_turns,
    )

    data_collator = Collator()
    train_dataset = Dataset(args.train_data)
    fewshot_dataset = Dataset(args.fewshot_data) if args.fewshot_data is not None else None

    ref_policy = trl.create_reference_model(model)
    trainer = SCoReTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_policy,
        reward_model=None,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        fewshot_dataset=fewshot_dataset,
        first_kl_coef=args.first_kl_coef,
    )
    trainer.train()
    trainer.save_model()

    if args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCoRe baseline")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--fewshot_data", type=str, default=None, help="Path to the few-shot data file")
    parser.add_argument("--global_batch_size", type=int, default=256, help="Global batch size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name of the W&B run")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model identifier from HuggingFace")
    parser.add_argument("--local_rollout_forward_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--rloo_k", type=int, default=2, help="RLOO k")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL coefficient for second turn")
    parser.add_argument("--first_kl_coef", type=float, default=0.25, help="KL coefficient for first turn")

    # DeepSpeed
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str, default="")

    args = parser.parse_args()
    main(args)
