"""CoCoS: Self-Correcting Code Generation Using Small Language Models."""

PAD_TOKEN_CONFIG = {
    "llama": {"pad_token_id": 128004, "pad_token": "<|finetune_right_pad_id|>"},
    "qwen": {"pad_token_id": 151643, "pad_token": "<|endoftext|>"},
    "deepseek": {"pad_token_id": 32014, "pad_token": "<|end▁of▁sentence|>"},
}

# notice: begin_token: [BEGIN], correct_token: [CORRECT], you should change manually if error occurs.
BOOST_TOKEN_CONFIG = {
    "llama": {"begin_token_id": [33722, 16841], "correct_token_id": [44604, 878, 45940]},
    "qwen": {"begin_token_id": [32622, 16436], "correct_token_id": [43504, 868, 44840]},
    "deepseek": {"begin_token_id": [58, 29509, 60], "correct_token_id": [58, 34, 1692, 25661, 60]},
}


def configure_padding(model, tokenizer, model_name_or_path):
    """Configure pad token for the model and tokenizer based on model architecture."""
    model_id = model_name_or_path.lower()
    for key, config in PAD_TOKEN_CONFIG.items():
        if key in model_id:
            model.config.pad_token_id = config["pad_token_id"]
            tokenizer.pad_token = config["pad_token"]
            tokenizer.pad_token_id = config["pad_token_id"]
            break
    model.resize_token_embeddings(len(tokenizer))
