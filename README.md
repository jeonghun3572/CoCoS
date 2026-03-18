<h1 align="center">CoCoS: Self-Correcting Code Generation Using Small Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2505.23060"><img src="https://img.shields.io/badge/arXiv-2501.13567-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/jeonghuncho/models"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="HuggingFace"></a>
</p>

## Project Structure
```
CoCoS/
├── cocos/                  # Core package
│   ├── config.py           # Training configuration
│   ├── trainer.py          # CoCoS RL trainer
│   ├── evaluation.py       # Code execution & evaluation
│   ├── data.py             # Dataset & data collator
│   ├── prompts.py          # Prompt construction
│   └── rewards.py          # Reward computation
├── boost/                  # Boost model (optional SFT stage)
│   ├── collator.py         # Custom data collator
│   ├── trainer.py          # SFT trainer
│   └── train.py            # Training entry point
├── baselines/score/        # SCoRe baseline
│   ├── trainer.py          # SCoRe RL trainer
│   └── train.py            # Training entry point
├── train.py                # CoCoS training entry point
├── test.py                 # Evaluation entry point
├── requirements.txt
└── environment.yml
```

## Environment Setup
```bash
conda env create -f environment.yml
```
or
```bash
pip install -r requirements.txt
```

## Datasets
Download the datasets:
* Program Synthesis with Large Language Models (MBPP) [[github](https://github.com/google-research/google-research/tree/master/mbpp)]
* (Optional) KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding (KodCode) [[github](https://github.com/KodCode-AI/kodcode)]


## Data Format
We follow the MBPP data format.
```json
{
    "text": "<question>",
    "code": "<canonical_solution>",
    "test_list": [
        "assert ...",
        "assert ...",
    ]
}
```
If you want to train or test using our code on a dataset other than MBPP, we recommend constructing your data to match the format of that dataset.


## (Optional) Boost Model
We trained CoCoS using the Boost model, but this is not mandatory. If you do not use the Boost model, you can train it using few-shot prompting.

* Data format
```json
[
    {
        "prompt": "<question>\n\n[BEGIN]",
        "completion": "<first turn>\n[DONE]"
    },
    {
        "prompt": "<question>\n\n[BEGIN]\n<first turn>\n[DONE]\n\n<auxiliary instruction>\n\n[CORRECT]",
        "completion": "<second turn>\n[DONE]"
    }
]
```
In this paper, we trained the Boost model by fixing the ratio of first turn to second turn to 1:1.

* Train
```bash
deepspeed \
    --num_gpus ${num_gpus} \
    --master_port ${master_port} \
    ./boost/train.py \
        --deepspeed ${deepspeed} \
        --model_name_or_path ${model_name_or_path} \
        --global_batch_size ${global_batch_size} \
        --train_data ${train_data} \
        --eval_data ${eval_data} \
        --output_dir ${output_dir} \
        --report_to wandb \
        --wandb_run_name ${wandb_run_name} \
        --weight_decay ${weight_decay}
```

## CoCoS
* Train
```bash
deepspeed \
    --num_gpus ${num_gpus} \
    --master_port ${master_port} \
    train.py \
        --deepspeed ${deepspeed} \
        --model_name_or_path ${model_name_or_path} \
        --local_rollout_forward_batch_size ${local_rollout_forward_batch_size} \
        --train_data ${train_data} \
        --output_dir ${output_dir} \
        --report_to wandb \
        --wandb_run_name ${wandb_run_name} \
        --rloo_k ${rloo_k} \
        --gamma ${gamma}
```

* Test
```bash
python test.py \
    --output_dir ${output_dir} \
    --test_data ${test_data} \
    --num_turns ${num_turns} \
    --batch_size ${batch_size} \
    --model_name_or_path ${model_name_or_path}
```

## Cite

```bibtex
@misc{cho2025selfcorrectingcodegenerationusing,
      title={Self-Correcting Code Generation Using Small Language Models}, 
      author={Jeonghun Cho and Deokhyung Kang and Hyounghun Kim and Gary Geunbae Lee},
      year={2025},
      eprint={2505.23060},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23060}, 
}
```

## Acknowledgement
This repo is partially based upon the following repos:
* Evaluating Large Language Models Trained on Code [[github](https://github.com/openai/human-eval)]
* TRL - Transformer Reinforcement Learning [[github](https://github.com/huggingface/trl)]
* Training Language Models to Self-Correct via Reinforcement Learning [[paper](https://arxiv.org/abs/2409.12917)]

Thanks for their wonderful work.
