<h1 align="center">Self-Correcting Code Generation Using Small Language Models</a></h2>


This is the official implementation of the following paper:

> **Self-Correcting Code Generation Using Small Language Models** [[Paper](https://arxiv.org/abs/2505.23060)]

## Environment Setup
```
conda env create -f environment.yml
```

## Datasets
Download the datasets:
* Program Synthesis with Large Language Models (MBPP) [[gihub](https://github.com/google-research/google-research/tree/master/mbpp)]
* (Optional) KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding (KodCode) [[gihub](https://github.com/KodCode-AI/kodcode)]


## Data Format
We follow the MBPP data format.
```
{
    "text": {question},
    "code": {canonical_solution},
    "test_list": [
        "assert 1",
        "assert 2",
        ...
    ]
}
```
If you want to train or test using our code on a dataset other than MBPP, we recommend constructing your data to match the format of that dataset.


## (Optional) Boost model
We trained CoCoS using the Boost model, but this is not mandatory. If you do not use the Boost model, you can train it using few-shot prompting.

* Data format
```
[
    {
        "prompt": {question}[BEGIN],
        "completion": {first turn}[DONE],
    },
    {
        "prompt": {question}[BEGIN]{first turn}[DONE]{auxiliary instruction}[CORRECT],
        "completion" {second turn}[DONE]
    }
]
```
In this paper, we trained the Boost model by fixing the ratio of first turn to second turn to 1:1.

* Train
```
echo "Boost model Training"
deepspeed \
    --num_gpus ${num_gpus} \
    --master_port ${master_port} \
    ./boost/boost_train.py \
        --deepspeed {deepspeed} \
        --model_name_or_path ${model_name_or_path} \
        --global_batch_size ${global_batch_size} \
        --train_data ${train_data} \
        --eval_data ${eval_data} \
        --output_dir ${output_dir} \
        --report_to wandb \
        --wandb_run_name ${wandb_run_name} \
        --weight_decay ${weight_decay} \
```

## CoCoS
* Train
```
echo "CoCoS Training"
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
        --weight ${weight} \
```

* Test
```
python test.py \
    --output_dir ${output_dir} \
    --test_data ${test_data} \
    --num_turns ${num_turns} \
    --batch_size ${batch_size} \
    --model_name_or_path ${model_name_or_path} \

```

## Cite

```
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
* Evaluating Large Language Models Trained on Code
 [[github](https://github.com/openai/human-eval)]
* TRL - Transformer Reinforcement Learning [[github](https://github.com/huggingface/trl)]
* Training Language Models to Self-Correct via Reinforcement Learning [[paper](https://arxiv.org/abs/2409.12917)]

Thanks for their wonderful work.