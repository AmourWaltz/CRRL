#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=3
export DATA_DIR='data/nq_hotpotqa_train'

WAND_PROJECT='CRPO'


export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=nq-hotpotqa-sft-qwen2.5-3b-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
 
nproc_per_node=1
save_path="./checkpoints"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/data/gsm8k/train.parquet \
    data.val_files=$DATA_DIR/data/gsm8k/test.parquet \
    +data.prompt_dict_keys="question" \
    +data.response_dict_keys="gold_answers" \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$BASE_MODEL \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=console \
    trainer.total_epochs=3