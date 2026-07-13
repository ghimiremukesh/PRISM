#!/bin/bash
#SBATCH --job-name=Olmo3-7B-Instruct-MajVote-DAPO
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=ml-p4d-24xlarge-us-east-1c
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue

export WANDB_API_KEY= # your_wandb_api_key_here

set -x -e

source ~/.bashrc
cd /fsx/mghmr/PRISM/open-r1-intuitor
# olmo3 architecture needs the newer stack (trl 0.29 / transformers 4.57 / vllm 0.11)
source ./openr1-olmo3/bin/activate

START_TIME=$(date +%s)
echo "START TIME: $(date)"

CONFIG_FILE=recipes/Olmo3-7B/majority_vote/config_majority_vote.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps:' $CONFIG_FILE | awk '{print $2}')
MODEL=$(grep 'model_name_or_path:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '\r')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

# Dedicate last node to vLLM serving, first node to training.
TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
VLLM_NODE=${NODELIST[-1]}
WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
NUM_NODES=$((NUM_NODES - 1))

# Olmo-3-7B: tp=4, dp=2 for fast generation on 8 GPUs.
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE \
    trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size 4 --data_parallel_size 2 --port 8001 --max_model_len 4096 \
    > ./logs/vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1 &

OPTIONAL_ARGS="--vllm_server_host=$VLLM_NODE --wandb_project=open-r1 --run_name=Olmo3-7B-Instruct-MajorityVote-DAPO"

# Force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CMD=" \
    src/open_r1/grpo.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    --role \$(hostname -s): \
    --tee 3 \
    "

NODELIST_STR=$(IFS=,; echo "${TRAIN_NODES[*]}")

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$NODELIST_STR
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
