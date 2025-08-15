#!/bin/bash
#SBATCH --job-name=mghmr-PRM-INT-DAPO
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=./logs_slurm/%x-%j.out
#SBATCH --error=./logs_slurm/%x-%j.err
#SBATCH --requeue

export WANDB_API_KEY=6690ae4dd209ea0517d035d7efd9c428ba921922 #optional

# export WANDB_MODE=disabled

export DATABRICKS_HOST=https://dbc-3eec905e-05ea.cloud.databricks.com
export DATABRICKS_TOKEN=dapi50c72486dd38244730fce9eb9b87f5d6
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_REGISTRY_URI=databricks-uc
export MLFLOW_EXPERIMENT_ID=1799336443725747

export HF_MLFLOW_LOG_ARTIFACTS=true  # Log model artifacts

export HF_TOKEN=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

# Specific configuration optimized for the Hugging Face Compute Cluster
# module load cuda/12.4
set -x -e

source ~/.bashrc
source ./openr1/bin/activate
echo "START TIME: $(date)"

# MODEL=$1
CONFIG_SUFFIX=$1
ACCELERATOR=$2
OPTIONAL_ARGS=$3
# CONFIG_FILE=/fsx/ubuntu/users/mghmr/Intuitor/open-r1-intuitor/recipes/Qwen2.5-3B/intuitor-critique/config_demo.yaml
CONFIG_FILE=/fsx/ubuntu/users/mghmr/Intuitor/open-r1-intuitor/recipes/Qwen2.5-7B/prm/config_genprm_sc.yaml

GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps:' $CONFIG_FILE | awk '{print $2}')
MODEL=$(grep 'model_name_or_path:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '\r')
PRM_MODEL=$(grep 'prm_model:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '\r')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6002
TRAIN_NODES=("${NODELIST[@]}")

USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi
# if using vllm
if [[ "$USE_VLLM" == "true" ]]; then
     TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 2))}")
     TRAIN_NODES=$(IFS=','; echo "${TRAIN_NODES[*]}")
     VLLM_NODE=${NODELIST[-1]} # Last node
     PRM_VLLM_NODE=${NODELIST[-2]} # Second-Last node
     TP=$(python scripts/get_tensor_parallel_size.py --model_name $MODEL --revision $REVISION --default_tp $GPUS_PER_NODE)
     WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE * 2))
     NUM_NODES=$((NUM_NODES - 2))
     srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $MODEL --tensor_parallel_size 4 --data_parallel_size 2 \
     > ./logs/vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1 &
     srun --nodes=1 --ntasks=1 --nodelist=$PRM_VLLM_NODE trl vllm-serve --model $PRM_MODEL --port 8081 --tensor_parallel_size 4 --data_parallel_size 2 \
     > ./logs/prm_vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1 &

     OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$VLLM_NODE --prm_server_host=$PRM_VLLM_NODE --wandb_project=open-r1 --run_name=Qwen2.5-7B-PRM-INT-DAPO"
fi

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/decay_intuitor_prm.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$TRAIN_NODES
    "
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD"  > ./logs/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1

echo "END TIME: $(date)"