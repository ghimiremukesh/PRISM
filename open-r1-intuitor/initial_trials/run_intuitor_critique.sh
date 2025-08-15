#!/usr/bin/bash
source ./openr1/bin/activate
export WANDB_API_KEY=6aa6f9ef7e3680b05cbf8e44c9a53590729c00a9
export ACCELERATE_LOG_LEVEL=info

# Run vllm-serve in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "Qwen/Qwen2.5-3B-Instruct" > vllm-serve-intuitor-ctq.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Run accelerate launch in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info\
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/intuitor_critique.py --config recipes/Qwen2.5-3B/intuitor-critique/config_demo.yaml --wandb_project open-r1 --run_name Qwen2.5-3B-Ins-INTUITOR-CRITIQUE > run_intuitor-ctq.log 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check vllm-serve.log and run_intuitor.log for output."