#!/bin/bash
source ./openr1/bin/activate
export WANDB_API_KEY=6aa6f9ef7e3680b05cbf8e44c9a53590729c00a9
export ACCELERATE_LOG_LEVEL=info

# Run vllm-serve in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "Qwen/Qwen1.5-0.5B" > vllm-serve-debug.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Run accelerate launch in the background with nohup
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info\
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/gated_intuitor.py --config recipes/Qwen2.5-3B/intuitor/config_demo_debug.yaml --wandb_project debug --run_name test_sc2
echo "Training process started with PID: $TRAINING_PID"


ps aux | grep "/users/mghmr" | grep -v grep | awk '{print $2}' | xargs kill -9
