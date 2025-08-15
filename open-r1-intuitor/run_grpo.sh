#!/bin/bash

source ./openr1/bin/activate
# no need since we cannot use wandb
export WANDB_API_KEY=6690ae4dd209ea0517d035d7efd9c428ba921922
export ACCELERATE_LOG_LEVEL=info
# we might not need it
export HF_TOKEN=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

# Run vllm-serve in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "HuggingFaceTB/SmolLM3-3B" > vllm-serve-1.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Run accelerate launch in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/grpo.py --config recipes/SmolLM3-3B/grpo/config_demo.yaml --wandb_project open-r1 --run_name SmolLM-GRPO-DAPO > run_grpo-1.log 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check vllm-serve.log and run_grpo.log for output."

wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully. Cleaning up processes..."
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE. Cleaning up processes..."
fi

# Clean up all processes
ps aux | grep "/users/mghmr" | grep -v grep | awk '{print $2}' | xargs kill -9
echo "Cleanup completed."