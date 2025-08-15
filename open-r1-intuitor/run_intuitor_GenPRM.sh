#!/usr/bin/bash
source ./openr1/bin/activate
export WANDB_API_KEY=6690ae4dd209ea0517d035d7efd9c428ba921922
export ACCELERATE_LOG_LEVEL=info
export HF_TOKEN=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

# Run vllm-serve in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "meta-llama/Llama-3.2-3B-Instruct" > vllm-serve-prm.log 2>&1 &
VLLM_PID=$!

echo "vLLM server started with PID: $VLLM_PID"

nohup env CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "GenPRM/GenPRM-7B" --port 8081 > vllm-serve-prm_teacher.log 2>&1 &
VLLM_PID=$!

echo "GenPRM server started with PID: $VLLM_PID"

# Run accelerate launch in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info\
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=6 \
    src/open_r1/intuitor_prm.py --config recipes/Llama3.2-3B/prm/config_genprm.yaml --wandb_project open-r1 --run_name Llama3.2-3B-PRM > run-prm-llama.log 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check vllm-serve.log and run_intuitor.log for output."

# ps aux | grep "/users/mghmr" | grep -v grep | awk '{print $2}' | xargs kill -9
# Wait for training to complete

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