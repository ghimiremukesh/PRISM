#!/usr/bin/bash
# Majority-vote (TTRL-style self-consistency) baseline on MATH with Qwen2.5-3B.
# Uses vanilla GRPO (stock trl GRPOTrainer) with the label-free majority_vote reward.
# No GenPRM server is needed (unlike PRISM/PRM runs).
source openr1/bin/activate
export WANDB_API_KEY= # your_wandb_api_key_here

# Serve the policy model for vLLM rollouts on GPU 0.
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "Qwen/Qwen2.5-3B" > vllm-serve.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Launch GRPO training on the remaining GPUs.
nohup env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/grpo.py --config recipes/Qwen2.5-3B/majority_vote/config_majority_vote.yaml \
    --wandb_project open-r1 --run_name Qwen2.5-3B-MajorityVote-MATH > run_majority_vote.log 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check vllm-serve.log and run_majority_vote.log for output."

wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully. Cleaning up processes..."
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE. Cleaning up processes..."
fi

# Clean up all processes
ps aux | grep "/fsx/mghmr" | grep -v grep | awk '{print $2}' | xargs kill -9
echo "Cleanup completed."
