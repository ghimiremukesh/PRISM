#!/usr/bin/bash
source ./openr1/bin/activate
export WANDB_API_KEY= # your_wandb_api_key_here
export ACCELERATE_LOG_LEVEL=info

# Array to store all PIDs
declare -a PIDS=()

# Function to clean up all processes
cleanup() {
    echo "Cleaning up all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing process $pid"
            kill -TERM $pid 2>/dev/null
            sleep 1
            kill -KILL $pid 2>/dev/null
        fi
    done
    
    # Additional cleanup for any remaining processes
    ps aux | grep "/users/your_username" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    echo "Cleanup completed."
    exit 1
}

# Set up trap to catch errors and signals
trap cleanup EXIT ERR INT TERM

# Function to monitor a process
monitor_process() {
    local pid=$1
    local name=$2
    
    while true; do
        if ! kill -0 $pid 2>/dev/null; then
            wait $pid
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "ERROR: $name (PID: $pid) failed with exit code $exit_code"
                cleanup
            fi
            break
        fi
        sleep 5
    done
}

# Run vllm-serve in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "Qwen/Qwen2.5-3B" > vllm-serve-base-code.log 2>&1 &
VLLM_PID=$!
PIDS+=($VLLM_PID)
echo "vLLM server started with PID: $VLLM_PID"

# Monitor first vLLM server in background
monitor_process $VLLM_PID "vLLM server (Qwen2.5-3B)" &
MONITOR_PID=$!
PIDS+=($MONITOR_PID)

# Run GenPRM server in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "GenPRM/GenPRM-7B" --port 8081 > vllm-serve-prm-code.log 2>&1 &
GENPRM_PID=$!
PIDS+=($GENPRM_PID)
echo "GenPRM server started with PID: $GENPRM_PID"

# Monitor GenPRM server in background
monitor_process $GENPRM_PID "GenPRM server" &
MONITOR_PID2=$!
PIDS+=($MONITOR_PID2)


# Run accelerate launch in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info\
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=6 \
    src/open_r1/decay_intuitor_prm.py --config recipes/Qwen2.5-3B/intuitor-critique/config_code_only_genprm.yaml --wandb_project open-r1 --run_name Qwen2.5-3B-PRM-CODE > run_intuitor-prm-code_300.log 2>&1 &
TRAINING_PID=$!
PIDS+=($TRAINING_PID)
echo "Training process started with PID: $TRAINING_PID"

echo "All processes started in the background. Check log files for output."
echo "Monitoring all processes for failures..."

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Remove trap to avoid double cleanup
trap - EXIT ERR INT TERM

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully. Cleaning up processes..."
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE. Cleaning up processes..."
fi

# Clean up all processes
cleanup