#!/usr/bin/bash
source ./openr1/bin/activate
export WANDB_API_KEY=6690ae4dd209ea0517d035d7efd9c428ba921922

# export DATABRICKS_HOST=https://dbc-3eec905e-05ea.cloud.databricks.com
# export DATABRICKS_TOKEN=dapi50c72486dd38244730fce9eb9b87f5d6
# export MLFLOW_TRACKING_URI=databricks
# export MLFLOW_REGISTRY_URI=databricks-uc
# export MLFLOW_EXPERIMENT_ID=1799336443725747
# export ACCELERATE_LOG_LEVEL=info


export HF_TOKEN=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

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
    ps aux | grep "/users/mghmr" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
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
nohup env CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "HuggingFaceTB/SmolLM3-3B" > vllm-serve-prm.log 2>&1 &
VLLM_PID=$!
PIDS+=($VLLM_PID)
echo "vLLM server started with PID: $VLLM_PID"

# Monitor first vLLM server in background
monitor_process $VLLM_PID "vLLM server Base" &
MONITOR_PID=$!
PIDS+=($MONITOR_PID)

# Run GenPRM server in the background with nohup
nohup env CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "GenPRM/GenPRM-7B" --port 8081 > vllm-serve-teacher-prm.log 2>&1 &
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
    src/open_r1/decay_intuitor_prm.py --config recipes/SmolLM3-3B/prm/config_genprm_sc.yaml --wandb_project open-r1 --run_name SmolLM-INT_PRM > run_intuitor-prm.log 2>&1 &
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