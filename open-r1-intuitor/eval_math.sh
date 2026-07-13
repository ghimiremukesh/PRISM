# 3B

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL=/fsx/mghmr/PRISM/open-r1-intuitor/data/Qwen2.5-3B-MajorityVote-300steps
# export HF_TOKEN=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
export MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=2,max_model_length=32768,generation_parameters={max_new_tokens:4096,temperature:0}"
export OUTPUT_DIR=./results/Qwen2.5-3B-MajorityVote-300steps/
export TASKA=math_500
export TASKB=minerva_math
export TASKC=gsm8k
export N=0
lighteval vllm $MODEL_ARGS "lighteval|$TASKA|$N|0,custom|$TASKB|$N|0,lighteval|$TASKC|$N|0" \
    --use-chat-template \
    --custom-tasks open_r1.minerva_math_task \
    --output-dir $OUTPUT_DIR \
    --save-details



# 7B

# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# # compile cache gets corrupted by concurrent Ray workers (JSONDecodeError at KV-cache init)
# export VLLM_DISABLE_COMPILE_CACHE=1
# export MODEL=/fsx/mghmr/PRISM/open-r1-intuitor/data/Qwen2.5-7B-MajorityVote-DAPO
# # export HF_TOKEN=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
# # gpu_memory_utilization=0.8 OOMs during vLLM profiling with the 7B model on A100-40GB
# export MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.7,data_parallel_size=4,max_model_length=32768,generation_parameters={max_new_tokens:4096,temperature:0}"
# export OUTPUT_DIR=./results/Qwen2.5-7B-MajorityVote-DAPO/
# export TASKA=math_500
# export TASKB=minerva_math
# export TASKC=gsm8k
# export N=0
# lighteval vllm $MODEL_ARGS "lighteval|$TASKA|$N|0,custom|$TASKB|$N|0,lighteval|$TASKC|$N|0" \
#     --use-chat-template \
#     --custom-tasks open_r1.minerva_math_task \
#     --output-dir $OUTPUT_DIR \
#     --save-details
