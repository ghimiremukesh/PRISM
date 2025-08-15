# """
# Generate *draft*, *critique*, and (optional) *revision* responses for a
# subset of any HF dataset and save a CSV with accuracy metrics. Defaults target 
# the **openai/gsm8k** `test` split with 30 random problems, but the script works for
# any dataset that has at least a question/answer field.

# This version supports multi-GPU inference using PyTorch DataParallel or DistributedDataParallel.

# CSV columns
# -----------
# * `id`                – row index or provided id
# * `problem`/`question` – source text
# * `answer`            – ground-truth answer from dataset
# * `initial_response`  – model's first reply
# * `final_response`    – revised if `was_revised==1`, else same as draft
# * `was_revised`       – 1 if a revision was actually generated
# * `critique`          – self-critique text
# * `initial_accuracy`  – 1 if initial response is correct, 0 otherwise
# * `final_accuracy`    – 1 if final response is correct, 0 otherwise
# * `improved_accuracy` – 1 if final > initial accuracy, 0 otherwise

# Accuracy calculation is performed using exact answer matching and numerical equivalence.

# Example (gsm8k)
# ~~~~~~~~~~~~~~~
# ```bash
# # Single GPU
# python evaluate_dataset.py \
#   --model Qwen/Qwen2.5-3B \
#   --hf_dataset openai/gsm8k \
#   --split test \
#   --num_samples 30 \
#   --tb_logdir runs/gsm8k_logs \
#   --output_csv gsm8k_qwen.csv

# # Multi-GPU with DataParallel
# python evaluate_dataset.py \
#   --model Qwen/Qwen2.5-3B \
#   --hf_dataset openai/gsm8k \
#   --split test \
#   --num_samples 30 \
#   --tb_logdir runs/gsm8k_logs \
#   --output_csv gsm8k_qwen.csv \
#   --multi_gpu

# # Multi-GPU with DistributedDataParallel (recommended for better performance)
# torchrun --nproc_per_node=4 evaluate_dataset.py \
#   --model Qwen/Qwen2.5-3B \
#   --hf_dataset openai/gsm8k \
#   --split test \
#   --num_samples 30 \
#   --tb_logdir runs/gsm8k_logs \
#   --output_csv gsm8k_qwen.csv \
#   --distributed
# ```
# """
# from __future__ import annotations

import argparse
import datetime
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from GenPRM.genprm_inference import GenPRM
from GenPRM.util import timestamped_print
from tqdm import tqdm
import ipdb

# # ---------------------------------------------------------------------------
# # Answer Extraction and Accuracy Utilities
# # ---------------------------------------------------------------------------

# def extract_boxed_answer(text: str) -> str:
#     """Extract answer from \\boxed{...} format."""
#     pattern = r'\\boxed\{([^}]*)\}'
#     matches = re.findall(pattern, text)
#     return matches[-1] if matches else ""

# def extract_numerical_answer(text: str) -> str:
#     """Extract the last number from text as a fallback."""
#     numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
#     return numbers[-1] if numbers else ""

# def normalize_answer(answer: str) -> str:
#     """Normalize answer for comparison."""
#     if not answer:
#         return ""
    
#     answer = answer.strip().lower()
#     answer = re.sub(r'[,$\s]', '', answer)
    
#     if '/' in answer:
#         try:
#             parts = answer.split('/')
#             if len(parts) == 2:
#                 num, den = float(parts[0]), float(parts[1])
#                 if den != 0:
#                     answer = str(num / den)
#         except:
#             pass
    
#     try:
#         return str(float(answer))
#     except:
#         return answer

# def extract_answer(response: str, dataset_name: str = "") -> str:
#     """Extract answer from model response based on dataset conventions."""
#     boxed = extract_boxed_answer(response)
#     if boxed:
#         return boxed
    
#     answer_patterns = [
#         r'(?:the answer is|answer:|final answer:)\s*([^\n.]*)',
#         r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*([^\n.]*)',
#     ]
    
#     for pattern in answer_patterns:
#         matches = re.findall(pattern, response, re.IGNORECASE)
#         if matches:
#             return matches[-1].strip()
    
#     return extract_numerical_answer(response)

# def build_chat(messages: List[Dict[str, str]], tok: AutoTokenizer) -> str:
#     """Format chat messages into a single prompt string."""
#     if hasattr(tok, "apply_chat_template"):
#         return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

# def compute_accuracy(predicted: str, ground_truth: str, dataset_name: str = "") -> bool:
#     """Compute accuracy between predicted and ground truth answers."""
#     if not predicted or not ground_truth:
#         return False
    
#     pred_norm = normalize_answer(predicted)
#     gt_norm = normalize_answer(ground_truth)
    
#     if pred_norm == gt_norm:
#         return True
    
#     try:
#         pred_float = float(pred_norm)
#         gt_float = float(gt_norm)
#         return abs(pred_float - gt_float) < 1e-6
#     except:
#         pass
    
#     return pred_norm in gt_norm or gt_norm in pred_norm

# # ---------------------------------------------------------------------------
# # Multi-GPU Utilities
# # ---------------------------------------------------------------------------

# def setup_distributed(rank: int, world_size: int):
#     """Initialize distributed training."""
#     os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
#     os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
#     print(f"[Rank {rank}] Initializing process group with world_size={world_size}")
#     dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
#     print(f"[Rank {rank}] Process group initialized successfully")

# def cleanup_distributed():
#     """Clean up distributed training."""
#     if dist.is_initialized():
#         dist.destroy_process_group()

# def load_model_multi_gpu(model_name: str, multi_gpu: bool = False, distributed: bool = False, 
#                         local_rank: int = -1) -> Tuple[nn.Module, AutoTokenizer]:
#     """Load model with multi-GPU support."""
#     print(f"Loading tokenizer from {model_name}...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     print("Tokenizer loaded successfully")
    
#     if distributed:
#         # For distributed training, load model on specific GPU
#         device = torch.device(f"cuda:{local_rank}")
#         print(f"[Rank {local_rank}] Loading model on {device}...")
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16,
#             device_map=None,  # Don't use auto device map with DDP
#             low_cpu_mem_usage=True
#         ).to(device)
#         print(f"[Rank {local_rank}] Model loaded, wrapping with DDP...")
#         model = DistributedDataParallel(model, device_ids=[local_rank])
#         print(f"[Rank {local_rank}] DDP wrapper applied successfully")
#     elif multi_gpu and torch.cuda.device_count() > 1:
#         # Use DataParallel for multi-GPU
#         print(f"Loading model for DataParallel on {torch.cuda.device_count()} GPUs...")
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16,
#             device_map=None,
#             low_cpu_mem_usage=True
#         )
#         print("Wrapping model with DataParallel...")
#         model = DataParallel(model)
#         model = model.cuda()
#         print("DataParallel wrapper applied successfully")
#     else:
#         # Single GPU or CPU
#         print(f"Loading model on single device...")
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             device_map="auto", 
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True
#         )
#         print("Model loaded successfully")
    
#     return model, tokenizer

# def generate_batch_single_gpu(
#     model: nn.Module,
#     tokenizer: AutoTokenizer,
#     prompts: List[str],
#     max_new_tokens: int,
#     temperature: float,
#     batch_size: int
# ) -> List[str]:
#     """Generate text for single GPU setup."""
#     model.eval()
#     results = []
    
#     for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
#         batch_prompts = prompts[i:i+batch_size]
        
#         inputs = tokenizer(
#             batch_prompts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=2048
#         )
        
#         device = next(model.parameters()).device
#         input_ids = inputs.input_ids.to(device)
#         attention_mask = inputs.attention_mask.to(device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=temperature > 0.0,
#                 temperature=temperature if temperature > 0.0 else 1.0,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
        
#         for j, output in enumerate(outputs):
#             input_length = input_ids[j].shape[0]
#             generated_tokens = output[input_length:]
#             generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#             results.append(generated_text.strip())
    
#     return results

# def generate_batch_multi_gpu(
#     model: nn.Module,
#     tokenizer: AutoTokenizer,
#     prompts: List[str],
#     max_new_tokens: int,
#     temperature: float,
#     batch_size: int,
#     distributed: bool = False,
#     local_rank: int = -1,
# ) -> List[str]:
#     """Generate text using multi-GPU setup with proper distributed handling."""
#     if not distributed:
#         # Use single GPU generation for DataParallel
#         return generate_batch_single_gpu(model, tokenizer, prompts, max_new_tokens, temperature, batch_size)
    
#     # Distributed generation
#     model.eval()
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     # Calculate which prompts this rank should process
#     prompts_per_rank = len(prompts) // world_size
#     extra_prompts = len(prompts) % world_size
    
#     start_idx = rank * prompts_per_rank + min(rank, extra_prompts)
#     end_idx = start_idx + prompts_per_rank + (1 if rank < extra_prompts else 0)
    
#     local_prompts = prompts[start_idx:end_idx]
#     local_indices = list(range(start_idx, end_idx))
    
#     print(f"[Rank {rank}] Processing prompts {start_idx} to {end_idx} ({len(local_prompts)} prompts)")
    
#     # Generate for local prompts
#     local_results = []
#     device = torch.device(f"cuda:{local_rank}")
    
#     # Get base model from DDP wrapper
#     base_model = model.module if isinstance(model, DistributedDataParallel) else model
    
#     for i in tqdm(range(0, len(local_prompts), batch_size), 
#                   desc=f"Rank {rank} Generating", 
#                   disable=rank != 0):
#         batch_prompts = local_prompts[i:i+batch_size]
        
#         inputs = tokenizer(
#             batch_prompts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=2048
#         )
        
#         input_ids = inputs.input_ids.to(device)
#         attention_mask = inputs.attention_mask.to(device)
        
#         with torch.no_grad():
#             outputs = base_model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=temperature > 0.0,
#                 temperature=temperature if temperature > 0.0 else 1.0,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
        
#         for j, output in enumerate(outputs):
#             input_length = input_ids[j].shape[0]
#             generated_tokens = output[input_length:]
#             generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
#             # Store with original index
#             orig_idx = local_indices[i + j]
#             local_results.append((orig_idx, generated_text.strip()))
    
#     # Gather results from all ranks
#     all_results = [None] * world_size
#     dist.all_gather_object(all_results, local_results)
    
#     # Combine and sort results
#     combined_results = []
#     for rank_results in all_results:
#         if rank_results:
#             combined_results.extend(rank_results)
    
#     combined_results.sort(key=lambda x: x[0])
#     return [text for _, text in combined_results]

# def evaluate(
#     model_name: str,
#     hf_dataset: str,
#     split: str,
#     out_csv: str,
#     *,
#     num_samples: int,
#     seed: int,
#     batch_size: int,
#     max_new_tokens: int,
#     temperature: float,
#     device: str,
#     tb_logdir: Optional[str] = None,
#     multi_gpu: bool = False,
#     distributed: bool = False,
#     local_rank: int = -1,
# ):
#     """Efficient distributed evaluation with proper data handling."""
    
#     # Initialize distributed if needed
#     if distributed:
#         world_size = dist.get_world_size()
#         rank = dist.get_rank()
#         print(f"[Rank {rank}/{world_size}] Starting evaluation...")
#         torch.cuda.set_device(local_rank)
#     else:
#         print("Starting evaluation...")
#         rank = 0
#         world_size = 1
    
#     # Load dataset (all ranks load the same dataset for consistency)
#     print(f"Loading dataset {hf_dataset}/{split}...")
#     ds = load_dataset(hf_dataset, split=split)
#     if 0 < num_samples < len(ds):
#         ds = ds.shuffle(seed=seed).select(range(num_samples))
#     print(f"Dataset loaded: {len(ds)} samples")
    
#     # Load model with multi-GPU support
#     print(f"Loading model...")
#     model, tok = load_model_multi_gpu(model_name, multi_gpu, distributed, local_rank)
#     print(f"Model loading complete!")
    
#     # Synchronize all processes
#     if distributed:
#         dist.barrier()
    
#     # TensorBoard writer (only on main process)
#     writer = None
#     if tb_logdir and rank == 0:
#         writer = SummaryWriter(tb_logdir)
    
#     system_prompt = (
#         "You are a helpful AI assistant. FIRST think step-by-step, show all your work step-by-step, "
#         "then provide the answer in the form \\boxed{…}."
#     )
    
#     # ---------- Pass 1: Draft Generation ----------
#     chats: List[List[Dict[str, str]]] = []
#     for row in ds:
#         problem_text = (
#             row.get("problem") or row.get("question") or row.get("prompt") or str(row)
#         )
#         chats.append([
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": problem_text.strip()},
#         ])
    
#     # Generate drafts
#     draft_prompts = [build_chat(c, tok) for c in chats]

#     # add step prefix
#     # ipdb.set_trace()
#     # draft_prompts = [d+'Step 1:' for d in draft_prompts]

#     print(f"Generating {len(draft_prompts)} drafts...")
    
#     if multi_gpu or distributed:
#         drafts = generate_batch_multi_gpu(
#             model, tok, draft_prompts, max_new_tokens, temperature,
#             batch_size, distributed, local_rank
#         )
#     else:
#         drafts = generate_batch_single_gpu(
#             model, tok, draft_prompts, max_new_tokens, temperature, batch_size
#         )
    
#     print(f"Generated {len(drafts)} drafts")

#     # can delete the model
#     # del model
#     # torch.cuda.empty_cache()
#     # torch.cuda.ipc_collect()

#     # dist.barrier()
#     # dist.destroy_process_group()
#     # if rank != 0:
#     #     import sys
#     #     sys.exit(0)

# #     for var in (
# #     "RANK", "LOCAL_RANK", "WORLD_SIZE",
# #     "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
# #     "TORCHELASTIC_RUN_ID",
# # ):
# #         os.environ.pop(var, None)

#     # ipdb.set_trace()
#     # load genprm model
#     # just use 1 gpu
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     # torch.cuda.set_device(0)


#     genprm = GenPRM('GenPRM/GenPRM-7B', 7)


    
#     # ipdb.set_trace()
#     # Update chats with drafts

#     original_drafts = drafts.copy()

#     ## split draft by paragraph as steps
#     drafts = [draft.split('\n\n') for draft in drafts]
#     for chat, draft in zip(chats, drafts):
#         for para in draft:
#             # replace the system prompt with a new one for verification
#             chat[0][0] = {"content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should neither solve the problem nor give the final answer.", "role": "system" }
#             chat.append({"role": "user", "content": para})
#             chat.append({"role": "assistant", "content": ''})
    
    
#     # now chat is a huge list of (questions, answers separated by paragraph)
#     all_reward_list = []
#     all_process_rewards = []
#     with tqdm(total=len(chats), desc='Generating Process Rewards') as pbar:
#         for sol in chats:
#             curr_reward = []
#             for i in range(len(sol)):
#                 if sol[i]['role'] != 'assistant':
#                     continue
#                 # ipdb.set_trace()
#                 output, reward = genprm.inference(sol[:i], cur_step=int(i/2), verify=False, logging=False)
#                 sol[i]['content'] = output[0]
#                 curr_reward.append(reward)
            
#             # store all process rewards and compute total reward
#             all_process_rewards.append(curr_reward)
#             all_reward_list.append(sum(curr_reward)/len(curr_reward))
        
#         pbar.update(1)

#     # ipdb.set_trace()
#     # ---------- Compute Metrics and Save CSV ----------
#     # Only rank 0 computes metrics and saves CSV in distributed mode
#     if rank == 0 or not distributed:
#         rows = []
#         dataset_name = hf_dataset.split("/")[-1]
        
#         print(f"Computing metrics for {len(drafts)} samples...")
        
#         for i, (row_data, draft, draft_prompt, process_reward) in enumerate(zip(ds, original_drafts, draft_prompts, all_reward_list)):
#             answer_text = row_data.get("answer") or row_data.get("solution") or ""
#             problem_text = (
#                 row_data.get("problem") or row_data.get("question") or row_data.get("prompt") or ""
#             )
#             initial_pred = extract_answer(draft, dataset_name)
            
#             initial_accuracy = compute_accuracy(initial_pred, str(answer_text), dataset_name)


#             # ipdb.set_trace()
#             row = {
#                 "id": row_data.get("id", i),
#                 "problem": problem_text,
#                 "full_initial_prompt": draft_prompt,
#                 "answer": answer_text,
#                 "initial_response": draft,
#                 "initial_accuracy": int(initial_accuracy),
#                 "process_reward": process_reward,
#             }
#             rows.append(row)
        
#         print("Saving results and computing final statistics...")
        
#         # Save CSV
#         df = pd.DataFrame(rows)
#         df.to_csv(out_csv, index=False)
        
#         # Compute aggregate statistics
#         total_samples = len(rows)
#         initial_correct = sum(row["initial_accuracy"] for row in rows)

        
#         initial_acc = initial_correct / total_samples if total_samples > 0 else 0


        
#         print(f"\nResults Summary:")
#         print(f"Saved {total_samples} rows to {out_csv} (dataset: {hf_dataset}/{split})")
#         print(f"Initial Accuracy: {initial_acc:.3f} ({initial_correct}/{total_samples})")

    
#     # Synchronize before cleanup
#     if distributed:
#         dist.barrier()
#         print(f"[Rank {rank}] Evaluation complete")

# # ---------------------------------------------------------------------------
# # CLI entry point
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser("Generate self-critique outputs with accuracy metrics")
#     parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
#     parser.add_argument("--hf_dataset", default="HuggingFaceH4/MATH-500")
#     parser.add_argument("--split", default="test")
#     parser.add_argument("--num_samples", type=int, default=32)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--output_csv", default="genprm_test.csv")
#     parser.add_argument("--tb_logdir", default=None, help="TensorBoard logdir (optional)")
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--max_new_tokens", type=int, default=4096)
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--device", default="auto")
    
#     # Multi-GPU options
#     parser.add_argument("--multi_gpu", action="store_true", 
#                        help="Use DataParallel for multi-GPU inference")
#     parser.add_argument("--distributed", action="store_true",
#                        help="Use DistributedDataParallel (launch with torchrun)")
#     parser.add_argument("--local_rank", type=int, default=-1,
#                        help="Local rank for distributed training (set automatically by torchrun)")
    
#     args = parser.parse_args()
    
#     # Setup distributed if needed
#     if args.distributed:
#         if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#             rank = int(os.environ["RANK"])
#             world_size = int(os.environ["WORLD_SIZE"])
#             local_rank = int(os.environ["LOCAL_RANK"])
#             args.local_rank = local_rank
#             setup_distributed(rank, world_size)
#         else:
#             print("WARNING: --distributed flag set but not launched with torchrun. Falling back to single GPU.")
#             args.distributed = False
    
#     # Check for multi-GPU availability
#     if args.multi_gpu and torch.cuda.device_count() <= 1:
#         print(f"WARNING: --multi_gpu flag set but only {torch.cuda.device_count()} GPU(s) available. Using single GPU.")
#         args.multi_gpu = False
    
#     if args.multi_gpu or args.distributed:
#         print(f"Using {'Distributed' if args.distributed else 'DataParallel'} with {torch.cuda.device_count()} GPUs")
    
#     try:
#         evaluate(
#             model_name=args.model,
#             hf_dataset=args.hf_dataset,
#             split=args.split,
#             out_csv=args.output_csv,
#             num_samples=args.num_samples,
#             seed=args.seed,
#             batch_size=args.batch_size,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             device=args.device,
#             tb_logdir=args.tb_logdir,
#             multi_gpu=args.multi_gpu,
#             distributed=args.distributed,
#             local_rank=args.local_rank,
#         )
#     finally:
#         if args.distributed:
#             cleanup_distributed()

# if __name__ == "__main__":
#     main()


# load data
# generation_data = pd.read_csv('outputs_math_500-Q2.5-3B-Ins.csv')[:1]

# problems = generation_data['problem']

generation_data = pd.read_json('sc_base_gsm8k.json')

problems = generation_data['example']

chats = []
for problem in problems:
    chats.append([
        {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should neither solve the problem nor give the final answer."},
        {"role": "user", "content": problem}
    ])

# drafts = generation_data['initial_response']
drafts = generation_data['predictions']



## split draft by paragraph as steps
drafts = [draft[0].split('\n\n') for draft in drafts]
for chat, draft in zip(chats, drafts):
    for para in draft:
        # replace the system prompt with a new one for verification
        # chat[0][0] = {"content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should neither solve the problem nor give the final answer.", "role": "system" }
        chat.append({"role": "user", "content": para})
    chat.append({"role": "assistant", "content": ''})


# load model

# ipdb.set_trace()

genprm = GenPRM('GenPRM/GenPRM-7B', 1)



# now chat is a huge list of (questions, answers separated by paragraph)
all_reward_list = []
all_process_rewards = []
prm_outputs = []
count = 0
for sol in chats:
    curr_reward = []
    curr_out = []
    for i in range(len(sol)):
        if sol[i]['role'] != 'assistant':
            continue
        # ipdb.set_trace()
        output, reward = genprm.inference(sol[:i], majority_num=1, cur_step=int(i/2), verify=False, execute=False)
        sol[i]['content'] = output[0]
        curr_reward.append(reward)
        curr_out.append(output[0])
    
    count += 1
    # store all process rewards and compute total reward
    all_process_rewards.append(curr_reward)
    all_reward_list.append(sum(curr_reward)/len(curr_reward))
    prm_outputs.append(curr_out)

    timestamped_print(f"{count} solution(s) graded.", level="INFO")    

# ipdb.set_trace()

generation_data['process_reward'] = all_reward_list
generation_data['prm_output'] = prm_outputs

generation_data.to_csv('genprm_base_rewards.csv')

