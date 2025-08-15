"""
Generate *draft*, *critique*, and (optional) *revision* responses for a
subset of any HF dataset and save a CSV with accuracy metrics. Defaults target 
the **openai/gsm8k** `test` split with 30 random problems, but the script works for
any dataset that has at least a question/answer field.

This version supports multi-GPU inference using PyTorch DataParallel or DistributedDataParallel.

CSV columns
-----------
* `id`                – row index or provided id
* `problem`/`question` – source text
* `answer`            – ground-truth answer from dataset
* `initial_response`  – model's first reply
* `final_response`    – revised if `was_revised==1`, else same as draft
* `was_revised`       – 1 if a revision was actually generated
* `critique`          – self-critique text
* `initial_accuracy`  – 1 if initial response is correct, 0 otherwise
* `final_accuracy`    – 1 if final response is correct, 0 otherwise
* `improved_accuracy` – 1 if final > initial accuracy, 0 otherwise

Accuracy calculation is performed using exact answer matching and numerical equivalence.

Example (gsm8k)
~~~~~~~~~~~~~~~
```bash
# Single GPU
python evaluate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --hf_dataset openai/gsm8k \
  --split test \
  --num_samples 30 \
  --tb_logdir runs/gsm8k_logs \
  --output_csv gsm8k_qwen.csv

# Multi-GPU with DataParallel
python evaluate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --hf_dataset openai/gsm8k \
  --split test \
  --num_samples 30 \
  --tb_logdir runs/gsm8k_logs \
  --output_csv gsm8k_qwen.csv \
  --multi_gpu

# Multi-GPU with DistributedDataParallel (recommended for better performance)
torchrun --nproc_per_node=4 evaluate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --hf_dataset openai/gsm8k \
  --split test \
  --num_samples 30 \
  --tb_logdir runs/gsm8k_logs \
  --output_csv gsm8k_qwen.csv \
  --distributed
```
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Answer Extraction and Accuracy Utilities
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

def extract_numerical_answer(text: str) -> str:
    """Extract the last number from text as a fallback."""
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    return numbers[-1] if numbers else ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    
    answer = answer.strip().lower()
    answer = re.sub(r'[,$\s]', '', answer)
    
    if '/' in answer:
        try:
            parts = answer.split('/')
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                if den != 0:
                    answer = str(num / den)
        except:
            pass
    
    try:
        return str(float(answer))
    except:
        return answer

def extract_answer(response: str, dataset_name: str = "") -> str:
    """Extract answer from model response based on dataset conventions."""
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed
    
    answer_patterns = [
        r'(?:the answer is|answer:|final answer:)\s*([^\n.]*)',
        r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*([^\n.]*)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    return extract_numerical_answer(response)

def build_chat(messages: List[Dict[str, str]], tok: AutoTokenizer) -> str:
    """Format chat messages into a single prompt string."""
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

def compute_accuracy(predicted: str, ground_truth: str, dataset_name: str = "") -> bool:
    """Compute accuracy between predicted and ground truth answers."""
    if not predicted or not ground_truth:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
    except:
        pass
    
    return pred_norm in gt_norm or gt_norm in pred_norm

# ---------------------------------------------------------------------------
# Multi-GPU Utilities
# ---------------------------------------------------------------------------

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    print(f"[Rank {rank}] Initializing process group with world_size={world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
    print(f"[Rank {rank}] Process group initialized successfully")

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_model_multi_gpu(model_name: str, multi_gpu: bool = False, distributed: bool = False, 
                        local_rank: int = -1) -> Tuple[nn.Module, AutoTokenizer]:
    """Load model with multi-GPU support."""
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully")
    
    if distributed:
        # For distributed training, load model on specific GPU
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {local_rank}] Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=None,  # Don't use auto device map with DDP
            low_cpu_mem_usage=True
        ).to(device)
        print(f"[Rank {local_rank}] Model loaded, wrapping with DDP...")
        model = DistributedDataParallel(model, device_ids=[local_rank])
        print(f"[Rank {local_rank}] DDP wrapper applied successfully")
    elif multi_gpu and torch.cuda.device_count() > 1:
        # Use DataParallel for multi-GPU
        print(f"Loading model for DataParallel on {torch.cuda.device_count()} GPUs...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True
        )
        print("Wrapping model with DataParallel...")
        model = DataParallel(model)
        model = model.cuda()
        print("DataParallel wrapper applied successfully")
    else:
        # Single GPU or CPU
        print(f"Loading model on single device...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully")
    
    return model, tokenizer

def generate_batch_single_gpu(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int
) -> List[str]:
    """Generate text for single GPU setup."""
    model.eval()
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        )
        
        device = next(model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        for j, output in enumerate(outputs):
            input_length = input_ids[j].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text.strip())
    
    return results

def generate_batch_multi_gpu(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    distributed: bool = False,
    local_rank: int = -1
) -> List[str]:
    """Generate text using multi-GPU setup with proper distributed handling."""
    if not distributed:
        # Use single GPU generation for DataParallel
        return generate_batch_single_gpu(model, tokenizer, prompts, max_new_tokens, temperature, batch_size)
    
    # Distributed generation
    model.eval()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Calculate which prompts this rank should process
    prompts_per_rank = len(prompts) // world_size
    extra_prompts = len(prompts) % world_size
    
    start_idx = rank * prompts_per_rank + min(rank, extra_prompts)
    end_idx = start_idx + prompts_per_rank + (1 if rank < extra_prompts else 0)
    
    local_prompts = prompts[start_idx:end_idx]
    local_indices = list(range(start_idx, end_idx))
    
    print(f"[Rank {rank}] Processing prompts {start_idx} to {end_idx} ({len(local_prompts)} prompts)")
    
    # Generate for local prompts
    local_results = []
    device = torch.device(f"cuda:{local_rank}")
    
    # Get base model from DDP wrapper
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    
    for i in tqdm(range(0, len(local_prompts), batch_size), 
                  desc=f"Rank {rank} Generating", 
                  disable=rank != 0):
        batch_prompts = local_prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        for j, output in enumerate(outputs):
            input_length = input_ids[j].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Store with original index
            orig_idx = local_indices[i + j]
            local_results.append((orig_idx, generated_text.strip()))
    
    # Gather results from all ranks
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    
    # Combine and sort results
    combined_results = []
    for rank_results in all_results:
        if rank_results:
            combined_results.extend(rank_results)
    
    combined_results.sort(key=lambda x: x[0])
    return [text for _, text in combined_results]

def evaluate(
    model_name: str,
    hf_dataset: str,
    split: str,
    out_csv: str,
    *,
    num_samples: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    tb_logdir: Optional[str] = None,
    multi_gpu: bool = False,
    distributed: bool = False,
    local_rank: int = -1,
):
    """Efficient distributed evaluation with proper data handling."""
    
    # Initialize distributed if needed
    if distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"[Rank {rank}/{world_size}] Starting evaluation...")
        torch.cuda.set_device(local_rank)
    else:
        print("Starting evaluation...")
        rank = 0
        world_size = 1
    
    # Load dataset (all ranks load the same dataset for consistency)
    print(f"Loading dataset {hf_dataset}/{split}...")
    ds = load_dataset(hf_dataset, split=split)
    if 0 < num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Load model with multi-GPU support
    print(f"Loading model...")
    model, tok = load_model_multi_gpu(model_name, multi_gpu, distributed, local_rank)
    print(f"Model loading complete!")
    
    # Synchronize all processes
    if distributed:
        dist.barrier()
    
    # TensorBoard writer (only on main process)
    writer = None
    if tb_logdir and rank == 0:
        writer = SummaryWriter(tb_logdir)
    
    system_prompt = (
        "You are a helpful AI assistant. FIRST think step-by-step, show all your work, "
        "then provide the answer in the form \\boxed{…}."
    )
    
    # ---------- Pass 1: Draft Generation ----------
    chats: List[List[Dict[str, str]]] = []
    for row in ds:
        problem_text = (
            row.get("problem") or row.get("question") or row.get("prompt") or str(row)
        )
        chats.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text.strip()},
        ])
    
    # Generate drafts
    draft_prompts = [build_chat(c, tok) for c in chats]
    print(f"Generating {len(draft_prompts)} drafts...")
    
    if multi_gpu or distributed:
        drafts = generate_batch_multi_gpu(
            model, tok, draft_prompts, max_new_tokens, temperature, 
            batch_size, distributed, local_rank
        )
    else:
        drafts = generate_batch_single_gpu(
            model, tok, draft_prompts, max_new_tokens, temperature, batch_size
        )
    
    print(f"Generated {len(drafts)} drafts")
    
    # Update chats with drafts
    for chat, draft in zip(chats, drafts):
        chat.append({"role": "assistant", "content": draft})
    
    # ---------- Pass 2: Critique ----------
    crit_prompt = (
        "Check the math solution step-by-step. If you find a mistake: state the wrong step, "
        "explain why it's wrong, and end your response with 'The answer is wrong'. "
        "If all steps are correct, end with 'The answer is correct'. "
        "If you are unsure or lack knowledge on a particular topic, elaborate and then end with 'I am unsure'. "
        "YOU MUST END WITH EITHER 'The answer is correct' OR 'The answer is wrong' OR 'I am unsure'."
    )
    
    for chat in chats:
        chat.append({"role": "user", "content": crit_prompt})
    
    # Generate critiques
    critique_prompts = [build_chat(c, tok) for c in chats]
    print(f"Generating {len(critique_prompts)} critiques...")
    
    if multi_gpu or distributed:
        critiques = generate_batch_multi_gpu(
            model, tok, critique_prompts, max_new_tokens, temperature,
            batch_size, distributed, local_rank
        )
    else:
        critiques = generate_batch_single_gpu(
            model, tok, critique_prompts, max_new_tokens, temperature, batch_size
        )
    
    print(f"Generated {len(critiques)} critiques")
    
    # Update chats with critiques
    for chat, crit in zip(chats, critiques):
        chat.append({"role": "assistant", "content": crit})
    
    # Determine which responses need revision
    needs_revision = []
    for crit in critiques:
        needs_revision.append("the answer is wrong" in crit.lower() or "the answer is incorrect" in crit.lower())

    needs_revision = [False] * len(chats)  # make it false for now. 
    
    # ---------- Pass 3: Optional Revision ----------
    revisions: List[Optional[str]] = [None] * len(chats)
    rev_indices = [i for i, flag in enumerate(needs_revision) if flag]

    store_rev_prompts = [None] * len(chats)
    
    if rev_indices:
    # if any(needs_revision):
        print(f"Processing {len(rev_indices)} revisions...")
        # print(f"Processing {sum(needs_revision)} revisions...")
        rev_prompt = (
            "You indicated that your previous answer was wrong. Based on your evaluation, "
            "please provide the correct step-by-step solution to the math problem. "
            "Make sure the answer is in a box: \\boxed{Your Answer}. Stop immediately after the box."
        )
        
        # Create revision chats
        rev_chats = []
        for idx in rev_indices:
            rev_chat = chats[idx].copy()
            rev_chat.append({"role": "user", "content": rev_prompt})
            rev_chats.append(rev_chat)
        
        rev_prompts = [build_chat(c, tok) for c in rev_chats]
        
        if multi_gpu or distributed:
            rev_texts = generate_batch_multi_gpu(
                model, tok, rev_prompts, max_new_tokens, temperature,
                batch_size, distributed, local_rank
            )
        else:
            rev_texts = generate_batch_single_gpu(
                model, tok, rev_prompts, max_new_tokens, temperature, batch_size
            )
        
        for idx, txt, prompt in zip(rev_indices, rev_texts, rev_prompts):
            revisions[idx] = txt
            store_rev_prompts[idx] = prompt
    
    # Create final responses
    finals = []
    for i in range(len(drafts)):
        if revisions[i] is not None:
            finals.append(revisions[i])
        else:
            finals.append(drafts[i])
    
    print(f"Created {len(finals)} final responses")
    
    # ---------- Compute Metrics and Save CSV ----------
    # Only rank 0 computes metrics and saves CSV in distributed mode
    if rank == 0 or not distributed:
        rows = []
        dataset_name = hf_dataset.split("/")[-1]
        
        print(f"Computing metrics for {len(drafts)} samples...")
        
        for i, (row_data, draft, final, crit, draft_prompt, rev_prompt, crit_prompt) in enumerate(zip(ds, drafts, finals, critiques, draft_prompts, store_rev_prompts, critique_prompts)):
            answer_text = row_data.get("answer") or row_data.get("solution") or ""
            problem_text = (
                row_data.get("problem") or row_data.get("question") or row_data.get("prompt") or ""
            )
            initial_pred = extract_answer(draft, dataset_name)
            final_pred = extract_answer(final, dataset_name)
            
            initial_accuracy = compute_accuracy(initial_pred, str(answer_text), dataset_name)
            final_accuracy = compute_accuracy(final_pred, str(answer_text), dataset_name)
            improved_accuracy = int(final_accuracy and not initial_accuracy)
            #opposite of improved accuracy
            correct_to_wrong = int(initial_accuracy and not final_accuracy)
            
            row = {
                "id": row_data.get("id", i),
                "problem": problem_text,
                "full_initial_prompt": draft_prompt,
                "answer": answer_text,
                "initial_response": draft,
                "final_response": final,
                "was_revised": int(revisions[i] is not None),
                "critique": crit,
                "critique_prompt": crit_prompt,
                "revision_prompt": rev_prompt,
                "initial_accuracy": int(initial_accuracy),
                "final_accuracy": int(final_accuracy),
                "improved_accuracy": improved_accuracy,
                "correct_to_wrong": correct_to_wrong,
            }
            rows.append(row)
        
        print("Saving results and computing final statistics...")
        
        # Save CSV
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        
        # Compute aggregate statistics
        total_samples = len(rows)
        initial_correct = sum(row["initial_accuracy"] for row in rows)
        final_correct = sum(row["final_accuracy"] for row in rows)
        improved = sum(row["improved_accuracy"] for row in rows)

        made_worse = sum(row["correct_to_wrong"] for row in rows)
        
        initial_acc = initial_correct / total_samples if total_samples > 0 else 0
        final_acc = final_correct / total_samples if total_samples > 0 else 0
        improvement_rate = improved / total_samples if total_samples > 0 else 0
        revision_rate = sum(row["was_revised"] for row in rows) / total_samples if total_samples > 0 else 0

        # correct recall: proportion of correct answers that are correctly idenfied by the critique
        initial_correct_idxs = [i for i, row in enumerate(rows) if row["initial_accuracy"]]
        deemed_correct_idxs = [i for i, row in enumerate(rows) if row["initial_accuracy"] and not ("the answer is wrong" in row["critique"].lower() or "the answer is incorrect" in row["critique"].lower())]
        correct_recall = len(deemed_correct_idxs) / len(initial_correct_idxs) if initial_correct_idxs else 0.0

        # wrong recall: proportion of incorrect answers that are correctly identified by critique as wrong
        initial_incorrect_idxs = [i for i, row in enumerate(rows) if not row["initial_accuracy"]]
        deemed_incorrect_idxs = [i for i, row in enumerate(rows) if not row["initial_accuracy"] and ("the answer is wrong" in row["critique"].lower() or "the answer is incorrect" in row["critique"].lower())]
        wrong_recall = len(deemed_incorrect_idxs) / len(initial_incorrect_idxs) if initial_incorrect_idxs else 0.0


        verifier_accuracy = [i for i, row in enumerate(rows) if (("the answer is wrong" in row["critique"].lower() or "the answer is incorrect" in row["critique"].lower()) and not row["initial_accuracy"]) or (("the answer is wrong" not in row["critique"].lower() or "the answer is incorrect" not in row["critique"].lower()) and row["initial_accuracy"])]
        verifier_accuracy = len(verifier_accuracy) / total_samples

        
        print(f"\nResults Summary:")
        print(f"Saved {total_samples} rows to {out_csv} (dataset: {hf_dataset}/{split})")
        print(f"Initial Accuracy: {initial_acc:.3f} ({initial_correct}/{total_samples})")
        print(f"Final Accuracy: {final_acc:.3f} ({final_correct}/{total_samples})")
        print(f"Improvement Rate: {improvement_rate:.3f} ({improved}/{total_samples})")
        print(f"Revision Rate: {revision_rate:.3f}")
        print(f"Verifier Accuracy: {verifier_accuracy:.3f}")        
        print(f"Wrong Recall: {wrong_recall:.3f}")
        print(f"Correct Recall: {correct_recall:.3f}")
        print(f"Revision Error: {made_worse}")
        
        # TensorBoard logging
        if writer:
            for i, row in enumerate(rows):
                writer.add_scalar("accuracy/initial", row["initial_accuracy"], i)
                writer.add_scalar("accuracy/final", row["final_accuracy"], i)
                writer.add_scalar("accuracy/improved", row["improved_accuracy"], i)
            
            writer.add_scalar("summary/initial_accuracy", initial_acc)
            writer.add_scalar("summary/final_accuracy", final_acc)
            writer.add_scalar("summary/improvement_rate", improvement_rate)
            writer.add_scalar("summary/revision_rate", revision_rate)
            print(f"TensorBoard logs written to {tb_logdir}")
            writer.close()
    
    # Synchronize before cleanup
    if distributed:
        dist.barrier()
        print(f"[Rank {rank}] Evaluation complete")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Generate self-critique outputs with accuracy metrics")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--hf_dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="output_Qwen3B_I_3_bins.csv")
    parser.add_argument("--tb_logdir", default=None, help="TensorBoard logdir (optional)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    
    # Multi-GPU options
    parser.add_argument("--multi_gpu", action="store_true", 
                       help="Use DataParallel for multi-GPU inference")
    parser.add_argument("--distributed", action="store_true",
                       help="Use DistributedDataParallel (launch with torchrun)")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training (set automatically by torchrun)")
    
    args = parser.parse_args()
    
    # Setup distributed if needed
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            args.local_rank = local_rank
            setup_distributed(rank, world_size)
        else:
            print("WARNING: --distributed flag set but not launched with torchrun. Falling back to single GPU.")
            args.distributed = False
    
    # Check for multi-GPU availability
    if args.multi_gpu and torch.cuda.device_count() <= 1:
        print(f"WARNING: --multi_gpu flag set but only {torch.cuda.device_count()} GPU(s) available. Using single GPU.")
        args.multi_gpu = False
    
    if args.multi_gpu or args.distributed:
        print(f"Using {'Distributed' if args.distributed else 'DataParallel'} with {torch.cuda.device_count()} GPUs")
    
    try:
        evaluate(
            model_name=args.model,
            hf_dataset=args.hf_dataset,
            split=args.split,
            out_csv=args.output_csv,
            num_samples=args.num_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            tb_logdir=args.tb_logdir,
            multi_gpu=args.multi_gpu,
            distributed=args.distributed,
            local_rank=args.local_rank,
        )
    finally:
        if args.distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main()