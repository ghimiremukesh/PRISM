"""
Generate *draft*, *critique*, and (optional) *revision* responses for a
subset of any HF dataset and save a CSV with accuracy metrics. Defaults target 
the **openai/gsm8k** `test` split with 30 random problems, but the script works for
any dataset that has at least a question/answer field.

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
python evaluate_dataset.py \
  --model Qwen/Qwen2.5-3B \
  --hf_dataset openai/gsm8k \
  --split test \
  --num_samples 30 \
  --tb_logdir runs/gsm8k_logs \
  --output_csv gsm8k_qwen.csv
```
"""
from __future__ import annotations

import argparse
import re
from typing import List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
    # Look for numbers (including decimals and fractions)
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    return numbers[-1] if numbers else ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    
    # Remove common formatting
    answer = answer.strip().lower()
    answer = re.sub(r'[,$\s]', '', answer)
    
    # Handle fractions
    if '/' in answer:
        try:
            parts = answer.split('/')
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                if den != 0:
                    answer = str(num / den)
        except:
            pass
    
    # Try to convert to float for numerical comparison
    try:
        return str(float(answer))
    except:
        return answer

def extract_answer(response: str, dataset_name: str = "") -> str:
    """Extract answer from model response based on dataset conventions."""
    # First try boxed format (common in math datasets)
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed
    
    # For GSM8K and similar, look for "The answer is" pattern
    answer_patterns = [
        r'(?:the answer is|answer:|final answer:)\s*([^\n.]*)',
        r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*([^\n.]*)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # Fallback to last number
    return extract_numerical_answer(response)

def compute_accuracy(predicted: str, ground_truth: str, dataset_name: str = "") -> bool:
    """Compute accuracy between predicted and ground truth answers."""
    if not predicted or not ground_truth:
        return False
    
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match after normalization
    if pred_norm == gt_norm:
        return True
    
    # Numerical tolerance for floating point answers
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
    except:
        pass
    
    # String containment as last resort
    return pred_norm in gt_norm or gt_norm in pred_norm

# ---------------------------------------------------------------------------
# Utilities (unchanged)
# ---------------------------------------------------------------------------

def build_chat(messages: List[dict[str, str]], tok) -> str:
    """Format chat messages into a single prompt string."""
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

def load_pipe(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    # decoder-only models need left padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tok, device_map=device), tok

def batch_generate(pipe, prompts: List[str], *, max_new_tokens: int, temperature: float) -> List[str]:
    """Generate continuations for *prompts* using the HF pipeline."""
    outs = pipe(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        batch_size=len(prompts),
        return_full_text=False,
    )  # List[dict] or List[List[dict]]

    gens: List[str] = []
    for res in outs:
        if isinstance(res, list):  # multiple sequences case
            res = res[0]
        gens.append(res["generated_text"].strip())
    return gens

# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

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
):
    """Run self-critique and write CSV + optional TensorBoard logs with accuracy metrics."""
    # Load and optionally subsample
    ds = load_dataset(hf_dataset, split=split)
    if 0 < num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))

    pipe, tok = load_pipe(model_name, device)

    # TensorBoard writer
    writer = SummaryWriter(tb_logdir) if tb_logdir else None

    system_prompt = (
        "You are a helpful AI assistant. FIRST think step-by-step, show all your work, "
        "then provide the answer in the form \\boxed{…}."
    )

    # ---------- pass 1: draft generation ----------
    chats: List[List[dict[str, str]]] = []
    for row in ds:
        problem_text = (
            row.get("problem") or row.get("question") or row.get("prompt") or str(row)
        )
        chats.append([
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": problem_text.strip()},
        ])

    drafts: List[str] = []
    for i in tqdm(range(0, len(chats), batch_size), desc="Drafts"):
        batch_prompts = [build_chat(c, tok) for c in chats[i : i + batch_size]]
        drafts.extend(
            batch_generate(
                pipe,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
    # for chat, draft in zip(chats, drafts):
    #     chat.append({"role": "assistant", "content": draft})

    # ---------- pass 2: critique ----------
    # crit_prompt = (
    #     "Check the math solution step-by-step. If you find a mistake: state the wrong step, explain why it's wrong, and end your response with 'The answer is wrong'. "
    #     "If all steps are correct, end with 'The answer is correct'. YOU MUST END WITH EITHER 'The answer is correct' OR 'The answer is wrong'. "
    # )
    # for chat in chats:
    #     chat.append({"role": "user", "content": crit_prompt})

    critiques: List[str] = []
    # for i in tqdm(range(0, len(chats), batch_size), desc="Critiques"):
    #     batch_prompts = [build_chat(c, tok) for c in chats[i : i + batch_size]]
    #     critiques.extend(
    #         batch_generate(
    #             pipe,
    #             batch_prompts,
    #             max_new_tokens=max_new_tokens,
    #             temperature=temperature,
    #         )
    #     )
    # for chat, crit in zip(chats, critiques):
    #     chat.append({"role": "assistant", "content": crit})

    # needs_revision = ["the answer is wrong" or "the answer is incorrect" in crit.lower() for crit in critiques]  

    ## no revision for now
    needs_revision = [False] * len(chats)

    # ---------- pass 3: optional revision ----------
    revisions: List[Optional[str]] = [None] * len(chats)
    rev_indices = [i for i, flag in enumerate(needs_revision) if flag]
    if rev_indices:
        rev_prompt = (
            "You indicated that your previous answer was wrong. Based on your evaluation, please provide the correct step-by-step solution to the math problem. "
            "Make sure the answer is in a box: \\boxed{Your Answer}. Stop immediately after the box."
        )
        for idx in rev_indices:
            chats[idx].append({"role": "user", "content": rev_prompt})

        rev_batches = [chats[i] for i in rev_indices]
        rev_texts: List[str] = []
        for i in tqdm(range(0, len(rev_batches), batch_size), desc="Revisions"):
            batch_prompts = [build_chat(c, tok) for c in rev_batches[i : i + batch_size]]
            rev_texts.extend(
                batch_generate(
                    pipe,
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )
        for idx, txt in zip(rev_indices, rev_texts):
            revisions[idx] = txt

    finals = [revisions[i] if revisions[i] is not None else drafts[i] for i in range(len(drafts))]

    # ---------- compute accuracy and write CSV + TensorBoard logs ----------
    rows = []
    dataset_name = hf_dataset.split("/")[-1]  # Extract dataset name for answer extraction
    
    initial_correct = 0
    final_correct = 0
    improved = 0
    
    for i, (row_data, draft, final, crit) in enumerate(zip(ds, drafts, finals, critiques)):
        answer_text  = row_data.get("answer") or row_data.get("solution") or ""
        problem_text = (
            row_data.get("problem") or row_data.get("question") or row_data.get("prompt") or ""
        )
        
        # Extract answers from responses
        initial_pred = extract_answer(draft, dataset_name)
        final_pred = extract_answer(final, dataset_name)
        
        # Compute accuracy
        initial_accuracy = compute_accuracy(initial_pred, str(answer_text), dataset_name)
        final_accuracy = compute_accuracy(final_pred, str(answer_text), dataset_name)
        improved_accuracy = int(final_accuracy and not initial_accuracy)
        
        # Track overall stats
        initial_correct += initial_accuracy
        final_correct += final_accuracy
        improved += improved_accuracy
        
        row = {
            "id":               row_data.get("id", i),
            "problem":          problem_text,
            "answer":           answer_text,
            "initial_response": draft,
            "final_response":   final,
            "was_revised":      int(revisions[i] is not None),
            "critique":         crit,
            "initial_accuracy": int(initial_accuracy),
            "final_accuracy":   int(final_accuracy),
            "improved_accuracy": improved_accuracy,
        }
        rows.append(row)
        
        if writer:
            # Log accuracy metrics
            writer.add_scalar("accuracy/initial", initial_accuracy, i)
            writer.add_scalar("accuracy/final", final_accuracy, i)
            writer.add_scalar("accuracy/improved", improved_accuracy, i)
            
            # log single-row Markdown table for this sample
            df_row = pd.DataFrame([row])
            writer.add_text("results", df_row.to_markdown(index=False), global_step=i)

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    
    # Print summary statistics
    total_samples = len(rows)
    initial_acc = initial_correct / total_samples
    final_acc = final_correct / total_samples
    improvement_rate = improved / total_samples
    revision_rate = sum(row["was_revised"] for row in rows) / total_samples

    # # correct recall: proportion of correct answers that are correctly idenfied by the critique
    # initial_correct_idxs = [i for i, row in enumerate(rows) if row["initial_accuracy"]]
    # deemed_correct_idxs = [i for i, row in enumerate(rows) if not row["was_revised"]]
    # correct_recall = len(deemed_correct_idxs) / len(initial_correct_idxs) if initial_correct_idxs else 0.0
    # print(f"Correct Recall: {correct_recall:.3f}")

    # # wrong recall: proportion of incorrect answers that are correctly identified by critique as wrong
    # initial_incorrect_idxs = [i for i, row in enumerate(rows) if not row["initial_accuracy"]]
    # deemed_incorrect_idxs = [i for i, row in enumerate(rows) if row["was_revised"]]
    # wrong_recall = len(deemed_incorrect_idxs) / len(initial_incorrect_idxs) if initial_incorrect_idxs else 0.0
    # print(f"Wrong Recall: {wrong_recall:.3f}")

    # verifier_accuracy = [i for i, row in enumerate(rows) if (row["was_revised"] and not row["initial_accuracy"]) or (not row["was_revised"] and row["initial_accuracy"])]
    # verifier_accuracy = len(verifier_accuracy) / total_samples
    # print(f"Verifier Accuracy: {verifier_accuracy:.3f}")


    print(f"Saved {total_samples} rows to {out_csv} (dataset: {hf_dataset}/{split}).")
    print(f"Initial Accuracy: {initial_acc:.3f} ({initial_correct}/{total_samples})")
    print(f"Final Accuracy: {final_acc:.3f} ({final_correct}/{total_samples})")
    print(f"Improvement Rate: {improvement_rate:.3f} ({improved}/{total_samples})")
    print(f"Revision Rate: {revision_rate:.3f}")
    
    if writer:
        # Log summary statistics
        writer.add_scalar("summary/initial_accuracy", initial_acc)
        writer.add_scalar("summary/final_accuracy", final_acc)
        writer.add_scalar("summary/improvement_rate", improvement_rate)
        writer.add_scalar("summary/revision_rate", revision_rate)
        
        print(f"TensorBoard logs written to {tb_logdir}")
        writer.close()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Generate self-critique outputs with accuracy metrics")
    parser.add_argument("--model",      default="~/users/mghmr/Intuitor/open-r1-intuitor/data/Qwen2.5-3B-PRM-Test_w_sc")
    parser.add_argument("--hf_dataset", default="openai/gsm8k")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_csv",  default="outputs_gsm8k_w_sc.csv")
    parser.add_argument("--tb_logdir",   default=None, help="TensorBoard logdir (optional)")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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
    )

if __name__ == "__main__":
    main()