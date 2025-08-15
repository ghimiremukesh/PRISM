# # evaluate_dataset.py — batched self‑critique for any HF dataset (no grading)
# """
# Generate *draft*, *critique*, and (optional) *revision* responses for a
# subset of any HF dataset and save a CSV. Defaults target the **openai/
#  gsm8k** `test` split with 30 random problems, but the script works for
# any dataset that has at least a question/answer field.

# CSV columns
# -----------
# * `id`                – row index or provided id
# * `problem`/`question` – source text
# * `answer`            – ground‑truth answer from dataset
# * `initial_response`  – model’s first reply
# * `final_response`    – revised if `was_revised==1`, else same as draft
# * `was_revised`       – 1 if a revision was actually generated
# * `critique`          – self‑critique text

# No accuracy calculation is performed.

# Example (gsm8k)
# ~~~~~~~~~~~~~~~
# ```bash
# python evaluate_dataset.py \
#   --model Qwen/Qwen2.5-3B \
#   --hf_dataset openai/gsm8k \
#   --split test \
#   --num_samples 30 \
#   --output_csv gsm8k_qwen.csv
# ```
# """
# from __future__ import annotations

# import argparse
# import re
# from typing import List, Optional

# import pandas as pd
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from tqdm import tqdm

# # ---------------------------------------------------------------------------
# # Utilities
# # ---------------------------------------------------------------------------

# def build_chat(messages: List[dict[str, str]], tok) -> str:
#     """Format chat messages into a single prompt string."""
#     if hasattr(tok, "apply_chat_template"):
#         return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"


# def load_pipe(model_name: str, device: str):
#     tok = AutoTokenizer.from_pretrained(model_name)
#     # decoder‑only models need left padding
#     tok.padding_side = "left"
#     if tok.pad_token_id is None:
#         tok.pad_token_id = tok.eos_token_id
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")
#     return pipeline("text-generation", model=model, tokenizer=tok, device_map=device), tok


# def batch_generate(pipe, prompts: List[str], *, max_new_tokens: int, temperature: float) -> List[str]:
#     """Generate continuations for *prompts* using the HF pipeline."""
#     outs = pipe(
#         prompts,
#         max_new_tokens=max_new_tokens,
#         do_sample=temperature > 0.0,
#         temperature=temperature,
#         batch_size=len(prompts),
#         return_full_text=False,
#     )  # List[dict] or List[List[dict]]

#     gens: List[str] = []
#     for res in outs:
#         if isinstance(res, list):  # multiple sequences case
#             res = res[0]
#         gens.append(res["generated_text"].strip())
#     return gens

# # ---------------------------------------------------------------------------
# # Main evaluation routine
# # ---------------------------------------------------------------------------

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
# ):
#     # Load and optionally subsample
#     ds = load_dataset(hf_dataset, split=split)
#     if 0 < num_samples < len(ds):
#         ds = ds.shuffle(seed=seed).select(range(num_samples))

#     pipe, tok = load_pipe(model_name, device)

#     system_prompt = (
#         "You are a helpful AI assistant. FIRST think step‑by‑step, show all your work, "
#         "then provide the answer in the form \\boxed{…}."
#     )

#     # -------- pass 1: draft generation --------
#     chats: List[List[dict[str, str]]] = []
#     for row in ds:
#         problem_text = (
#             row.get("problem") or row.get("question") or row.get("prompt") or str(row)
#         )
#         chats.append([
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": problem_text.strip()},
#         ])

#     drafts: List[str] = []
#     for i in tqdm(range(0, len(chats), batch_size), desc="Drafts"):
#         batch_prompts = [build_chat(c, tok) for c in chats[i : i + batch_size]]
#         drafts.extend(
#             batch_generate(
#                 pipe,
#                 batch_prompts,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#             )
#         )
#     for chat, draft in zip(chats, drafts):
#         chat.append({"role": "assistant", "content": draft})

#     # -------- pass 2: critique --------
#     crit_prompt = (
#         "Check the math solution step-by-step. If you find a mistake: state the wrong step, explain why it’s wrong, and end your response with ’The answer is wrong’. If all steps are correct, end your response with ’The answer is correct’."
#     )
#     for chat in chats:
#         chat.append({"role": "user", "content": crit_prompt})

#     critiques: List[str] = []
#     for i in tqdm(range(0, len(chats), batch_size), desc="Critiques"):
#         batch_prompts = [build_chat(c, tok) for c in chats[i : i + batch_size]]
#         critiques.extend(
#             batch_generate(
#                 pipe,
#                 batch_prompts,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#             )
#         )
#     for chat, crit in zip(chats, critiques):
#         chat.append({"role": "assistant", "content": crit})

#     # Detect which need revision
#     needs_revision = [
#         "wrong" in crit.lower()
#         for crit in critiques
#     ]

#     # -------- pass 3: optional revision --------
#     revisions: List[Optional[str]] = [None] * len(chats)
#     rev_indices = [i for i, flag in enumerate(needs_revision) if flag]
#     if rev_indices:
#         rev_prompt = (
# "You indicated that your previous answer was wrong. Please provide the correct solution to the math problem. Make sure the answer is in a box: \\boxed{Your Answer}. Please stop generation immediately after outputting the box."
#         )
#         for idx in rev_indices:
#             chats[idx].append({"role": "user", "content": rev_prompt})

#         rev_batches = [chats[i] for i in rev_indices]
#         rev_texts: List[str] = []
#         for i in tqdm(range(0, len(rev_batches), batch_size), desc="Revisions"):
#             batch_prompts = [build_chat(c, tok) for c in rev_batches[i : i + batch_size]]
#             rev_texts.extend(
#                 batch_generate(
#                     pipe,
#                     batch_prompts,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                 )
#             )
#         for idx, txt in zip(rev_indices, rev_texts):
#             revisions[idx] = txt

#     # Final response: either draft or revision
#     finals = [revisions[i] if revisions[i] is not None else drafts[i] for i in range(len(drafts))]

#     # -------- write CSV --------
#     rows = []
#     for i, (row_data, draft, final, crit) in enumerate(zip(ds, drafts, finals, critiques)):
#         answer_text  = row_data.get("answer") or row_data.get("solution") or ""
#         problem_text = (
#             row_data.get("problem") or row_data.get("question") or row_data.get("prompt") or ""
#         )
#         rows.append({
#             "id":               row_data.get("id", i),
#             "problem":          problem_text,
#             "answer":           answer_text,
#             "initial_response": draft,
#             "final_response":   final,
#             "was_revised":      int(revisions[i] is not None),
#             "critique":         crit,
#         })

#     pd.DataFrame(rows).to_csv(out_csv, index=False)
#     print(f"Saved {len(rows)} rows to {out_csv} (dataset: {hf_dataset}/{split}).")

# # ---------------------------------------------------------------------------
# # CLI entry point
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser("Generate self‑critique outputs (no grading)")
#     parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
#     parser.add_argument("--hf_dataset", default="knoveleng/Minerva-Math")
#     parser.add_argument("--split", default="test")
#     parser.add_argument("--num_samples", type=int, default=16)
#     parser.add_argument("--seed",        type=int, default=420)
#     parser.add_argument("--output_csv", default="outputs.csv")
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--max_new_tokens", type=int, default=4096)
#     parser.add_argument("--temperature",    type=float, default=0.0)
#     parser.add_argument("--device", default="auto")
#     args = parser.parse_args()

#     evaluate(
#         model_name=args.model,
#         hf_dataset=args.hf_dataset,
#         split=args.split,
#         out_csv=args.output_csv,
#         num_samples=args.num_samples,
#         seed=args.seed,
#         batch_size=args.batch_size,
#         max_new_tokens=args.max_new_tokens,
#         temperature=args.temperature,
#         device=args.device,
#     )


# if __name__ == "__main__":
#     main()


# evaluate_dataset.py — batched self-critique for any HF dataset (no grading)
"""
Generate *draft*, *critique*, and (optional) *revision* responses for a
subset of any HF dataset and save a CSV. Defaults target the **openai/
 gsm8k** `test` split with 30 random problems, but the script works for
any dataset that has at least a question/answer field.

CSV columns
-----------
* `id`                – row index or provided id
* `problem`/`question` – source text
* `answer`            – ground-truth answer from dataset
* `initial_response`  – model’s first reply
* `final_response`    – revised if `was_revised==1`, else same as draft
* `was_revised`       – 1 if a revision was actually generated
* `critique`          – self-critique text

No accuracy calculation is performed.

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
# Utilities
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
    """Run self-critique and write CSV + optional TensorBoard logs."""
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
    for chat, draft in zip(chats, drafts):
        chat.append({"role": "assistant", "content": draft})

    # ---------- pass 2: critique ----------
    crit_prompt = (
        "Check the math solution step-by-step. If you find a mistake: state the wrong step, explain why it’s wrong, and end your response with 'The answer is wrong'. "
        "If all steps are correct, end with 'The answer is correct'."
    )
    for chat in chats:
        chat.append({"role": "user", "content": crit_prompt})

    critiques: List[str] = []
    for i in tqdm(range(0, len(chats), batch_size), desc="Critiques"):
        batch_prompts = [build_chat(c, tok) for c in chats[i : i + batch_size]]
        critiques.extend(
            batch_generate(
                pipe,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
    for chat, crit in zip(chats, critiques):
        chat.append({"role": "assistant", "content": crit})

    needs_revision = ["wrong" in crit.lower() for crit in critiques]

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

    # ---------- write CSV + TensorBoard logs ----------
    rows = []
    for i, (row_data, draft, final, crit) in enumerate(zip(ds, drafts, finals, critiques)):
        answer_text  = row_data.get("answer") or row_data.get("solution") or ""
        problem_text = (
            row_data.get("problem") or row_data.get("question") or row_data.get("prompt") or ""
        )
        row = {
            "id":               row_data.get("id", i),
            "problem":          problem_text,
            "answer":           answer_text,
            "initial_response": draft,
            "final_response":   final,
            "was_revised":      int(revisions[i] is not None),
            "critique":         crit,
        }
        rows.append(row)
        if writer:
            # log single-row Markdown table for this sample
            df_row = pd.DataFrame([row])
            writer.add_text("results", df_row.to_markdown(index=False), global_step=i)

    # save CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv} (dataset: {hf_dataset}/{split}).")
    if writer:
        print(f"TensorBoard logs written to {tb_logdir}")
        writer.close()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Generate self-critique outputs (no grading)")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf_dataset", default="yentinglin/aime_2025")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_csv",  default="outputs.csv")
    parser.add_argument("--tb_logdir",   default=None, help="TensorBoard logdir (optional)")
    parser.add_argument("--batch_size",  type=int, default=4)
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
