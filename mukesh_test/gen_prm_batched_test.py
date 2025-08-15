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

from GenPRM.genprm_batch_inference import GenPRM
from GenPRM.util import timestamped_print
from tqdm import tqdm
import ipdb



# load data
# generation_data = pd.read_csv('outputs_math_500-Q2.5-3B-Ins.csv')

generation_data = pd.read_json('sc_base_math500.json')


problems = generation_data['example']

problems = [re.search(r"Think step by step[^\n]*\n\n([\s\S]*)", p, re.S).group(1).strip() for p in problems]


chats = []
for problem in problems:
    chats.append([
        {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should neither solve the problem nor give the final answer."},
        {"role": "user", "content": problem}
    ])

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
# all_reward_list = []
# all_process_rewards = []
# prm_outputs = []

# ipdb.set_trace()

prm_outputs, prm_rewards = genprm.batch_inference(chats, verify=False, execute=False, logging=False)

# ipdb.set_trace()


generation_data['process_reward'] = prm_rewards
generation_data['prm_output'] = prm_outputs

generation_data.to_json('genprm_base_rewards_math500.json')

