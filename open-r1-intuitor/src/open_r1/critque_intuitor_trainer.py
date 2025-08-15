# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from trl-0.18.0 grpo_trainer.py

import copy
import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

import numpy as np
import ipdb 



if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class INTUITORTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `processing_class.pad_token`, must be set. If the processing class has not set a padding
            token, `processing_class.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = [],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            if is_peft_model(model):
                raise TypeError("Liger loss is not supported with a PEFT model.")

            if self.loss_type != "bnpo":
                raise ValueError(
                    f"The provided loss type (`{self.loss_type}`) is not supported with `use_liger_loss`. Liger loss "
                    "only supports `bnpo` for now."
                )

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.ref_model is not None,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        # self._textual_logs = {
        #     "prompt": deque(maxlen=maxlen),
        #     "completion": deque(maxlen=maxlen),
        #     "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        # }
        self._textual_logs = {
        "prompt": deque(maxlen=maxlen),
        "initial_response": deque(maxlen=maxlen),  # renamed from "completion"
        "critique": deque(maxlen=maxlen),         # new
        "final_response": deque(maxlen=maxlen),   # new
        "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        num_processes = self.accelerator.num_processes
        effective_batch_size = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [
            n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x "
                f"{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current effective eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )
                self.vllm_client.init_communicator()

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch, our dataloader loads an *accumulated* batch
    # (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate completions
    # once per optimization step—rather than once per gradient accumulation step—which is significantly more efficient.
    # The only change from the original implementation is multiplying the batch size by `gradient_accumulation_steps`.
    # Thus, `_prepare_inputs` is called with the accumulated batch size, and it handles the splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |     Accum step 0      |     Accum step 1      |
        #                                      |   GPU 0   |   GPU 1   |   GPU 0   |   GPU 1   |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Take the stored generations and use the first slice to compute the loss
        #  num_iterations=2 ▼  1          3      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #
        #                      2          4     [6   6   7   7   8   8]  9   9  10  10  11  11    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #                      2          5      6   6   7   7   8   8 [ 9   9  10  10  11  11]   <- ...
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)
    
    @profiling_decorator
    def _get_per_token_self_certainty(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        """
        Computes per-token self-certainty in batches. We add constnat logV, which is equivalent to the original
        formula as we are goting to normalize it in batch.
        """
        batch_size = batch_size or input_ids.size(0)
        all_sce = []
        # Process inputs in chunks
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # Get logits: shape (B, seq_len, V)
            logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            # Keep only the last `logits_to_keep` positions
            sce_chunk = logits[:, -logits_to_keep:, :]  # (B, L_keep, V)
            # Compute log-sum-exp across vocabulary and subtract mean logit
            # logsumexp - mean gives a measure of dispersion (self-certainty)
            sce_values = torch.logsumexp(sce_chunk, dim=-1) - sce_chunk.mean(dim=-1)
            all_sce.append(sce_values)

        # Concatenate over batch dimension
        return torch.cat(all_sce, dim=0)

    
    @profiling_decorator
    @torch.no_grad()
    def _get_advantage_from_sce(
        self,
        SCe: torch.Tensor,
        completion_mask: torch.Tensor,          
    ) -> torch.Tensor:
        """
        Calculates advantage scores from SCe values and returns a 1D tensor with a single advantage per sample.

        Args
        ----
        SCe : (B, L) tensor
            Per-token score.
        completion_mask : (B, L) bool / 0-1 tensor
            1 for valid tokens, 0 for padding.
        Returns
        -------
        advantage : (B,) tensor 
        """
        output_lengths = completion_mask.sum(dim=1, dtype=torch.long) # (B,)
        advantage = torch.zeros(SCe.size(0), device=SCe.device, dtype=SCe.dtype)
        valid_mask = output_lengths > 0
        if valid_mask.any():
            sce_sum = (SCe * completion_mask).sum(dim=1)
            advantage[valid_mask] = sce_sum[valid_mask] / output_lengths[valid_mask].to(SCe.dtype)
        return advantage.detach()

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the accumulated local batch (Per-GPU batch size × Gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire accumulated batch and splits it into smaller batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, there is neither gradient accumulation, nor multiple iterations
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # Constants
        CRITIC_INSTRUCTIONS = "Check the math solution step-by-step. If you find a mistake: state the wrong step, explain why it's wrong, and end your response with 'The answer is wrong'. If all steps are correct, end your response with 'The answer is correct'. YOU MUST END WITH EITHER 'The answer is correct' OR 'The answer is wrong'."
        REVISION_INSTRUCTIONS = "You indicated that your previous answer was wrong. Please provide the correct step-by-step solution to the math problem, and make sure to BOX YOUR ANSWER in \\boxed{Your answer}."
        CRITIQUE_ADVANTAGE_WEIGHT = 0.5  # γ in your notation

        # Extract prompts and prepare inputs
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # Prepare prompt inputs
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Store original prompt info for final output
        original_prompt_ids = prompt_ids.clone()
        original_prompt_mask = prompt_mask.clone()

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate initial completions
        initial_completion_ids, initial_completion_mask, initial_attention_mask = self._generate_completions(
            prompt_ids, prompt_mask, prompts_text, mode="initial"
        )
        
        # Compute initial self-certainty and logps
        initial_logits_to_keep = initial_completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        
        with torch.no_grad():
            initial_sce_raw = self._compute_self_certainty_and_logps(
                prompt_ids, initial_completion_ids, initial_attention_mask, 
                initial_completion_mask, initial_logits_to_keep, batch_size
            )
            
        # Decode initial completions
        initial_completions_text = self.processing_class.batch_decode(
            initial_completion_ids, skip_special_tokens=True
        )
        
        # Compute initial rewards and advantages
        initial_rewards_per_func = self._compute_rewards(prompts, initial_completions_text, inputs)
        initial_advantages, initial_sce_stats = self._compute_advantages(
            initial_rewards_per_func, initial_sce_raw, initial_completion_mask, len(prompts)
        )
        
        # Generate critiques
        critique_completion_ids, critique_text = self._generate_critiques(
            inputs, initial_completions_text, CRITIC_INSTRUCTIONS
        )
        
        # Identify wrong answers
        wrong_idxs = [i for i in range(len(critique_text)) if "the answer is wrong" in critique_text[i].lower() or "the answer is incorrect" in critique_text[i].lower()]
        
        # Store initial completion mask for logging
        initial_completion_mask = initial_completion_mask.clone()
        
        # Initialize final values with initial values
        final_completion_ids = initial_completion_ids.clone()
        final_completion_mask = initial_completion_mask.clone()
        final_sce_raw = initial_sce_raw.clone()
        final_completions_text = initial_completions_text.copy()
        
        # If there are wrong answers, generate revisions
        if wrong_idxs:
            # Generate revisions for wrong answers
            revised_completion_ids, revised_completion_mask, revised_sce_raw, revised_completions_text = \
                self._generate_revisions(
                    inputs, wrong_idxs, critique_text, REVISION_INSTRUCTIONS, batch_size
                )
            
            # Update final values with revisions
            self._update_with_revisions(
                wrong_idxs, final_completion_ids, final_completion_mask, 
                final_sce_raw, final_completions_text,
                revised_completion_ids, revised_completion_mask, 
                revised_sce_raw, revised_completions_text
            )
            
            # Compute revised rewards for logging
            revised_rewards_per_func = self._compute_rewards(
                [prompts[i] for i in wrong_idxs], revised_completions_text, 
                [inputs[i] for i in wrong_idxs]
            )
        else:
            revised_rewards_per_func = torch.zeros(0, max(len(self.reward_funcs), 1), device=device)
        
        # Compute final rewards and advantages
        final_rewards_per_func = initial_rewards_per_func.clone()
        if wrong_idxs:
            final_rewards_per_func[wrong_idxs] = revised_rewards_per_func
        
        # Compute critique advantage
        critique_advantage_raw = self._compute_critique_advantage(
            initial_sce_raw, final_sce_raw, wrong_idxs, CRITIQUE_ADVANTAGE_WEIGHT
        )
        
        # Compute final advantages with critique bonus
        final_advantages, final_sce_stats, critique_stats = self._compute_final_advantages(
            final_rewards_per_func, final_sce_raw, critique_advantage_raw, 
            final_completion_mask, len(prompts)
        )
        
        # Log metrics
        self._log_metrics(
            mode, initial_rewards_per_func, revised_rewards_per_func,
            initial_sce_stats, final_sce_stats, final_completion_mask,
            final_completion_ids, wrong_idxs, critique_text, 
            initial_sce_raw, final_sce_raw, initial_completion_mask
        )
        
        # Log textual outputs
        self._log_textual_outputs(
            prompts_text, initial_completions_text, critique_text, 
            final_completions_text, initial_rewards_per_func, final_rewards_per_func,
        )
        
        # Compute log probabilities for final completions
        final_old_per_token_logps, final_ref_per_token_logps = self._compute_final_logps(
            original_prompt_ids, final_completion_ids, final_completion_mask, batch_size
        )

        # ipdb.set_trace()
        
        # Return values for loss computation
        return {
            "prompt_ids": original_prompt_ids,
            "prompt_mask": original_prompt_mask,
            "completion_ids": final_completion_ids,
            "completion_mask": final_completion_mask,
            "advantages": final_advantages,
            "old_per_token_logps": final_old_per_token_logps,
            "ref_per_token_logps": final_ref_per_token_logps,
        }

    def _generate_completions(self, prompt_ids, prompt_mask, prompts_text, mode="initial"):
        """Generate completions and return ids, masks, and attention masks."""
        device = self.accelerator.device
        
        if self.use_vllm:
            completion_ids = self._vllm_generate(prompts_text, mode)
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, 
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, 
                    generation_config=self.generation_config
                )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Create completion mask
        completion_mask = self._create_completion_mask(completion_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        return completion_ids, completion_mask, attention_mask

    def _vllm_generate(self, prompts_text, mode):
        """Handle vLLM generation with proper gathering/broadcasting."""
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        
        all_prompts_text = gather_object(prompts_text)
        # all_prompts_text = prompts_text
        
        if self.accelerator.is_main_process:
            if mode == "initial":
                # For initial generation, deduplicate prompts
                ordered_prompts = all_prompts_text[::self.num_generations]
                n = self.num_generations
            else:
                # For critique/revision, generate one per prompt
                ordered_prompts = all_prompts_text
                n = 1
                
            with profiling_context(self, f"vLLM.{mode}_generate"):
                completion_ids = self.vllm_client.generate(
                    prompts=ordered_prompts,
                    n=n,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding_regex=self.guided_decoding_regex,
                )
        else:
            completion_ids = [None] * len(all_prompts_text)
        
        # Broadcast and slice
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        return completion_ids[process_slice]


    def _create_completion_mask(self, completion_ids):
        """Create mask for completions, handling EOS tokens."""
        device = completion_ids.device
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()
        
        return completion_mask

    def _compute_self_certainty_and_logps(self, prompt_ids, completion_ids, attention_mask, 
                                        completion_mask, logits_to_keep, batch_size):
        """Compute self-certainty scores and log probabilities."""
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        sce_raw = self._get_per_token_self_certainty(
            self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
        )
        sce_raw = self._get_advantage_from_sce(sce_raw, completion_mask)
        
        return sce_raw

    def _compute_rewards(self, prompts, completions_text, inputs):
        """Compute rewards for completions."""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), max(len(self.reward_funcs), 1), device=device)
        
        # Convert completions to proper format
        if is_conversational(inputs[0]):
            completions = []
            for i, (prompt, completion) in enumerate(zip(prompts, completions_text)):
                # Check if there's a bootstrap from the original prompt
                if prompt[-1]["role"] == "assistant":
                    bootstrap = prompt[-1]["content"]
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
                else:
                    completions.append([{"role": "assistant", "content": completion}])
        else:
            completions = completions_text
        
        # Compute rewards for each function
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # Neural reward model
                    if is_conversational(inputs[0]):
                        messages = []
                        for j, (prompt, comp) in enumerate(zip(prompts, completions)):
                            # Create a copy of the prompt to avoid modifying the original
                            prompt_copy = [msg.copy() for msg in prompt]
                            messages.append({"messages": prompt_copy + comp})
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, 
                        padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # Function-based reward
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    output = [r if r is not None else torch.nan for r in output]
                    rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        
        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            if 'reward_kwargs' in locals():
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
                row_reward_kwargs["prompt"] = prompts[nan_row_idx]
                row_reward_kwargs["completion"] = completions[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )
        
        return rewards_per_func 

    def _compute_advantages(self, rewards_per_func, sce_raw, completion_mask, num_prompts):
        """Compute advantages from rewards and self-certainty."""
        device = self.accelerator.device
        
        # Gather rewards and SCE
        rewards_per_func = gather(rewards_per_func)
        sce_mean = gather(sce_raw)
        
        # Apply reward weights
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Compute grouped statistics
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Compute reward-based advantages
        if torch.any(self.reward_weights != 0) and len(self.reward_funcs) > 0:
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
        else:
            advantages = torch.zeros_like(rewards)
        
        # Slice to local process
        process_slice = slice(
            self.accelerator.process_index * num_prompts,
            (self.accelerator.process_index + 1) * num_prompts,
        )
        advantages = advantages[process_slice]
        
        # Compute SCE statistics
        sce_grouped = sce_mean.view(-1, self.num_generations)
        mean_sce = sce_grouped.mean(dim=1)
        std_sce = sce_grouped.std(dim=1)
        
        sce_stats = {
            "mean": mean_sce.mean().item(),
            "std": std_sce.mean().item(),
            "mean_expanded": mean_sce.repeat_interleave(self.num_generations, dim=0)[process_slice],
            "std_expanded": std_sce.repeat_interleave(self.num_generations, dim=0)[process_slice]
        }
        
        # Normalize SCE advantages
        sce_advantage = (sce_raw - sce_stats["mean_expanded"]) / (sce_stats["std_expanded"] + 1e-4)
        
        # Combine advantages
        total_advantage = sce_advantage + advantages
        
        return total_advantage, sce_stats

    def _generate_critiques(self, inputs, completions_text, critic_instructions):
        """Generate critiques for the completions."""
        # Prepare inputs with critiques
        critique_inputs = copy.deepcopy(inputs)
        for i in range(len(critique_inputs)):
            critique_inputs[i]["prompt"].append({"content": completions_text[i], "role": "assistant"})
            critique_inputs[i]["prompt"].append({"content": critic_instructions, "role": "user"})
        
        # Prepare text inputs
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] 
                    for example in critique_inputs]
        
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        # Generate critiques
        critique_completion_ids, _, _ = self._generate_completions(
            prompt_ids, prompt_mask, prompts_text, mode="critique"
        )
        
        # Decode critiques
        critique_text = self.processing_class.batch_decode(
            critique_completion_ids, skip_special_tokens=True
        )
        
        return critique_completion_ids, critique_text

    # def _generate_revisions(self, inputs, wrong_idxs, critique_text, revision_instructions, batch_size):
    #     """Generate revisions for wrong answers."""
    #     # Prepare revision inputs
    #     wrong_inputs = [copy.deepcopy(inputs[idx]) for idx in wrong_idxs]
    #     for i, idx in enumerate(wrong_idxs):
    #         wrong_inputs[i]["prompt"].append({"content": critique_text[idx], "role": "assistant"})
    #         wrong_inputs[i]["prompt"].append({"content": revision_instructions, "role": "user"})
        
    #     # Prepare prompts
    #     prompts = [x["prompt"] for x in wrong_inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] 
    #                 for example in wrong_inputs]
        
    #     prompt_inputs = self.processing_class(
    #         text=prompts_text, return_tensors="pt", padding=True, 
    #         padding_side="left", add_special_tokens=False
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)
    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
    #     # Generate revisions
    #     revised_completion_ids, revised_completion_mask, revised_attention_mask = \
    #         self._generate_completions(prompt_ids, prompt_mask, prompts_text, mode="revision")
        
    #     # Compute revised SCE
    #     logits_to_keep = revised_completion_ids.size(1)
    #     with torch.no_grad():
    #         revised_sce_raw = self._compute_self_certainty_and_logps(
    #             prompt_ids, revised_completion_ids, revised_attention_mask, 
    #             revised_completion_mask, logits_to_keep, batch_size
    #         )
        
    #     # Decode revisions
    #     revised_completions_text = self.processing_class.batch_decode(
    #         revised_completion_ids, skip_special_tokens=True
    #     )
        
    #     return revised_completion_ids, revised_completion_mask, revised_sce_raw, revised_completions_text

    def _generate_revisions(self, inputs, wrong_idxs, critique_text, revision_instructions, batch_size):
        """Generate revisions for wrong answers."""
        if not wrong_idxs:
            return None, None, None, None
        
        device = self.accelerator.device
        
        # Prepare revision inputs
        wrong_inputs = [copy.deepcopy(inputs[idx]) for idx in wrong_idxs]
        for i, idx in enumerate(wrong_idxs):
            wrong_inputs[i]["prompt"].append({"content": critique_text[idx], "role": "assistant"})
            wrong_inputs[i]["prompt"].append({"content": revision_instructions, "role": "user"})
        
        # ===== MULTI-GPU FIX STARTS HERE =====
        # Check if any process has revisions to make
        local_has_revisions = len(wrong_inputs) > 0
        global_has_revisions = self.accelerator.reduce(
            torch.tensor([local_has_revisions], device=device), reduction="max"
        ).item()
        
        if not global_has_revisions:
            return None, None, None, None
        
        # Find max revisions across all processes
        max_revisions = self.accelerator.reduce(
            torch.tensor([len(wrong_inputs)], device=device), reduction="max"
        ).item()
        
        # Pad with dummy inputs if needed
        num_padding = max_revisions - len(wrong_inputs)
        if num_padding > 0:
            dummy_input = copy.deepcopy(wrong_inputs[0] if wrong_inputs else inputs[0])
            wrong_inputs.extend([dummy_input] * num_padding)
        # ===== MULTI-GPU FIX ENDS HERE =====
        
        # Prepare prompts
        prompts = [x["prompt"] for x in wrong_inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] 
                    for example in wrong_inputs]
        
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        # Generate revisions
        revised_completion_ids, revised_completion_mask, revised_attention_mask = \
            self._generate_completions(prompt_ids, prompt_mask, prompts_text, mode="revision")
        
        # ===== MULTI-GPU FIX: REMOVE PADDING =====
        if num_padding > 0:
            revised_completion_ids = revised_completion_ids[:-num_padding]
            revised_completion_mask = revised_completion_mask[:-num_padding]
            revised_attention_mask = revised_attention_mask[:-num_padding]
            prompt_ids = prompt_ids[:-num_padding]
            prompt_mask = prompt_mask[:-num_padding]
        # ===== END PADDING REMOVAL =====
        
        # Compute revised SCE
        logits_to_keep = revised_completion_ids.size(1)
        with torch.no_grad():
            revised_sce_raw = self._compute_self_certainty_and_logps(
                prompt_ids, revised_completion_ids, revised_attention_mask, 
                revised_completion_mask, logits_to_keep, batch_size
            )
        
        
        # Decode revisions
        revised_completions_text = self.processing_class.batch_decode(
            revised_completion_ids, skip_special_tokens=True
        )
        
        return revised_completion_ids, revised_completion_mask, revised_sce_raw, revised_completions_text

    def _update_with_revisions(self, wrong_idxs, final_completion_ids, final_completion_mask,
                            final_sce_raw, final_completions_text, revised_completion_ids,
                            revised_completion_mask, revised_sce_raw, revised_completions_text):
        """Update final values with revisions, handling padding."""
        device = final_completion_ids.device
        
        for i, idx in enumerate(wrong_idxs):
            # Get dimensions
            final_len = final_completion_ids[idx].size(0)
            revised_len = revised_completion_ids[i].size(0)
            
            if final_len > revised_len:
                # Pad revised to match final
                padding = torch.full((final_len - revised_len,), 
                                self.processing_class.pad_token_id, 
                                device=device)
                revised_ids_padded = torch.cat([revised_completion_ids[i], padding])
                revised_mask_padded = torch.cat([revised_completion_mask[i], 
                                            torch.zeros(final_len - revised_len, device=device)])
                
                final_completion_ids[idx] = revised_ids_padded
                final_completion_mask[idx] = revised_mask_padded
            elif revised_len > final_len:
                # Pad all final tensors to match revised
                max_len = revised_len
                
                # Pad final_completion_ids
                padding = torch.full((final_completion_ids.size(0), max_len - final_len), 
                                self.processing_class.pad_token_id, 
                                device=device)
                final_completion_ids = torch.cat([final_completion_ids, padding], dim=1)
                
                # Pad final_completion_mask
                mask_padding = torch.zeros(final_completion_mask.size(0), max_len - final_len, 
                                        device=device)
                final_completion_mask = torch.cat([final_completion_mask, mask_padding], dim=1)
                
                # Now update the specific index
                final_completion_ids[idx] = revised_completion_ids[i]
                final_completion_mask[idx] = revised_completion_mask[i]
            else:
                # Same length, direct assignment
                final_completion_ids[idx] = revised_completion_ids[i]
                final_completion_mask[idx] = revised_completion_mask[i]
            
            # Update other values
            final_sce_raw[idx] = revised_sce_raw[i]
            final_completions_text[idx] = revised_completions_text[i]

    def _compute_critique_advantage(self, initial_sce_raw, final_sce_raw, wrong_idxs, weight):
        """Compute critique advantage bonus."""
        critique_advantage_raw = weight * final_sce_raw.clone()
        
        # For revised responses, advantage is γ * (new_sc - old_sc)
        for idx in wrong_idxs:
            critique_advantage_raw[idx] = weight * (final_sce_raw[idx] - initial_sce_raw[idx])
        
        return critique_advantage_raw

    def _compute_final_advantages(self, final_rewards_per_func, final_sce_raw, 
                                critique_advantage_raw, final_completion_mask, num_prompts):
        """Compute final advantages including critique bonus."""
        # Compute base advantages
        final_advantages, final_sce_stats = self._compute_advantages(
            final_rewards_per_func, final_sce_raw, final_completion_mask, num_prompts
        )
        
        # Compute critique advantage statistics
        critique_advantage_gathered = gather(critique_advantage_raw)
        critique_grouped = critique_advantage_gathered.view(-1, self.num_generations)
        mean_critique = critique_grouped.mean(dim=1)
        std_critique = critique_grouped.std(dim=1)
        
        # Normalize critique advantages
        process_slice = slice(
            self.accelerator.process_index * num_prompts,
            (self.accelerator.process_index + 1) * num_prompts,
        )
        
        mean_critique_expanded = mean_critique.repeat_interleave(self.num_generations, dim=0)[process_slice]
        std_critique_expanded = std_critique.repeat_interleave(self.num_generations, dim=0)[process_slice]
        
        critique_advantage = (critique_advantage_raw - mean_critique_expanded) / (std_critique_expanded + 1e-4)
        
        # Add critique bonus to final advantages
        final_advantages = final_advantages + critique_advantage
        
        critique_stats = {
            "mean": mean_critique.mean().item(),
            "std": std_critique.mean().item()
        }
        
        # Store for logging
        self._critique_stats = critique_stats
        
        return final_advantages, final_sce_stats, critique_stats

    def _log_metrics(self, mode, initial_rewards_per_func, revised_rewards_per_func,
                    initial_sce_stats, final_sce_stats, final_completion_mask,
                    final_completion_ids, wrong_idxs, critique_text, 
                    initial_sce_raw, final_sce_raw, initial_completion_mask):
        """Log all metrics."""
        device = self.accelerator.device

            # --- Gather all torch.Tensor metrics ---
        initial_rewards_per_func   = self.accelerator.gather_for_metrics(initial_rewards_per_func)
        revised_rewards_per_func   = self.accelerator.gather_for_metrics(revised_rewards_per_func)  # This will handle variable sizes
        initial_sce_raw            = self.accelerator.gather_for_metrics(initial_sce_raw)
        final_sce_raw              = self.accelerator.gather_for_metrics(final_sce_raw)

        # --- Gather Python objects ---
        wrong_idxs                 = gather_object(wrong_idxs)
        critique_text              = gather_object(critique_text)
        # initial_sce_stats          = gather_object(initial_sce_stats)
        # final_sce_stats            = gather_object(final_sce_stats)


        # After gather_object calls, extract the first dictionary
        # initial_sce_stats = initial_sce_stats[0]  # Take first dict
        # final_sce_stats = final_sce_stats[0]      # Take first dict


        # Token count
        if mode == "train":
            total_tokens = self.accelerator.gather_for_metrics(final_completion_mask.sum()).sum().item()
            self.state.num_input_tokens_seen += total_tokens
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        # Completion lengths
        agg_completion_mask = self.accelerator.gather_for_metrics(final_completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())
        
        # EOS statistics
        is_eos = final_completion_ids == self.processing_class.eos_token_id
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_ratio)
        
        if len(term_completion_mask) == 0:
            term_completion_mask = torch.zeros(1, device=device)
        
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())
        
        # Reward metrics
        for i, reward_func_name in enumerate(self.reward_func_names):
            # Initial accuracy
            mean_initial = torch.nanmean(initial_rewards_per_func[:, i]).item()
            std_initial = nanstd(initial_rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean_initial"].append(mean_initial)
            self._metrics[mode][f"rewards/{reward_func_name}/std_initial"].append(std_initial)
            
            # Revision accuracy (if any revisions were made)
            if len(revised_rewards_per_func) > 0:
                mean_revision = torch.nanmean(revised_rewards_per_func[:, i]).item()
                std_revision = nanstd(revised_rewards_per_func[:, i]).item()
                self._metrics[mode][f"rewards/{reward_func_name}/mean_revision"].append(mean_revision)
                self._metrics[mode][f"rewards/{reward_func_name}/std_revision"].append(std_revision)
        
        # SCE metrics        
        # print(initial_sce_stats)
        self._metrics[mode]["sce_advantage_initial"].append(initial_sce_stats["mean"])
        self._metrics[mode]["sce_advantage_std_initial"].append(initial_sce_stats["std"])
        self._metrics[mode]["sce_advantage_final"].append(final_sce_stats["mean"])
        self._metrics[mode]["sce_advantage_std_final"].append(final_sce_stats["std"])
        
        # Critique statistics
        self._metrics[mode]["critique/num_wrong"].append(len(wrong_idxs))
        self._metrics[mode]["critique/wrong_ratio"].append(len(wrong_idxs) / len(initial_rewards_per_func))
        
        # 1. Critique Quality Metrics
        if len(self.reward_funcs) > 0:
            # Assuming the first reward function gives correctness (1 for correct, 0 for wrong)
            initial_correctness = (initial_rewards_per_func[:, 0] > 0.5).cpu()
            critique_says_wrong = torch.zeros(len(initial_rewards_per_func), dtype=torch.bool)
            critique_says_wrong[wrong_idxs] = True
            
            # False positive: critique says correct but answer was wrong
            false_positives = (~critique_says_wrong & ~initial_correctness).sum().item()
            true_positives = (~critique_says_wrong & initial_correctness).sum().item()
            false_negatives = (critique_says_wrong & initial_correctness).sum().item()
            true_negatives = (critique_says_wrong & ~initial_correctness).sum().item()
            
            total_correct = initial_correctness.sum().item()
            total_incorrect = (~initial_correctness).sum().item()
            
            false_positive_rate = false_positives / max(total_incorrect, 1)
            false_negative_rate = false_negatives / max(total_correct, 1)

            # sensitivity
            sensitivity = true_positives / (true_positives + false_negatives)
            
            # specifity
            specifity = true_negatives / (true_negatives + false_positives)


            
            self._metrics[mode]["critique/false_positive_rate"].append(false_positive_rate)
            self._metrics[mode]["critique/false_negative_rate"].append(false_negative_rate)
            self._metrics[mode]["critique/precision"].append(true_positives / max(true_positives + false_positives, 1))
            self._metrics[mode]["critique/recall"].append(true_positives / max(true_positives + false_negatives, 1))
            self._metrics[mode]["critique/sensitivity"].append(sensitivity)
            self._metrics[mode]["critique/specifity"].append(specifity)
        
        # Critique length
        critique_lengths = [len(self.processing_class.encode(c)) for c in critique_text]
        self._metrics[mode]["critique/mean_critique_length"].append(np.mean(critique_lengths))
        
        # 2. Revision Efficiency
        # if len(wrong_idxs) > 0 and len(revised_rewards_per_func) > 0:
        #     # Success rate: how many revisions achieved correct answer
        #     revision_correctness = (revised_rewards_per_func[:, 0] > 0.5).cpu()
        #     revision_success_rate = revision_correctness.float().mean().item()
        #     self._metrics[mode]["critique/revision_success_rate"].append(revision_success_rate)
            
        #     # Track which initially wrong answers became correct
        #     initially_wrong_idxs = [idx for idx in wrong_idxs if initial_rewards_per_func[idx, 0] <= 0.5]
        #     if initially_wrong_idxs:
        #         fixed_count = sum(1 for i, idx in enumerate(wrong_idxs) 
        #                         if idx in initially_wrong_idxs and revised_rewards_per_func[i, 0] > 0.5)
        #         fix_rate = fixed_count / len(initially_wrong_idxs)
        #         self._metrics[mode]["critique/wrong_to_correct_rate"].append(fix_rate)
        
        # 3. Confidence Calibration
        # correct_idxs = [i for i in range(len(initial_rewards_per_func)) if i not in wrong_idxs]
        
        # if correct_idxs:
        #     # Confidence for correctly identified correct answers
        #     correct_confidence = initial_sce_raw[correct_idxs].mean().item()
        #     self._metrics[mode]["critique/correct_confidence_mean"].append(correct_confidence)
        
        # if wrong_idxs:
        #     # Confidence for correctly identified wrong answers
        #     wrong_confidence = initial_sce_raw[wrong_idxs].mean().item()
        #     self._metrics[mode]["critique/wrong_confidence_mean"].append(wrong_confidence)
            
        #     # Confidence gap
        #     if correct_idxs:
        #         confidence_gap = correct_confidence - wrong_confidence
        #         self._metrics[mode]["critique/confidence_gap"].append(confidence_gap)
        
        # # Additional improvement metrics
        # if len(wrong_idxs) > 0:
        #     # Improvement metrics for revised responses
        #     initial_wrong_rewards = initial_rewards_per_func[wrong_idxs]
        #     min_len = min(len(revised_rewards_per_func), len(initial_wrong_rewards))
        #     for i, reward_func_name in enumerate(self.reward_func_names):
        #         if len(revised_rewards_per_func) > 0:
        #             improvement = revised_rewards_per_func[:min_len, i] - initial_wrong_rewards[:min_len, i]
        #             self._metrics[mode][f"critique/{reward_func_name}/mean_improvement"].append(torch.nanmean(improvement).item())
        #             self._metrics[mode][f"critique/{reward_func_name}/improvement_rate"].append((improvement > 0).float().mean().item())
            
        #     # SCE improvement
        #     initial_sce_wrong = initial_sce_raw[wrong_idxs]
        #     final_sce_wrong = final_sce_raw[wrong_idxs]
        #     sce_improvement = final_sce_wrong - initial_sce_wrong
        #     self._metrics[mode]["critique/sce_mean_improvement"].append(sce_improvement.mean().item())
        #     self._metrics[mode]["critique/sce_improvement_std"].append(sce_improvement.std().item())
        
        # Critique advantage statistics
        if hasattr(self, '_critique_stats'):
            self._metrics[mode]["critique/advantage_mean"].append(self._critique_stats["mean"])
            self._metrics[mode]["critique/advantage_std"].append(self._critique_stats["std"])
        
        # Response length changes
        if len(wrong_idxs) > 0:
            initial_lengths = initial_completion_mask[wrong_idxs].sum(dim=1)
            final_lengths = final_completion_mask[wrong_idxs].sum(dim=1)
            length_change = final_lengths.float() - initial_lengths.float()
            self._metrics[mode]["critique/mean_length_change"].append(length_change.mean().item())
            self._metrics[mode]["critique/length_increase_rate"].append((length_change > 0).float().mean().item())

    def _compute_final_logps(self, prompt_ids, completion_ids, completion_mask, batch_size):
        """Compute old and reference log probabilities for final completions."""
        device = self.accelerator.device
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_ids != self.processing_class.pad_token_id, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            # Compute old_per_token_logps if needed
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None
            
            # Compute ref_per_token_logps if needed
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )
        
        return old_per_token_logps, ref_per_token_logps
    
    def _log_textual_outputs(self, prompts_text, initial_completions_text, critique_text,
                        final_completions_text, initial_rewards_per_func, final_rewards_per_func):
        """Log textual outputs for analysis."""
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["initial_response"].extend(gather_object(initial_completions_text))
        self._textual_logs["critique"].extend(gather_object(critique_text))
        self._textual_logs["final_response"].extend(gather_object(final_completions_text))
        
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(gather(initial_rewards_per_func)[:, i].tolist())
            self._textual_logs["rewards"][f"{name}_final"].extend(gather(final_rewards_per_func)[:, i].tolist())  # figure out later. 

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            return self.compute_liger_loss(model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"].unsqueeze(1)  # (B, 1)
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    # self._textual_logs["initial_response"],
                    # self._textual_logs["critique"],
                    self._textual_logs["final_response"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                # table = {
                #     "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                #     "prompt": self._textual_logs["prompt"],
                #     "completion": self._textual_logs["completion"],
                #     **self._textual_logs["rewards"],
                # }

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "initial_response": self._textual_logs["initial_response"],
                    "critique": self._textual_logs["critique"],
                    "final_response": self._textual_logs["final_response"],
                    **self._textual_logs["rewards"],
                }
                
                # Add critique and revised completion to wandb logs if available
                # if "critique" in self._textual_logs:
                #     table["critique"] = self._textual_logs["critique"]
                # if "revised" in self._textual_logs:
                #     table["revised"] = self._textual_logs["revised"]
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
