"""Custom lighteval tasks for greedy math evals (see eval_math.sh).

- custom|minerva_math: math-ai/minervamath, absent from lighteval's registry.
- lighteval|math_500: overrides the default task to drop the math_pass@1:4_samples
  metric — vLLM rejects n=4 sampling when temperature is 0 (greedy).

Usage: lighteval vllm <model_args> "custom|minerva_math|0|0" --custom-tasks open_r1.minerva_math_task

Must be passed as a module path, not a file path: with datasets 5.x installed,
lighteval's file-path loader crashes on a removed `trust_remote_code` kwarg, but the
module-import fallback works.
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Same template as lighteval's math_500 / aime24 tasks: math-verify needs the
# final answer in a box to extract it reliably.
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()


def minerva_math_prompt(line, task_name: str = None):
    # Gold is a bare answer string (number, scientific notation, or LaTeX),
    # so this mirrors aime_prompt_fn rather than math_500 (which uses the full solution).
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


minerva_math = LightevalTaskConfig(
    name="minerva_math",
    suite=["custom"],
    prompt_function=minerva_math_prompt,
    hf_repo="math-ai/minervamath",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[Metrics.math_pass_at_1_1n],
    version=1,
)

math_500_greedy = LightevalTaskConfig(
    name="math_500",
    suite=["lighteval"],
    prompt_function=prompt.math_500,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[Metrics.math_pass_at_1_1n],
    version=2,
)

TASKS_TABLE = [minerva_math, math_500_greedy]
