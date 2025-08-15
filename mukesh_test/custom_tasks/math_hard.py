from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def prompt_fn(line, task_name: str=None):
    return Doc(
        task_name=task_name,
        choices=[str(line["solution"])],
        gold_index=0,
        query=f"Question: {line['problem']}\nAnswer:",
    )


task = LightevalTaskConfig(
    name="math_hard",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="lighteval/MATH-Hard",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metric=[
        Metrics.expr_gold_metric, Metrics.math_pass_at_1_1n,
    ],
    stop_sequence=["Problem:"],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [task]