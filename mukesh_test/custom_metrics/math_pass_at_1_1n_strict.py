# my_metrics/math_pass_at_1_1n_strict.py
import re
from aenum import extend_enum
from typing import List
from lighteval.metrics import Metrics
from lighteval.metrics.utils import MetricUseCase
from lighteval.metrics.utils.metric_utils import SampleLevelMetric

# --- helpers -----------------------------------------------------------------
_BOX_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")

def _multi_boxed(pred_text: str) -> bool:
    return len(_BOX_RE.findall(pred_text)) > 1

# --- grab the base metric ----------------------------------------------------
# Metrics is an Enum whose values are Metric objects (SampleLevelMetric here)
_base = Metrics.math_pass_at_1_1n  # the existing one

def _strict_sample_fn(predictions: List[str], formatted_doc, **kwargs) -> float | bool | dict:
    """
    Delegate to the original metric's sample fn, but force 0 if >1 boxed answers.
    Signature follows the docs: (predictions: list[str], formatted_doc: Doc, **kwargs)
    """
    # If multiple boxed, short-circuit to 0 / False
    if _multi_boxed(predictions[0]):
        # base metric returns a bool/float; we mirror that type
        return 0.0

    # Otherwise, call the base metric's sample function
    return _base.value.sample_level_fn(predictions, formatted_doc, **kwargs)

# Reuse the same corpus aggregator (usually np.mean wrapped)
_strict_corpus_fn = _base.value.corpus_level_fn

# --- define the new metric object -------------------------------------------
math_pass_at_1_1n_strict = SampleLevelMetric(
    metric_name="math_pass_at_1_1n_strict",
    higher_is_better=_base.value.higher_is_better,
    category=_base.value.category,
    use_case=_base.value.use_case,
    sample_level_fn=_strict_sample_fn,
    corpus_level_fn=_strict_corpus_fn,
)

# --- register it -------------------------------------------------------------
extend_enum(Metrics, "math_pass_at_1_1n_strict", math_pass_at_1_1n_strict)

if __name__ == "__main__":
    print("Imported math_pass_at_1_1n_strict")
