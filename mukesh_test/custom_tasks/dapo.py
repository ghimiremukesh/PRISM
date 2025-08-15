from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from aenum import extend_enum
import numpy as np


from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
    multilingual_extractive_match_metric,
)
from lighteval.utils.language import Language
from lighteval.metrics.metrics_sample import PassAtK
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# ---------- NORMALIZERS (training-style) ----------
GOLD_PARSE_KW = dict(extraction_mode="first_match")
PRED_PARSE_KW = dict(
    extraction_config=[
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                equations=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ],
    extraction_mode="first_match",
)

def _norm_gold_train(s: str):
    return parse(s, **GOLD_PARSE_KW)

def _norm_pred_train(s: str):
    return parse(s, **PRED_PARSE_KW)

# ---------- SCORER ----------
def _score_verify(gold, pred, formatted_doc) -> float:
    try:
        return float(verify(gold, pred))
    except Exception:
        return 0.0

math_pass_at_1_7n = SampleLevelMetric(
    metric_name="math_pass@1:7_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=7,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Uses sympy for comparison
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

    

grpo_math_reward_new = SampleLevelMetric(
    metric_name="grpo_math_reward_new",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        strip_strings=True,
        normalize_gold=_norm_gold_train,   # CHANGED
        normalize_pred=_norm_pred_train,   # CHANGED 
        sample_scoring_function=_score_verify,  # CHANGED
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

extend_enum(Metrics, "math_pass_at_1_7n", math_pass_at_1_7n)
extend_enum(Metrics, "grpo_math_reward_new", grpo_math_reward_new)




def prompt_fn(line, task_name: str=None):
    return Doc(
        task_name=task_name,
        choices=[str(line["solution"])],
        gold_index=0,
        query=f"Question: {line['prompt']}\nAnswer:",
        # query=f"Question: {line['problem']}\nAnswer:",

    )


task = LightevalTaskConfig(
    name="dapo",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="open-r1/DAPO-Math-17k-Processed",
    # hf_repo='HuggingFaceH4/aime_2024',
    hf_subset="en",
    # hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        # Metrics.strict_math_pass_at_1_1n,
        Metrics.grpo_math_reward_new,
        Metrics.grpo_math_reward,
        # Metrics.math_pass_at_1_16n
    ],
    stop_sequence=[],
    num_samples=16,
    trust_dataset=True,
    version=0,
)

taskb = LightevalTaskConfig(
    name="dapo_cn",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="open-r1/DAPO-Math-17k-Processed",
    # hf_repo='HuggingFaceH4/aime_2024',
    hf_subset="cn",
    # hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[
        Metrics.math_pass_at_1_1n,
        # Metrics.strict_math_pass_at_1_1n,
        Metrics.grpo_math_reward_new,
        Metrics.math_pass_at_1_16n
    ],
    stop_sequence=[],
    num_samples=16,
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [task, taskb]




# logger = logging.getLogger(__name__)

# ------------------------- helpers -------------------------
# class MetricUseCase(str, Enum):
#     # General
#     ACCURACY = auto()
#     PERPLEXITY = auto()
#     # Task specific
#     CODE = auto()
#     COPYRIGHT = auto()
#     MATH = auto()
#     REASONING = auto()
#     SOCIAL_IMPACTS = auto()
#     SUMMARIZATION = auto()
#     TRANSLATION = auto()
#     NONE = auto()


# _BOX_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")

# def _has_multiple_boxes(text: str) -> bool:
#     return len(_BOX_RE.findall(text)) > 1

# @timeout(2)
# def _add_to_specifics_with_timeout(
#     formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
# ) -> None:
#     if formatted_doc.specific is None:
#         formatted_doc.specific = {}
#     formatted_doc.specific["extracted_predictions"] = [
#         str(pred) for preds in extracted_predictions for pred in preds
#     ]
#     formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

# # ------------------------- metric factory -------------------------

# def strict_multilingual_extractive_match_metric(
#     language: Language = Language.ENGLISH,
#     gold_extraction_target: Sequence[ExtractionTarget] = [ExprExtractionConfig(),],
#     pred_extraction_target: Sequence[ExtractionTarget] = [ExprExtractionConfig(), LatexExtractionConfig()],
#     aggregation_function: Callable[[list[float]], float] = max,
#     fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
#     extraction_mode: Literal["first_match", "any_match"] = "any_match",
#     precision: int = 6,
#     timeout_seconds: int = 5,
# ) -> SampleLevelMetric:
#     """
#     Same as multilingual_extractive_match_metric, but if ANY prediction string
#     contains more than one \\boxed{...}, the sample score is forced to 0.0.
#     """

#     def sample_level_fn(doc: Doc, model_response: ModelResponse) -> float:
#         golds = doc.get_golds()
#         predictions = model_response.text

#         # --- strict rule: multiple boxed answers => 0 ------------------------
#         if any(_has_multiple_boxes(p) for p in predictions):
#             return 0.0

#         gold_extraction_regexes = get_extraction_regexes(doc, gold_extraction_target, language)
#         pred_extraction_regexes = get_extraction_regexes(doc, pred_extraction_target, language)

#         extracted_predictions = [
#             extract_target_from_pred(pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
#             for pred in predictions
#         ]
#         extracted_golds = [
#             extract_target_from_pred(gold, gold_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
#             for gold in golds
#         ]

#         # Assert on empty gold and warn on empty pred
#         if any(len(g) == 0 for g in extracted_golds):
#             logger.warning(f"We did not manage to extract a gold in the correct format. Gold: {golds}")
#             extracted_golds = [[gold] for gold in golds]

#         if all(len(p) == 0 for p in extracted_predictions):
#             logger.warning(
#                 f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
#             )

#         try:
#             _add_to_specifics_with_timeout(doc, extracted_predictions, extracted_golds)
#         except Exception:
#             logger.warning("Timeout when adding extracted predictions and golds to specific")

#         return aggregation_function(
#             [
#                 1.0
#                 if any(
#                     compare_gold_target(gold, pred, precision, timeout_seconds=timeout_seconds)
#                     for gold in extracted_golds
#                 )
#                 else 0.0
#                 for pred in extracted_predictions
#             ]
#         )

#     return SampleLevelMetric(
#         metric_name="extractive_match",
#         sample_level_fn=sample_level_fn,
#         category="GENERATIVE",
#         corpus_level_fn=np.mean,
#         higher_is_better=True,
#         use_case=MetricUseCase.ACCURACY,
#     )



# # --- define the new metric object -------------------------------------------
# math_pass_at_1_1n_strict = SampleLevelMetric(
#     metric_name="math_pass_at_1_1n_strict",
#     sample_level_fn=PassAtK(
#         k=1, 
#         n=1, 
#         strip_strings=True,
#         sample_scoring_function = lambda doc, model_response: strict_multilingual_extractive_match_metric(
#             language=Language.English,
#             pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()], 
#             precision=6,
#         ).sample_level_fn(doc, model_response),
#     ).compute, 
#     category="GENERATIVE",
#     corpus_level_fn=np.mean,
#     higher_is_better=True,
#     use_case=MetricUseCase.ACCURACY,
# )

# # --- register it -------------------------------------------------------------
# extend_enum(Metrics, "math_pass_at_1_1n_strict", math_pass_at_1_1n_strict)