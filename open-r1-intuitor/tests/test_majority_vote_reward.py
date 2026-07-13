"""Unit tests for the label-free majority-vote (TTRL-style) reward.

These tests exercise the REAL ``get_majority_vote_reward`` implementation in
``open_r1.rewards`` (grouping by prompt, clustering by equivalence, plurality
selection, the ``min_agreement`` gate, handling of unparseable answers). The
heavy external dependencies that ``rewards.py`` imports at module load
(``math_verify``, ``latex2sympy2_extended``, and the code-execution providers)
are replaced with lightweight stubs so the test runs without GPUs or the full
training environment.

What is stubbed and why:
  * ``math_verify.parse``  -> extracts the ``\\boxed{...}`` payload and returns
    it as a single-element list (``[]`` if no box). This is deterministic so the
    vote tallies are predictable.
  * ``math_verify.verify`` -> equality over the parsed payloads, with a small
    explicit equivalence map so we can also test that two *different* string
    forms of the same answer (e.g. ``1/2`` and ``0.5``) are clustered into one
    vote -- which is exactly what real ``math_verify`` does in production.
Everything else (the grouping, plurality, threshold, and 0.0-for-unparseable
behaviour) is the project's own code under test.

Run:  python3 tests/test_majority_vote_reward.py
"""

import importlib.util
import os
import re
import sys
import types


# --------------------------------------------------------------------------- #
# Build stub modules BEFORE importing open_r1.rewards.
# --------------------------------------------------------------------------- #

# Answers we want the stub `verify` to treat as mathematically equal, even
# though their string forms differ (mirrors math_verify's equivalence).
_EQUIVALENCE_CLASSES = [
    {"1/2", "0.5"},
    {"2", "2.0"},
]


def _canonical(token: str) -> str:
    for cls in _EQUIVALENCE_CLASSES:
        if token in cls:
            return sorted(cls)[0]
    return token


_BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _stub_parse(content, *args, **kwargs):
    """Return [payload] for the last \\boxed{...} in `content`, else []."""
    matches = _BOX_RE.findall(content if isinstance(content, str) else str(content))
    if not matches:
        return []
    return [matches[-1].strip()]


def _stub_verify(a, b):
    """Equality over parsed payloads, honouring the equivalence map."""
    if not a or not b:
        return False
    return _canonical(a[0]) == _canonical(b[0])


def _install_stubs():
    # math_verify
    mv = types.ModuleType("math_verify")
    mv.parse = _stub_parse
    mv.verify = _stub_verify
    mv.LatexExtractionConfig = lambda *a, **k: None
    sys.modules["math_verify"] = mv

    # latex2sympy2_extended
    l2s = types.ModuleType("latex2sympy2_extended")
    l2s.NormalizationConfig = lambda *a, **k: None
    sys.modules["latex2sympy2_extended"] = l2s

    # Package stubs so the relative imports in rewards.py resolve from sys.modules
    # without touching the filesystem.
    open_r1 = types.ModuleType("open_r1")
    open_r1.__path__ = []  # mark as package
    sys.modules["open_r1"] = open_r1

    utils = types.ModuleType("open_r1.utils")
    utils.__path__ = []
    sys.modules["open_r1.utils"] = utils

    code_providers = types.ModuleType("open_r1.utils.code_providers")
    code_providers.get_provider = lambda *a, **k: None
    sys.modules["open_r1.utils.code_providers"] = code_providers

    ioi = types.ModuleType("open_r1.utils.ioi")
    for name in (
        "SubtaskResult",
        "add_includes",
        "get_morph_client_from_env",
        "get_piston_client_from_env",
        "score_subtask",
    ):
        setattr(ioi, name, lambda *a, **k: None)
    sys.modules["open_r1.utils.ioi"] = ioi


def _load_rewards():
    _install_stubs()
    here = os.path.dirname(os.path.abspath(__file__))
    rewards_path = os.path.join(here, "..", "src", "open_r1", "rewards.py")
    spec = importlib.util.spec_from_file_location("open_r1.rewards", rewards_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_r1.rewards"] = module
    spec.loader.exec_module(module)
    return module


rewards = _load_rewards()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _comp(text):
    """Wrap text in the conversational completion structure the reward expects."""
    return [{"role": "assistant", "content": text}]


def _user_prompt(question):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": question},
    ]


def _boxed(ans):
    return f"Some reasoning here.\nThe final answer is \\boxed{{{ans}}}"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

_failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        _failures.append(name)


def test_single_group_plurality():
    """3 votes for '4', 1 for '5' -> the three '4's get 1.0, the '5' gets 0.0."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("What is 2+2?")
    prompts = [q, q, q, q]
    completions = [_comp(_boxed("4")), _comp(_boxed("4")), _comp(_boxed("5")), _comp(_boxed("4"))]
    out = reward(completions=completions, prompts=prompts)
    check("single_group_plurality", out == [1.0, 1.0, 0.0, 1.0])


def test_equivalent_forms_cluster():
    """'1/2' and '0.5' must count as the SAME answer (3 votes) and all win."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Half?")
    prompts = [q, q, q, q]
    completions = [_comp(_boxed("1/2")), _comp(_boxed("0.5")), _comp(_boxed("1/2")), _comp(_boxed("7"))]
    out = reward(completions=completions, prompts=prompts)
    check("equivalent_forms_cluster", out == [1.0, 1.0, 1.0, 0.0])


def test_unparseable_gets_zero_and_no_vote():
    """An answer with no \\boxed{} casts no vote and receives 0.0."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Q")
    prompts = [q, q, q]
    completions = [_comp(_boxed("9")), _comp("no box at all here"), _comp(_boxed("9"))]
    out = reward(completions=completions, prompts=prompts)
    check("unparseable_gets_zero_and_no_vote", out == [1.0, 0.0, 1.0])


def test_two_prompts_grouped_independently():
    """Two different prompts in one batch must vote independently."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    qa = _user_prompt("A")
    qb = _user_prompt("B")
    # Group A: majority '1'; Group B: majority '8'
    prompts = [qa, qa, qb, qb]
    completions = [_comp(_boxed("1")), _comp(_boxed("2")), _comp(_boxed("8")), _comp(_boxed("8"))]
    out = reward(completions=completions, prompts=prompts)
    check("two_prompts_grouped_independently", out == [1.0, 0.0, 1.0, 1.0])


def test_non_contiguous_same_prompt_still_grouped():
    """Even if a prompt's samples are interleaved, key-based grouping merges them."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    qa = _user_prompt("A")
    qb = _user_prompt("B")
    prompts = [qa, qb, qa, qb]  # interleaved
    completions = [_comp(_boxed("1")), _comp(_boxed("8")), _comp(_boxed("1")), _comp(_boxed("9"))]
    out = reward(completions=completions, prompts=prompts)
    # A -> both '1' win; B -> '8' vs '9' tie, first-encountered ('8') wins deterministically
    check("non_contiguous_same_prompt_still_grouped", out == [1.0, 1.0, 1.0, 0.0])


def test_min_agreement_gate_blocks_low_consensus():
    """With a 0.6 threshold, a 2/2/1 split (top=0.4) yields no signal at all."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.6)
    q = _user_prompt("Q")
    prompts = [q] * 5
    completions = [
        _comp(_boxed("1")), _comp(_boxed("1")),
        _comp(_boxed("2")), _comp(_boxed("2")),
        _comp(_boxed("3")),
    ]
    out = reward(completions=completions, prompts=prompts)
    check("min_agreement_gate_blocks_low_consensus", out == [0.0, 0.0, 0.0, 0.0, 0.0])


def test_min_agreement_gate_allows_strong_consensus():
    """Same threshold, but a 4/1 split (top=0.8) passes -> the 4 win."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.6)
    q = _user_prompt("Q")
    prompts = [q] * 5
    completions = [
        _comp(_boxed("1")), _comp(_boxed("1")), _comp(_boxed("1")), _comp(_boxed("1")),
        _comp(_boxed("2")),
    ]
    out = reward(completions=completions, prompts=prompts)
    check("min_agreement_gate_allows_strong_consensus", out == [1.0, 1.0, 1.0, 1.0, 0.0])


def test_all_unparseable_group_is_all_zero():
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Q")
    prompts = [q, q]
    completions = [_comp("nope"), _comp("still nope")]
    out = reward(completions=completions, prompts=prompts)
    check("all_unparseable_group_is_all_zero", out == [0.0, 0.0])


def test_label_free_ignores_solution_kwarg():
    """The reward must NOT use ground truth even if `solution` is passed in."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Q")
    prompts = [q, q, q]
    # Plurality answer is '5' (wrong); ground truth says '4'. Reward follows the
    # plurality, proving it ignores `solution`.
    completions = [_comp(_boxed("5")), _comp(_boxed("5")), _comp(_boxed("4"))]
    out = reward(completions=completions, prompts=prompts, solution=["\\boxed{4}"] * 3)
    check("label_free_ignores_solution_kwarg", out == [1.0, 1.0, 0.0])


def test_string_prompts_supported():
    """Prompts may be plain strings, not just conversational lists."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    prompts = ["question one", "question one", "question two"]
    completions = [_comp(_boxed("3")), _comp(_boxed("3")), _comp(_boxed("9"))]
    out = reward(completions=completions, prompts=prompts)
    check("string_prompts_supported", out == [1.0, 1.0, 1.0])


if __name__ == "__main__":
    test_single_group_plurality()
    test_equivalent_forms_cluster()
    test_unparseable_gets_zero_and_no_vote()
    test_two_prompts_grouped_independently()
    test_non_contiguous_same_prompt_still_grouped()
    test_min_agreement_gate_blocks_low_consensus()
    test_min_agreement_gate_allows_strong_consensus()
    test_all_unparseable_group_is_all_zero()
    test_label_free_ignores_solution_kwarg()
    test_string_prompts_supported()
    print()
    if _failures:
        print(f"{len(_failures)} test(s) FAILED: {_failures}")
        sys.exit(1)
    print("All majority-vote reward tests passed.")
