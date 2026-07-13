"""Integration test for the majority-vote reward using the REAL math_verify.

Unlike ``test_majority_vote_reward.py`` (which stubs ``math_verify`` for a
dependency-free run), this test requires the actual ``math_verify`` and
``latex2sympy2_extended`` packages to be installed (they are part of the
``openr1`` environment). It verifies the behaviour that matters most in
production: that genuinely equivalent LaTeX answer forms are clustered into a
single vote by the real verifier.

Only the unrelated code-execution provider imports are stubbed, since importing
``open_r1.rewards`` triggers them at module load and they are irrelevant here.

Run (inside the openr1 venv):  python tests/test_majority_vote_reward_real.py
"""

import importlib.util
import os
import sys
import types


def _stub_only_code_providers():
    """Stub the code-provider imports so rewards.py imports cleanly, but leave
    math_verify / latex2sympy2_extended as the REAL installed packages."""
    open_r1 = sys.modules.get("open_r1") or types.ModuleType("open_r1")
    open_r1.__path__ = []
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
    # Hard requirement: real math_verify must be importable.
    import math_verify  # noqa: F401
    import latex2sympy2_extended  # noqa: F401

    _stub_only_code_providers()
    here = os.path.dirname(os.path.abspath(__file__))
    rewards_path = os.path.join(here, "..", "src", "open_r1", "rewards.py")
    spec = importlib.util.spec_from_file_location("open_r1.rewards", rewards_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_r1.rewards"] = module
    spec.loader.exec_module(module)
    return module


rewards = _load_rewards()

_failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


def _comp(text):
    return [{"role": "assistant", "content": text}]


def _user_prompt(q):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": q}]


def test_real_equivalent_latex_forms_cluster():
    """\\frac{1}{2}, 1/2, and 0.5 should all be ONE answer under real math_verify,
    out-voting a different answer that appears twice."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Compute the value.")
    completions = [
        _comp("answer is \\boxed{\\frac{1}{2}}"),
        _comp("so \\boxed{0.5}"),
        _comp("hence \\boxed{1/2}"),
        _comp("I think \\boxed{3}"),
        _comp("maybe \\boxed{3}"),
    ]
    prompts = [q] * 5
    out = reward(completions=completions, prompts=prompts)
    # The half-cluster has 3 votes vs 2 for "3", so the first three win.
    check(
        "real_equivalent_latex_forms_cluster",
        out == [1.0, 1.0, 1.0, 0.0, 0.0],
        detail=f"got {out}",
    )


def test_real_distinct_answers_split():
    """Distinct answers must NOT be merged; clean 3-vs-1 plurality."""
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Q")
    completions = [
        _comp("\\boxed{7}"), _comp("\\boxed{7}"), _comp("\\boxed{7}"), _comp("\\boxed{8}"),
    ]
    prompts = [q] * 4
    out = reward(completions=completions, prompts=prompts)
    check("real_distinct_answers_split", out == [1.0, 1.0, 1.0, 0.0], detail=f"got {out}")


def test_real_unparseable_no_vote():
    reward = rewards.get_majority_vote_reward(min_agreement=0.0)
    q = _user_prompt("Q")
    completions = [
        _comp("\\boxed{42}"), _comp("I am not sure, no final answer."), _comp("\\boxed{42}"),
    ]
    prompts = [q] * 3
    out = reward(completions=completions, prompts=prompts)
    check("real_unparseable_no_vote", out == [1.0, 0.0, 1.0], detail=f"got {out}")


if __name__ == "__main__":
    test_real_equivalent_latex_forms_cluster()
    test_real_distinct_answers_split()
    test_real_unparseable_no_vote()
    print()
    if _failures:
        print(f"{len(_failures)} test(s) FAILED: {_failures}")
        sys.exit(1)
    print("All REAL-math_verify majority-vote tests passed.")
