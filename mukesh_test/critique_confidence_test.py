
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import ipdb 


def reliability_curve(df: pd.DataFrame,
                      conf_col: str,
                      n_bins: int = 10,
                      show_plot: bool = True):
    """
    Bin the confidence signal and compute
        p_wrong = P(verdict == -1 | confidence in bin)
    Returns (edges, p_wrong) where
        edges: ndarray shape (n_bins+1,)
        p_wrong: ndarray shape (n_bins,)
    """
    edges = np.quantile(df[conf_col].values, np.linspace(0, 1, n_bins + 1))
    df["bin"] = pd.cut(df[conf_col], bins=edges,
                       labels=False, include_lowest=True)

    p_wrong = (
        df.groupby("bin")["verdict"]
          .apply(lambda s: (s == -1).mean())
          .reindex(range(n_bins), fill_value=np.nan)
          .values
    )

    p_correct = (
        df.groupby("bin")["verdict"]
          .apply(lambda s: (s == 1).mean())
          .reindex(range(n_bins), fill_value=np.nan)
          .values
    )

    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(range(n_bins), p_wrong, marker="o", label='critic says wrong')
        plt.plot(range(n_bins), p_correct, marker="o", label='critic says correct')
        plt.xticks(range(n_bins),
                   [f"{edges[i]:.2f}" for i in range(n_bins)],
                   rotation=45, ha="right", fontsize=8)
        plt.xlabel(f"{conf_col} bin lower edge")
        plt.ylabel("P(critic = correct/wrong)")
        plt.title("Reliability curve")
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.legend()
        # plt.show()
        plt.savefig('reliability.png')

    return edges, p_wrong


import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def reliability_curve_with_truth(df: pd.DataFrame,
                                 conf_col: str,
                                 truth_col: str,
                                 verdict_col: str = "verdict",
                                 n_bins: int = 10,
                                 show_plot: bool = True):
    """
    confidence → two reliability curves:
        • critic verdict vs. confidence
        • ground-truth error vs. confidence
    Assumes:
        verdict_col  = ±1  (+1 = critic thinks correct)
        truth_col    = ±1  (+1 = actually correct)
    """
    # ---------------- bin edges -----------------
    edges = np.quantile(df[conf_col].values,
                        np.linspace(0, 1, n_bins + 1))
    df["bin"] = pd.cut(df[conf_col], bins=edges,
                       labels=False, include_lowest=True)

    # ------------- conditional frequencies -------
    by_bin = df.groupby("bin")
    p_critic_wrong = by_bin[verdict_col].apply(lambda s: (s == -1).mean())
    p_truth_wrong  = by_bin[truth_col].apply(lambda s: (s == -1).mean())

    if show_plot:
        xs = p_critic_wrong.index
        plt.figure(figsize=(6, 4))
        plt.plot(xs, p_critic_wrong.values, marker="o",
                 label="P(critic = wrong)")
        plt.plot(xs, p_truth_wrong.values,  marker="s",
                 label="P(truth  = wrong)")
        plt.xticks(xs, [f"{edges[i]:.2f}" for i in xs],
                   rotation=45, ha="right", fontsize=8)
        plt.xlabel(f"{conf_col} bin lower edge")
        plt.ylabel("probability")
        plt.title("Reliability curves: critic vs. ground-truth")
        plt.legend()
        plt.grid(ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig("reliability_with_truth.png", dpi=150)

    return edges, p_critic_wrong.values, p_truth_wrong.values


def choose_tau(edges, p_wrong, tolerance: float):
    """
    First bin edge whose P(wrong) ≤ tolerance.
    If none qualifies, return the rightmost edge.
    """
    for idx, p in enumerate(p_wrong):
        if np.isfinite(p) and p <= tolerance:
            return float(edges[idx])
    return float(edges[-1])


results_file = 'self_sc_init_response.json'


# Load the results
with open(results_file, 'r') as f:
    results = json.load(f)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(results)


# ipdb.set_trace()
model_said_wrong = [idx for idx in range(len(df)) if ('the answer is incorrect' in df['critique'][idx][0].lower() or 'the answer is wrong' in df['critique'][idx][0].lower())]

df['verdict'] = [1 if idx not in model_said_wrong else -1 for idx in range(len(df))]
df["truth"] = np.where(df["initial_accuracy"] == 1, +1, -1)

# clean up the confidence list
df['confidence_list'] = [df['confidence_list'][idx][0] for idx in range(len(df))]


edges, p_wrong = reliability_curve(df, 'confidence_list', show_plot=True)
tau = choose_tau(edges, p_wrong, tolerance=0.25)
print(tau)

# edges, p_c, p_t = reliability_curve_with_truth(
#     df,
#     conf_col="confidence_list",   # or critic_cert, entropy, etc.
#     truth_col="truth",
#     verdict_col="verdict",
#     n_bins=10
# )

cm = confusion_matrix(df["truth"], df["verdict"], labels=[+1, -1])
ConfusionMatrixDisplay(cm,
    display_labels=["Correct", "Wrong"]
).plot(cmap="Blues")
plt.title("Critic vs. Ground-Truth"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# plot confidence for critique when model said wrong, model said correct, model said wrong when the solution was correct, model said correct when the solution was wrong, 
# model said correct when the solution was correct, and model said incorrect when the solution was incorrect

CONF_COL = 'confidence_list'
# Masks
crit_wrong   = df["verdict"] == -1
crit_correct = df["verdict"] == +1

mask_fn = crit_wrong   & (df["truth"] == 1)     # False-negative
mask_fp = crit_correct & (df["truth"] == -1)     # False-positive
mask_tp = crit_correct & (df["truth"] == 1)     # True-positive
mask_tn = crit_wrong   & (df["truth"] == -1)     # True-negative


categories = {
    "Critic WRONG":   crit_wrong,
    "Critic CORRECT": crit_correct,
    "True-Positive":  mask_tp,
    "False-Positive": mask_fp,
    "False-Negative": mask_fn,
    "True-Negative":  mask_tn,
}

# -------------------------------------------------------------------
# 2) Mean ± 1 SD confidence for each slice
# -------------------------------------------------------------------
labels, means, sds, counts = [], [], [], []
for name, m in categories.items():
    vals = df.loc[m, CONF_COL]
    labels.append(name)
    means.append(vals.mean() if len(vals) else np.nan)
    sds.append(vals.std(ddof=0) if len(vals) else np.nan)
    counts.append(len(vals))

# -------------------------------------------------------------------
# 3) Bar plot
# -------------------------------------------------------------------
x = np.arange(len(labels))
plt.figure(figsize=(8, 4.5))
bars = plt.bar(x, means, yerr=sds, alpha=0.75, capsize=5)
plt.xticks(x, labels, rotation=15)
plt.ylabel("Mean confidence  (±1 SD)")
plt.title(f"{CONF_COL} by verdict × ground-truth category")

# annotate n on each bar
for bar, n in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"n={n}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig('critic_performance.png')

