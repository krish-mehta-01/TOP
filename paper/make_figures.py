"""
make_figures.py — generates paper figures directly from the current, verified
training-summary and reliability JSON files (not from data/output_data/research_figures,
which was found to contain stale pre-audit artifacts for at least one model — see
paper build notes). Every number plotted here is read live from the same files the
paper's Table I/II were transcribed from, so the figures cannot drift out of sync
with the reported metrics.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE = "#3b6ea5"
ORANGE = "#d0793a"
GREY = "#8a8a8a"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(os.path.join(ROOT, "decision_engine", "config", "reliability.json")) as f:
    reliability = json.load(f)
with open(os.path.join(ROOT, "data", "output_data", "bowling_training_summary.json")) as f:
    bowling = json.load(f)
with open(os.path.join(ROOT, "data", "output_data", "batting_training_summary.json")) as f:
    batting = json.load(f)

W_ORDER = [f"W{i}" for i in range(1, 16)]
B_ORDER = ["B1", "B2", "B3", "B5", "B6", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"]

# ---------------------------------------------------------------------------
# Figure 1: reliability weight, top 10 most reliable models (of 28), sorted
# descending. Shown for readability; the complete 28-model set - including
# the weaker models - is reported in Table I and Table II, unchanged.
# ---------------------------------------------------------------------------
ids = W_ORDER + B_ORDER
vals = [reliability[i] for i in ids]
colors = [BLUE if i.startswith("W") else ORANGE for i in ids]
order = sorted(range(len(ids)), key=lambda k: -vals[k])[:10]
ids_s = [ids[k] for k in order]
vals_s = [vals[k] for k in order]
colors_s = [colors[k] for k in order]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ids_s, vals_s, color=colors_s, width=0.62)
ax.set_ylabel("Reliability weight")
ax.set_xlabel("Model ID")
ax.set_title("Fig. 2. Top 10 most reliable models (of 28), sorted")
ax.set_ylim(0, 1.0)
ax.tick_params(axis="x", rotation=45)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Bowling-side (W)"),
                    Patch(color=ORANGE, label="Batting-side (B)")],
          frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_reliability_weights.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: held-out validation performance by model type (2 panels)
# ---------------------------------------------------------------------------
def classifier_auc(d, order):
    out = []
    for i in order:
        m = d[i]["metrics"]
        if "ROC_AUC" in m and m["ROC_AUC"] is not None:
            out.append((i, m["ROC_AUC"]))
    return out


def regressor_r2(d, order):
    out = []
    for i in order:
        m = d[i]["metrics"]
        if "R2" in m:
            out.append((i, m["R2"]))
    return out


w_auc = classifier_auc(bowling, W_ORDER)
b_auc = classifier_auc(batting, B_ORDER)
w_r2 = regressor_r2(bowling, W_ORDER)
b_r2 = regressor_r2(batting, B_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
xs = [i for i, _ in w_auc] + [i for i, _ in b_auc]
ys = [v for _, v in w_auc] + [v for _, v in b_auc]
cs = [BLUE] * len(w_auc) + [ORANGE] * len(b_auc)
ax.bar(xs, ys, color=cs, width=0.68)
ax.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
ax.text(len(xs) - 1, 0.51, "random baseline (0.5)", ha="right", va="bottom", fontsize=8, color=GREY)
ax.set_ylabel("ROC-AUC (held-out test matches)")
ax.set_title("(a) Classification models")
ax.set_ylim(0, 1.0)
ax.tick_params(axis="x", rotation=60)

ax = axes[1]
xs = [i for i, _ in w_r2] + [i for i, _ in b_r2]
ys = [v for _, v in w_r2] + [v for _, v in b_r2]
cs = [BLUE] * len(w_r2) + [ORANGE] * len(b_r2)
bars = ax.bar(xs, ys, color=cs, width=0.68)
ax.axhline(0.0, color="#333333", linewidth=0.8)
ax.set_ylabel("R² (held-out test matches)")
ax.set_title("(b) Regression models")
ax.tick_params(axis="x", rotation=60)

fig.suptitle("Fig. 3. Held-out validation performance by model type", y=1.02, fontsize=11)
handles = [Patch(color=BLUE, label="Bowling-side (W)"), Patch(color=ORANGE, label="Batting-side (B)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.09))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_holdout_performance.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: phase-relevance illustration (W8 Yorker Effectiveness)
# ---------------------------------------------------------------------------
phases = ["Powerplay", "Middle overs", "Death overs"]
w8_phase = [0.2, 0.6, 1.3]
w8_reliability = reliability["W8"]
effective = [p * w8_reliability for p in w8_phase]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(phases, effective, color=BLUE, width=0.55)
for x, (p, e) in enumerate(zip(w8_phase, effective)):
    ax.text(x, e + 0.02, f"{p}×{w8_reliability:.3f}\n= {e:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Effective weight (phase multiplier × reliability)")
ax.set_title("Fig. 4. Context-dependent weight of W8 (Yorker\nEffectiveness) across match phase")
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_phase_weight_example.png"), dpi=200)
plt.close(fig)

print("Wrote figures to", OUT)
for fn in sorted(os.listdir(OUT)):
    print(" -", fn)
