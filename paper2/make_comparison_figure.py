"""
make_comparison_figure.py — builds the core empirical comparison chart:
Model Suite A (checkpoint-restricted baseline) vs Model Suite B (proposed
ball-by-ball training), per bowling model, computed directly from the two
verified training-summary JSON files.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLUE = "#3b6ea5"
ORANGE = "#d0793a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

old = json.load(open(os.path.join(ROOT, "data", "output_data", "bowling_training_summary.json")))
new = json.load(open(os.path.join(ROOT, "data", "output_data_ballbyball", "bowling_ballbyball_training_summary.json")))

W_ORDER = [f"W{i}" for i in range(1, 16)]

auc_ids, auc_old, auc_new = [], [], []
r2_ids, r2_old, r2_new = [], [], []
for mid in W_ORDER:
    om, nm = old[mid]["metrics"], new[mid]["metrics"]
    if "ROC_AUC" in om and om["ROC_AUC"] is not None:
        auc_ids.append(mid); auc_old.append(om["ROC_AUC"]); auc_new.append(nm["ROC_AUC"])
    elif "R2" in om:
        r2_ids.append(mid); r2_old.append(om["R2"]); r2_new.append(nm["R2"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
x = range(len(auc_ids))
w = 0.35
ax.bar([i - w/2 for i in x], auc_old, width=w, color=BLUE, label="Baseline (checkpoint-restricted)")
ax.bar([i + w/2 for i in x], auc_new, width=w, color=ORANGE, label="Proposed (ball-by-ball)")
ax.set_xticks(list(x)); ax.set_xticklabels(auc_ids, rotation=45)
ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
ax.set_ylabel("ROC-AUC (held-out test matches)")
ax.set_title("(a) Classification models")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
x = range(len(r2_ids))
ax.bar([i - w/2 for i in x], r2_old, width=w, color=BLUE, label="Baseline (checkpoint-restricted)")
ax.bar([i + w/2 for i in x], r2_new, width=w, color=ORANGE, label="Proposed (ball-by-ball)")
ax.set_xticks(list(x)); ax.set_xticklabels(r2_ids, rotation=45)
ax.axhline(0.0, color="#333", linewidth=0.8)
ax.set_ylabel("R² (held-out test matches)")
ax.set_title("(b) Regression models")
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle("Fig. C. Held-out performance: checkpoint-restricted baseline vs. proposed ball-by-ball training", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_c_suite_comparison.png"), dpi=200, bbox_inches="tight")
print("Wrote fig_c_suite_comparison.png")
