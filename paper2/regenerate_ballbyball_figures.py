"""
regenerate_ballbyball_figures.py — generates verified diagnostic figures for the
15 ball-by-ball-retrained bowling models, directly from their *_results.csv files
under data/output_data_ballbyball/bowling/. Same plotting logic and standards as
paper 1's regenerate_research_figures.py, pointed at the new, isolated data.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "data", "output_data_ballbyball", "fig_bxb", "bowling")
DATA_ROOT = os.path.join(ROOT, "data", "output_data_ballbyball", "bowling")

BLUE = "#3b6ea5"
ORANGE = "#d0793a"
GREEN = "#3a8f5a"
GREY = "#8a8a8a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

W_ORDER = [f"W{i}" for i in range(1, 16)]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def out_dir(mid):
    d = os.path.join(OUT_ROOT, mid)
    os.makedirs(d, exist_ok=True)
    return d


def find_cols(fieldnames):
    actual = predicted = None
    for c in fieldnames:
        lc = c.lower()
        if lc.endswith("_actual") or "_actual_" in lc:
            actual = c
        if "predicted_prob" in lc:
            predicted = c
        elif predicted is None and ("_predicted_" in lc or lc.endswith("_predicted")):
            predicted = c
    return actual, predicted


def plot_regression(mid, results_path, metrics):
    rows = read_csv(results_path)
    actual_col, pred_col = find_cols(rows[0].keys())
    y_true = np.array([float(r[actual_col]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])

    d = out_dir(mid)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(y_true, y_pred, s=6, alpha=0.15, color=BLUE, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#c0392b", linewidth=1.3, label="y = x")
    r2 = metrics.get("R2")
    mae = metrics.get("MAE")
    ax.set_title(f"{mid}: Predicted vs. Actual (ball-by-ball, held-out test matches)\n"
                 f"R² = {r2:.3f}, MAE = {mae:.3f}  (n = {len(rows):,})", fontsize=10.2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_predicted_vs_actual.png"), dpi=180)
    plt.close(fig)


def plot_classification(mid, results_path, metrics):
    rows = read_csv(results_path)
    actual_col, pred_col = find_cols(rows[0].keys())
    y_true = np.array([int(float(r[actual_col])) for r in rows])
    y_prob = np.array([float(r[pred_col]) for r in rows])

    d = out_dir(mid)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=GREEN, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{mid}: ROC Curve (ball-by-ball, n = {len(rows):,})")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    y_pred_cls = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_cls, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    acc = metrics.get("Accuracy")
    ax.set_title(f"{mid}: Confusion Matrix (thr=0.5)\nAccuracy = {acc:.3f}")

    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_classification_performance.png"), dpi=180)
    plt.close(fig)


def main():
    import json
    summary = json.load(open(os.path.join(ROOT, "data", "output_data_ballbyball",
                                           "bowling_ballbyball_training_summary.json")))
    for mid in W_ORDER:
        info = summary[mid]
        results_path = os.path.join(DATA_ROOT, mid, f"{mid}_results.csv")
        if mid in ("W5", "W10", "W12") or not os.path.exists(results_path):
            print(f"  SKIP {mid} (formula/lookup)")
            continue
        try:
            if info["type"] == "regression":
                plot_regression(mid, results_path, info["metrics"])
            elif info["type"] == "probability":
                plot_classification(mid, results_path, info["metrics"])
            print(f"  OK   {mid}")
        except Exception as e:
            print(f"  FAIL {mid}: {e}")

    # Reliability summary figure, from verified JSON - top 10 by reliability for
    # readability; the complete 15-model set (weak models included) is reported
    # in Table I, unchanged.
    reliability = json.load(open(os.path.join(ROOT, "decision_engine", "config", "reliability_ballbyball.json")))
    ids = W_ORDER
    vals = [reliability[i] for i in ids]
    order = sorted(range(len(ids)), key=lambda k: -vals[k])[:10]
    ids_s = [ids[k] for k in order]
    vals_s = [vals[k] for k in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ids_s, vals_s, color=BLUE, width=0.65)
    ax.set_ylabel("Reliability weight")
    ax.set_title("Fig. Top 10 most reliable bowling models (ball-by-ball suite)")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "..", "fig_bxb_reliability_weights.png"), dpi=200)
    plt.close(fig)
    print("Wrote reliability figure (top 10).")


if __name__ == "__main__":
    main()
