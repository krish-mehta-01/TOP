"""
regenerate_research_figures.py — replaces every PNG under data/output_data/research_figures/
with figures generated directly from the CURRENT results/metrics/feature-importance CSVs
(the same ones training/train_bowling.py and train_batting.py produce today).

Why this exists: several of the pre-existing research_figures files were found to display
numeric values (e.g. an AUC baked into a plotted ROC curve) that do not match the current,
post-audit metrics reported in PROJECT_REPORT.md and data/output_data/*_training_summary.json
-- most likely left over from before the leakage/split-methodology fixes described there.
Rather than trying to guess which of the 79 original files were stale and which were not
(and rather than reverse-engineer bespoke chart types built on columns that no longer exist,
e.g. the old W5 dot-ball/boundary breakdown when the current W5 results.csv only has a single
optimization_score column), every model's figure set is replaced with a small, standardized,
directly-computed pair: a primary performance chart, and a feature-importance chart where a
*_feature_importance.csv exists. All numbers are computed live from the CSVs, so they cannot
drift out of sync with what the paper (or PROJECT_REPORT.md) reports.

B4 (never implemented) and B7 (a static, non-predictive lookup table with no results/metrics
file) have no current data to plot from; their stale figure directories are removed rather
than replaced with fabricated content. (B7 gets a separate, real, descriptive figure built
directly from the raw dataset by build_b7_matchup.py.)

Usage: python regenerate_research_figures.py
"""

import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "data", "output_data", "research_figures")

BLUE = "#3b6ea5"
ORANGE = "#d0793a"
GREEN = "#3a8f5a"
GREY = "#8a8a8a"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BOWLING_IDS = [f"W{i}" for i in range(1, 16)]
BATTING_IDS = ["B1", "B2", "B3", "B5", "B6", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"]
REMOVE_STALE = ["B4", "B7"]  # no current model/results to plot; B7 handled separately


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def out_dir(role, mid):
    d = os.path.join(OUT_ROOT, role, mid)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return d


def find_cols(fieldnames, mid):
    actual = predicted = None
    for c in fieldnames:
        lc = c.lower()
        if lc.endswith("_actual") or "_actual_" in lc or lc == "actual_wicket":
            actual = c
        if "predicted_prob" in lc:
            predicted = c
        elif predicted is None and ("_predicted_" in lc or lc.endswith("_predicted")):
            predicted = c
    return actual, predicted


def plot_regression(mid, role, results_path, feat_path, metrics):
    rows = read_csv(results_path)
    fieldnames = rows[0].keys()
    actual_col, pred_col = find_cols(fieldnames, mid)
    y_true = np.array([float(r[actual_col]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])

    d = out_dir(role, mid)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(y_true, y_pred, s=8, alpha=0.25, color=BLUE, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#c0392b", linewidth=1.3, label="y = x")
    r2 = metrics.get("R2")
    mae = metrics.get("MAE")
    title = f"{mid}: Predicted vs. Actual (held-out test matches)"
    sub = f"R² = {r2:.3f}" + (f",  MAE = {mae:.3f}" if mae is not None else "")
    ax.set_title(title + "\n" + sub, fontsize=10.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_predicted_vs_actual.png"), dpi=180)
    plt.close(fig)

    if feat_path and os.path.exists(feat_path):
        plot_feature_importance(mid, role, feat_path, d)


def plot_classification(mid, role, results_path, feat_path, metrics):
    rows = read_csv(results_path)
    fieldnames = rows[0].keys()
    actual_col, pred_col = find_cols(fieldnames, mid)
    y_true = np.array([int(float(r[actual_col])) for r in rows])
    y_prob = np.array([float(r[pred_col]) for r in rows])

    d = out_dir(role, mid)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    if len(set(y_true.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=GREEN, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{mid}: ROC Curve")
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Degenerate test set\n(single class)", ha="center", va="center")

    ax = axes[1]
    y_pred_cls = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_cls, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    acc = metrics.get("Accuracy")
    ax.set_title(f"{mid}: Confusion Matrix (thr=0.5)\nAccuracy = {acc:.3f}" if acc is not None else f"{mid}: Confusion Matrix")

    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_classification_performance.png"), dpi=180)
    plt.close(fig)

    if feat_path and os.path.exists(feat_path):
        plot_feature_importance(mid, role, feat_path, d)


def plot_multiclass(mid, role, results_path, metrics):
    rows = read_csv(results_path)
    actual_col = f"{mid.lower()}_actual_class"
    pred_col = f"{mid.lower()}_predicted_class"
    y_true = np.array([int(float(r[actual_col])) for r in rows])
    y_pred = np.array([int(float(r[pred_col])) for r in rows])
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    d = out_dir(role, mid)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([f"Pred {c}" for c in classes])
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([f"True {c}" for c in classes])
    acc = metrics.get("Accuracy")
    ax.set_title(f"{mid}: Confusion Matrix (3-class)\nAccuracy = {acc:.3f}" if acc is not None else f"{mid}: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_confusion_matrix.png"), dpi=180)
    plt.close(fig)


def plot_survival(mid, role, results_path, metrics):
    rows = read_csv(results_path)
    risk = np.array([float(r["b2_predicted_risk"]) for r in rows])
    actual = np.array([int(float(r["actual_wicket"])) for r in rows])

    d = out_dir(role, mid)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bins = np.linspace(risk.min(), risk.max(), 40)
    ax.hist(risk[actual == 0], bins=bins, alpha=0.55, color=BLUE, label="No dismissal", density=True)
    ax.hist(risk[actual == 1], bins=bins, alpha=0.55, color=ORANGE, label="Dismissal followed", density=True)
    ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="High-risk threshold (hazard = 1.0)")
    pct_flag = metrics.get("pct_flagged_high_risk")
    pct_dis = metrics.get("pct_actually_dismissed")
    ax.set_title(f"{mid}: Cox Hazard Score Distribution by Outcome\n"
                 f"Flagged high-risk = {pct_flag*100:.1f}%,  Actually dismissed = {pct_dis*100:.1f}%")
    ax.set_xlabel("Predicted hazard (Cox proportional-hazards score)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_hazard_distribution.png"), dpi=180)
    plt.close(fig)


def plot_formula(mid, role, results_path, metrics):
    rows = read_csv(results_path)
    scores = np.array([float(r["optimization_score"]) for r in rows])

    d = out_dir(role, mid)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.hist(scores, bins=40, color=BLUE, alpha=0.8)
    mean, std = metrics.get("mean"), metrics.get("std")
    ax.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.2, label=f"mean = {mean:.3f}")
    ax.set_title(f"{mid}: Hand-Tuned Score Distribution (n = {int(metrics.get('n', len(scores)))})\n"
                 f"mean = {mean:.3f}, std = {std:.3f}")
    ax.set_xlabel("Optimization score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_score_distribution.png"), dpi=180)
    plt.close(fig)


def plot_lookup(mid, role, results_path, metrics):
    rows = read_csv(results_path)
    vals = np.array([float(r["career_economy"]) for r in rows])

    d = out_dir(role, mid)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.hist(vals, bins=30, color=BLUE, alpha=0.8)
    corr = metrics.get("correlation_career_economy_vs_future_economy")
    n_bowlers = metrics.get("n_bowlers_in_lookup")
    ax.set_title(f"{mid}: Train-Only Career-Economy Lookup Distribution\n"
                 f"n bowlers = {n_bowlers}, corr(lookup, future economy) = {corr:.3f}")
    ax.set_xlabel("Career economy (training matches only)")
    ax.set_ylabel("Count of test-set snapshots")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_lookup_distribution.png"), dpi=180)
    plt.close(fig)


def plot_feature_importance(mid, role, feat_path, d):
    rows = read_csv(feat_path)
    rows = sorted(rows, key=lambda r: -float(r["Importance"]))[:12]
    feats = [r["Feature"] for r in rows][::-1]
    vals = [float(r["Importance"]) for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.32 * len(feats) + 1)))
    ax.barh(feats, vals, color=ORANGE)
    ax.set_xlabel("Importance")
    ax.set_title(f"{mid}: Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{mid}_feature_importance.png"), dpi=180)
    plt.close(fig)


def process(role, mid, summary):
    base = os.path.join(ROOT, "data", "output_data", role, mid)
    results_path = os.path.join(base, f"{mid}_results.csv")
    feat_path = os.path.join(base, f"{mid}_feature_importance.csv")
    if not os.path.exists(results_path):
        print(f"  SKIP {mid}: no results.csv")
        return
    metrics = summary[mid]["metrics"]
    mtype = summary[mid]["type"]

    try:
        if mid in ("W5", "W10"):
            plot_formula(mid, role, results_path, metrics)
        elif mid == "W12":
            plot_lookup(mid, role, results_path, metrics)
        elif mid == "B2":
            plot_survival(mid, role, results_path, metrics)
        elif mid == "B3":
            plot_multiclass(mid, role, results_path, metrics)
        elif mtype == "regression":
            plot_regression(mid, role, results_path, feat_path, metrics)
        elif mtype == "probability":
            plot_classification(mid, role, results_path, feat_path, metrics)
        else:
            print(f"  SKIP {mid}: unhandled type {mtype}")
            return
        print(f"  OK   {mid}")
    except Exception as e:
        print(f"  FAIL {mid}: {e}")


def main():
    import json
    with open(os.path.join(ROOT, "data", "output_data", "bowling_training_summary.json")) as f:
        bowling_summary = json.load(f)
    with open(os.path.join(ROOT, "data", "output_data", "batting_training_summary.json")) as f:
        batting_summary = json.load(f)

    print("Bowling models:")
    for mid in BOWLING_IDS:
        process("bowling", mid, bowling_summary)

    print("Batting models:")
    for mid in BATTING_IDS:
        process("batting", mid, batting_summary)

    print("Removing stale, non-current figure sets (B4 fully; B7 rebuilt separately):")
    for mid in REMOVE_STALE:
        d = os.path.join(OUT_ROOT, "batting", mid)
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  removed {d}")


if __name__ == "__main__":
    main()
