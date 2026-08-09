"""
verify_all_figures.py — independently recomputes each model's headline metric
directly from its *_results.csv (the same file regenerate_research_figures.py reads)
using sklearn, and compares it against the officially reported metric in
*_model_metrics.csv / *_training_summary.json. This checks that the two are not
just "both present" but numerically consistent, for every model that has a
results.csv, before trusting the regenerated figures.
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W_ORDER = [f"W{i}" for i in range(1, 16)]
B_ORDER = ["B1", "B2", "B3", "B5", "B6", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_cols(fieldnames):
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


def check(role, mid, summary):
    base = os.path.join(ROOT, "data", "output_data", role, mid)
    results_path = os.path.join(base, f"{mid}_results.csv")
    if not os.path.exists(results_path):
        return f"{mid}: SKIP (no results.csv)"

    metrics = summary[mid]["metrics"]
    mtype = summary[mid]["type"]
    rows = read_csv(results_path)
    fieldnames = rows[0].keys()

    if mid in ("W5", "W10", "W12"):
        return f"{mid}: SKIP (formula/lookup, no held-out predictive metric to cross-check)"
    if mid == "B2":
        risk = np.array([float(r["b2_predicted_risk"]) for r in rows])
        actual = np.array([int(float(r["actual_wicket"])) for r in rows])
        flagged = float((risk > 1.0).mean())
        dismissed = float(actual.mean())
        rep_flag = metrics["pct_flagged_high_risk"]
        rep_dis = metrics["pct_actually_dismissed"]
        ok = abs(flagged - rep_flag) < 0.01 and abs(dismissed - rep_dis) < 0.01
        return (f"{mid}: {'OK' if ok else 'MISMATCH'} "
                f"(recomputed flagged={flagged:.4f} vs reported={rep_flag:.4f}; "
                f"recomputed dismissed={dismissed:.4f} vs reported={rep_dis:.4f})")
    if mid == "B3":
        y_true = np.array([int(float(r["b3_actual_class"])) for r in rows])
        y_pred = np.array([int(float(r["b3_predicted_class"])) for r in rows])
        acc = accuracy_score(y_true, y_pred)
        rep = metrics["Accuracy"]
        ok = abs(acc - rep) < 0.005
        return f"{mid}: {'OK' if ok else 'MISMATCH'} (recomputed Acc={acc:.4f} vs reported={rep:.4f})"

    actual_col, pred_col = find_cols(fieldnames)
    if actual_col is None or pred_col is None:
        return f"{mid}: SKIP (could not identify actual/predicted columns: {list(fieldnames)})"

    y_true = np.array([float(r[actual_col]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])

    if mtype == "regression":
        r2 = r2_score(y_true, y_pred)
        rep = metrics["R2"]
        ok = abs(r2 - rep) < 0.005
        return f"{mid}: {'OK' if ok else 'MISMATCH'} (recomputed R2={r2:.4f} vs reported={rep:.4f})"
    elif mtype == "probability":
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            return f"{mid}: SKIP (roc_auc_score error: {e})"
        rep = metrics.get("ROC_AUC")
        if rep is None:
            return f"{mid}: SKIP (no reported ROC_AUC to compare)"
        ok = abs(auc - rep) < 0.005
        return f"{mid}: {'OK' if ok else 'MISMATCH'} (recomputed ROC_AUC={auc:.4f} vs reported={rep:.4f})"
    else:
        return f"{mid}: SKIP (unhandled type {mtype})"


def main():
    with open(os.path.join(ROOT, "data", "output_data", "bowling_training_summary.json")) as f:
        bowling_summary = json.load(f)
    with open(os.path.join(ROOT, "data", "output_data", "batting_training_summary.json")) as f:
        batting_summary = json.load(f)

    print("=== Bowling ===")
    for mid in W_ORDER:
        print(" ", check("bowling", mid, bowling_summary))
    print("=== Batting ===")
    for mid in B_ORDER:
        print(" ", check("batting", mid, batting_summary))


if __name__ == "__main__":
    main()
