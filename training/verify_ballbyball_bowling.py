"""
verify_ballbyball_bowling.py — independently recomputes each ball-by-ball bowling
model's headline metric directly from its *_results.csv using sklearn, and compares
it against the reported metric in bowling_ballbyball_training_summary.json. Same
verification standard as paper 1's verify_all_figures.py.
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "data", "output_data_ballbyball", "bowling")
SUMMARY_PATH = os.path.join(ROOT, "data", "output_data_ballbyball", "bowling_ballbyball_training_summary.json")

W_ORDER = [f"W{i}" for i in range(1, 16)]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def check(mid, summary):
    results_path = os.path.join(OUT_ROOT, mid, f"{mid}_results.csv")
    if not os.path.exists(results_path):
        return f"{mid}: SKIP (no results.csv)"
    metrics = summary[mid]["metrics"]
    mtype = summary[mid]["type"]
    rows = read_csv(results_path)
    fieldnames = rows[0].keys()

    if mid in ("W5", "W10", "W12"):
        return f"{mid}: SKIP (formula/lookup, no held-out predictive metric to cross-check)"

    actual_col, pred_col = find_cols(fieldnames)
    if actual_col is None or pred_col is None:
        return f"{mid}: SKIP (could not identify columns: {list(fieldnames)})"

    y_true = np.array([float(r[actual_col]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])

    if mtype == "regression":
        r2 = r2_score(y_true, y_pred)
        rep = metrics["R2"]
        ok = abs(r2 - rep) < 0.005
        return f"{mid}: {'OK' if ok else 'MISMATCH'} (recomputed R2={r2:.4f} vs reported={rep:.4f}, n={len(rows)})"
    elif mtype == "probability":
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            return f"{mid}: SKIP (roc_auc_score error: {e})"
        rep = metrics.get("ROC_AUC")
        ok = abs(auc - rep) < 0.005
        return f"{mid}: {'OK' if ok else 'MISMATCH'} (recomputed ROC_AUC={auc:.4f} vs reported={rep:.4f}, n={len(rows)})"
    return f"{mid}: SKIP (unhandled type {mtype})"


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    for mid in W_ORDER:
        print(" ", check(mid, summary))


if __name__ == "__main__":
    main()
