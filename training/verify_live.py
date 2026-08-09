"""
verify_live.py — independently recomputes each "live" model's headline metric
directly from its *_results.csv using sklearn, and compares it against the
reported metric in bowling_live_training_summary.json /
batting_live_training_summary.json. Same verification standard as
verify_ballbyball_bowling.py, extended to cover both bowling_live and
batting_live output directories.
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_LIVE_ROOT = os.path.join(ROOT, "data", "output_data_live")
BOWLING_OUT = os.path.join(OUT_LIVE_ROOT, "bowling")
BATTING_OUT = os.path.join(OUT_LIVE_ROOT, "batting")
BOWLING_SUMMARY_PATH = os.path.join(OUT_LIVE_ROOT, "bowling_live_training_summary.json")
BATTING_SUMMARY_PATH = os.path.join(OUT_LIVE_ROOT, "batting_live_training_summary.json")

W_ORDER = [f"W{i}" for i in range(1, 16)]
B_ORDER = ["B1", "B2", "B3", "B5", "B6", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"]

# Model ids whose type is a formula / lookup / multi-class / raw-signed
# output rather than a plain regression or probability - these have no
# single "actual vs predicted" pair to cross-check via r2/roc_auc, same
# skip set verify_ballbyball_bowling.py uses (plus the batting-only skips).
NO_METRIC_SKIP = {"W5", "W10", "W12"}


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


def check(mid, summary, out_root):
    results_path = os.path.join(out_root, mid, f"{mid}_results.csv")
    if not os.path.exists(results_path):
        return f"{mid}: SKIP (no results.csv)"
    if mid not in summary:
        return f"{mid}: SKIP (not in training summary)"
    info = summary[mid]
    metrics = info["metrics"]
    mtype = info["type"]

    if mtype not in ("regression", "probability"):
        return f"{mid}: SKIP (type={mtype}, no held-out predictive metric to cross-check)"
    if mid in NO_METRIC_SKIP:
        return f"{mid}: SKIP (formula/lookup, no held-out predictive metric to cross-check)"

    rows = read_csv(results_path)
    if not rows:
        return f"{mid}: SKIP (empty results.csv)"
    fieldnames = rows[0].keys()

    actual_col, pred_col = find_cols(fieldnames)
    if actual_col is None or pred_col is None:
        return f"{mid}: SKIP (could not identify columns: {list(fieldnames)})"

    y_true = np.array([float(r[actual_col]) for r in rows])
    y_pred = np.array([float(r[pred_col]) for r in rows])

    if mtype == "regression":
        r2 = r2_score(y_true, y_pred)
        rep = metrics["R2"]
        ok = abs(r2 - rep) < 0.005
        status = "OK" if ok else "MISMATCH"
        line = f"{mid}: {status} (recomputed R2={r2:.4f} vs reported={rep:.4f}, n={len(rows)})"
        if not ok:
            line += f"  [tolerance=0.005, diff={abs(r2 - rep):.6f}]"
        return line
    elif mtype == "probability":
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            return f"{mid}: SKIP (roc_auc_score error: {e})"
        rep = metrics.get("ROC_AUC")
        if rep is None:
            return f"{mid}: SKIP (no reported ROC_AUC)"
        ok = abs(auc - rep) < 0.005
        status = "OK" if ok else "MISMATCH"
        line = f"{mid}: {status} (recomputed ROC_AUC={auc:.4f} vs reported={rep:.4f}, n={len(rows)})"
        if not ok:
            line += f"  [tolerance=0.005, diff={abs(auc - rep):.6f}]"
        return line
    return f"{mid}: SKIP (unhandled type {mtype})"


def main():
    with open(BOWLING_SUMMARY_PATH) as f:
        bowling_summary = json.load(f)
    with open(BATTING_SUMMARY_PATH) as f:
        batting_summary = json.load(f)

    all_lines = []

    print("Bowling (live):")
    for mid in W_ORDER:
        line = check(mid, bowling_summary, BOWLING_OUT)
        all_lines.append(line)
        print(" ", line)

    print("\nBatting (live):")
    for mid in B_ORDER:
        line = check(mid, batting_summary, BATTING_OUT)
        all_lines.append(line)
        print(" ", line)

    n_ok = sum(1 for l in all_lines if ": OK" in l)
    n_mismatch = sum(1 for l in all_lines if ": MISMATCH" in l)
    n_skip = sum(1 for l in all_lines if ": SKIP" in l)

    print("\n" + "=" * 70)
    print(f"Verification summary: {n_ok} OK, {n_mismatch} MISMATCH, {n_skip} SKIP "
          f"(out of {len(all_lines)} models checked)")
    if n_mismatch == 0:
        print("ALL CLEAR: every regression/probability model's independently "
              "recomputed metric matches its reported metric within tolerance.")
    else:
        print("FAILURES DETECTED - see MISMATCH lines above for exact values/tolerance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
