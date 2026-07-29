"""
build_reliability.py
---------------------
Turns each model's real validation metric into a reliability multiplier
for the Weighter, and writes decision_engine/config/reliability.json.

Run this after (re)training any model, so the engine's confidence in a
model always reflects how well that model actually validated - not a flat
1.0 for everything regardless of whether it beats a coin flip.

Mapping (deliberately simple, documented, and adjustable):
  classifiers (have ROC_AUC) : reliability = clip(0.4 + 1.2*(AUC-0.5), 0.2, 1.6)
      AUC 0.50 (random)  -> 0.20 (floor)
      AUC 0.75 (decent)  -> 0.70
      AUC 1.00 (perfect) -> 1.00 (clipped from 1.0, i.e. capped, not boosted further)
  regressors (have R2)      : reliability = clip(0.2 + max(R2, 0), 0.15, 1.6)
      R2 <= 0 (no better than the mean) -> 0.15-0.20 floor
      R2 = 0.68                          -> 0.88
  W12 (lookup table)         : reliability = clip(|correlation| * 2, 0.15, 1.0)
  W5 / W10 (fixed formulas)  : 0.7 flat - these were never fit to data, so
                               there's no accuracy metric to base a score on;
                               0.7 is a documented, neutral "trust it about as
                               much as a middling model" default.
  B2 (Cox survival)          : 0.6 flat - same reasoning as W5/W10; hazard
                               ratios don't have an R2/AUC in the same sense.
  B3 (3-class accuracy)      : reliability scaled against the 0.333 random
                               baseline instead of the binary 0.5 baseline.
"""

import json
import os

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "output_data")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "decision_engine", "config")


def compute_reliability():
    bowling = json.load(open(os.path.join(OUT_ROOT, "bowling_training_summary.json")))
    batting = json.load(open(os.path.join(OUT_ROOT, "batting_training_summary.json")))
    all_models = {**bowling, **batting}

    reliability = {}
    for mid, info in all_models.items():
        m = info["metrics"]
        if "ROC_AUC" in m and m["ROC_AUC"] is not None:
            auc = m["ROC_AUC"]
            rel = max(0.2, min(1.6, 0.4 + 1.2 * (auc - 0.5)))
        elif "R2" in m:
            r2 = m["R2"]
            rel = max(0.15, min(1.6, 0.2 + max(r2, 0)))
        elif mid == "W12":
            corr = m.get("correlation_career_economy_vs_future_economy", 0) or 0
            rel = max(0.15, min(1.0, abs(corr) * 2))
        elif mid in ("W5", "W10"):
            rel = 0.7
        elif mid == "B2":
            rel = 0.6
        elif mid == "B3":
            acc = m.get("Accuracy", 0.333)
            rel = max(0.15, min(1.0, (acc - 0.333) / 0.667 * 1.2 + 0.3))
        else:
            rel = 1.0
        reliability[mid] = round(rel, 3)
    return reliability


def main():
    reliability = compute_reliability()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, "reliability.json"), "w") as f:
        json.dump(reliability, f, indent=2)
    for mid in sorted(reliability, key=lambda x: (x[0], int(x[1:]))):
        print(f"{mid:5s} reliability={reliability[mid]:.3f}")
    print(f"\nWrote {len(reliability)} entries to config/reliability.json")


if __name__ == "__main__":
    main()
