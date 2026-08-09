"""
build_reliability_ballbyball.py
---------------------------------
Computes reliability weights for the ball-by-ball-retrained bowling suite
(W1-W15) using the exact same documented formula as training/build_reliability.py,
and merges them with the EXISTING, unchanged batting reliability values from
decision_engine/config/reliability.json (batting is not retrained between the
two papers). Writes to a new, isolated file -
decision_engine/config/reliability_ballbyball.json - never touching the
original reliability.json paper 1's live system depends on.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOWLING_SUMMARY = os.path.join(ROOT, "data", "output_data_ballbyball", "bowling_ballbyball_training_summary.json")
ORIGINAL_RELIABILITY = os.path.join(ROOT, "decision_engine", "config", "reliability.json")
OUT_PATH = os.path.join(ROOT, "decision_engine", "config", "reliability_ballbyball.json")


def compute_bowling_reliability():
    bowling = json.load(open(BOWLING_SUMMARY))
    reliability = {}
    for mid, info in bowling.items():
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
        else:
            rel = 1.0
        reliability[mid] = round(rel, 3)
    return reliability


def main():
    original = json.load(open(ORIGINAL_RELIABILITY))
    batting_only = {k: v for k, v in original.items() if k.startswith("B")}
    bowling_ballbyball = compute_bowling_reliability()

    merged = {**bowling_ballbyball, **batting_only}
    with open(OUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print("Ball-by-ball bowling reliability (new):")
    for mid in sorted(bowling_ballbyball, key=lambda x: int(x[1:])):
        old = original.get(mid, "n/a")
        print(f"  {mid:5s} old={old}  new={bowling_ballbyball[mid]:.3f}")
    print(f"\nBatting reliability copied unchanged from paper 1 ({len(batting_only)} models).")
    print(f"Wrote {len(merged)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
