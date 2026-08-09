"""
Normalizer
----------
Every one of the models outputs a single raw scalar in its last column,
but those scalars live on totally different scales:

  - regression models   -> unbounded continuous values (economy, runs, score...)
  - probability models  -> bounded [0, 1]
  - raw_signed models   -> a score that is already directional but on its own scale
                           (e.g. W10's field-optimization score, which is negative
                           for defensive setups and positive for attacking ones)

The Normalizer's only job is to turn each of these into a common "signal"
on a fixed scale of -1 (strongly against) to +1 (strongly for), so the
Aggregator can combine them without one model's units dominating another's.
"""

import math


class Normalizer:
    def __init__(self, model_stats: dict):
        """
        model_stats: { model_id: {"mean": float, "std": float} }
        Computed once from each model's historical output distribution
        (see scripts/compute_model_stats.py).
        """
        self.model_stats = model_stats

    def normalize(self, model_id: str, raw_value: float, model_type: str) -> float:
        if model_type == "probability":
            # 0.5 is "neutral" -> map [0,1] to [-1,1]
            signal = (raw_value - 0.5) * 2
        elif model_type == "raw_signed":
            # Already directional (e.g. optimization score); just squash with tanh
            # scaled by the model's own std so different models' raw scales don't
            # produce wildly different signal magnitudes.
            std = self.model_stats.get(model_id, {}).get("std", 1.0) or 1.0
            signal = math.tanh(raw_value / std)
        else:  # "regression" -> z-score against historical distribution, then clip
            stats = self.model_stats.get(model_id, {"mean": 0.0, "std": 1.0})
            std = stats["std"] or 1.0
            z = (raw_value - stats["mean"]) / std
            signal = max(-1.0, min(1.0, z / 2.0))  # clip at +-2 std -> +-1 signal

        return max(-1.0, min(1.0, signal))
