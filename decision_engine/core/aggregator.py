"""
Aggregator
----------
Takes the normalized [-1, 1] signal from every model that fired for this
context, multiplies it by that model's action-direction (+1/-1) and its
phase weight, and sums the contributions per candidate action.

Also keeps a record of which models contributed how much to which action,
so the Text Generator can cite the actual reasons (not just a number).
"""

from collections import defaultdict


class Aggregator:
    def __init__(self, model_config: dict):
        self.model_config = model_config

    def aggregate(self, signals: dict, phase: str, weighter) -> dict:
        """
        signals: { model_id: normalized_signal_float }
        returns: {
          action_scores: { action: float },
          contributions: { action: [ (model_id, label, contribution), ... ] }
        }
        """
        action_scores = defaultdict(float)
        contributions = defaultdict(list)

        for model_id, signal in signals.items():
            cfg = self.model_config.get(model_id)
            if cfg is None:
                continue
            w = weighter.weight(model_id, phase)
            for action, direction in cfg.get("actions", {}).items():
                contribution = signal * direction * w
                action_scores[action] += contribution
                contributions[action].append((model_id, cfg["label"], contribution))

        # Sort each action's contributors by absolute contribution, biggest first
        for action in contributions:
            contributions[action].sort(key=lambda t: -abs(t[2]))

        return {
            "action_scores": dict(action_scores),
            "contributions": dict(contributions),
        }
