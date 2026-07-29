"""
Text Generator
--------------
Purely templated - no LLM call needed for this to work end to end. If you
want more natural phrasing later, feed this same structured object to an
LLM as a *rewriting* step (never let it see raw model numbers directly, and
never let it choose the action).

Confidence is deliberately NOT "0.5 + |score|/6" anymore (that was an
arbitrary bounded-arithmetic guess with no connection to what "confidence"
should mean). It's now built from two things a coach would actually want
to know before trusting a recommendation:
  1. margin  - how much better the chosen action scored than the runner-up.
               A win by a landslide should read as more confident than a
               near-tie, regardless of the chosen action's raw score.
  2. coverage - what fraction of the model roster actually had a signal
               for this ball (some models only fire once specific context
               is known, e.g. a matchup model needs a known batter/bowler
               pairing). Recommending off 4 of 15 models firing should
               never look as confident as recommending off 13 of 15.
This is still a heuristic (there's no ground truth "was this the right
call" label to calibrate against), but it now varies with information the
engine actually has, instead of being a function of one number's magnitude.
"""

import math


class TextGenerator:
    def __init__(self, action_labels: dict):
        self.action_labels = action_labels

    def generate(self, role: str, decision: dict, top_n: int = 3) -> str:
        chosen = decision["chosen"]
        if chosen is None:
            return "No strong recommendation - signals are mixed or all top options were blocked by rule checks."

        action, score = chosen
        label = self.action_labels.get(role, {}).get(action, action)

        non_chosen_scores = [a["score"] for a in decision.get("audit", []) if a["action"] != action]
        runner_up = max(non_chosen_scores) if non_chosen_scores else 0.0
        margin = score - runner_up

        n_signals = decision.get("n_signals", 0)
        n_total = max(decision.get("n_models_total", 1), 1)
        coverage = n_signals / n_total

        margin_term = math.tanh(margin / 1.5)  # scale chosen so a ~1.5-point margin is a clear win
        confidence = 0.5 + 0.45 * margin_term * (0.4 + 0.6 * coverage)
        confidence = min(0.97, max(0.35, confidence))

        contributors = decision["contributions"].get(action, [])[:top_n]
        reason_bits = []
        for model_id, model_label, contribution in contributors:
            direction = "supports" if contribution > 0 else "weighs against"
            reason_bits.append(f"{model_label} ({model_id}) {direction} this")

        reasons = "; ".join(reason_bits) if reason_bits else "aggregate signal across models"

        blocked_note = ""
        if decision["blocked"]:
            blocked_desc = ", ".join(f"{a} (blocked: {rid})" for a, rid in decision["blocked"])
            blocked_note = f" Note: {blocked_desc} scored higher but was ruled out."

        coverage_note = ""
        if coverage < 0.6:
            coverage_note = f" (based on {n_signals}/{n_total} models - several had no signal for this state)"

        return (
            f"Recommendation: {label}. Confidence: {confidence * 100:.0f}%{coverage_note}. "
            f"Key signals: {reasons}.{blocked_note}"
        )
