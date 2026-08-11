"""
Weighter
--------
Not every model matters equally at every moment. A death-over yorker model
(W8) should barely move the needle during the powerplay, and a powerplay
containment model (W15) shouldn't matter much in the death overs.

PHASE_RELEVANCE is deliberately simple and editable: a small lookup table of
(model_id -> phase -> multiplier). Models not listed default to 1.0 in every
phase (they're assumed phase-agnostic, e.g. matchup bias).

RELIABILITY is no longer a flat 1.0 for every model - it's loaded from
config/reliability.json, which is derived from each model's REAL validation
metric (ROC_AUC for classifiers, R2 for regressors, a correlation for the
W12 lookup table, a documented neutral default for the two hand-tuned
formula "models" W5/W10). A model with weak signal (e.g. W14's rebuilt
economy-trend regressor, R2 = -0.12) is now actually down-weighted instead
of being trusted exactly as much as a strong one (e.g. W3, R2 = 0.68).
Regenerate this file by re-running training/build_reliability.py after
retraining.
"""

import json
import os

_CONFIG_DIR = os.path.dirname(__file__).replace("core", "config")
# Defaults to reliability.json (paper 1), unchanged unless explicitly overridden -
# set TOP_RELIABILITY_FILE=reliability_ballbyball.json to load the reliability
# weights derived from the ball-by-ball-retrained bowling suite (paper 2) instead.
_RELIABILITY_FILE = os.environ.get("TOP_RELIABILITY_FILE", "reliability.json")
_RELIABILITY_PATH = os.path.join(_CONFIG_DIR, _RELIABILITY_FILE)

PHASE_RELEVANCE = {
    # bowling models
    "W8":  {"powerplay": 0.2, "middle": 0.6, "death": 1.3},   # yorker effectiveness
    "W9":  {"powerplay": 0.0, "middle": 0.0, "death": 1.3},   # death-over accuracy: trained ONLY on over=15 snapshots
    "W15": {"powerplay": 1.3, "middle": 0.0, "death": 0.0},   # powerplay containment: trained ONLY on over=5 snapshots
    "W7":  {"powerplay": 0.6, "middle": 1.2, "death": 0.7},   # spin control (more of a middle-overs weapon)

    # batting models
    "B8":  {"powerplay": 1.3, "middle": 0.5, "death": 0.3},   # powerplay exploitation
    "B15": {"powerplay": 0.2, "middle": 0.6, "death": 1.3},   # death-over optimization
    "B14": {"powerplay": 0.2, "middle": 0.6, "death": 1.3},   # acceleration capability: trained on death overs only
    "B9":  {"powerplay": 0.6, "middle": 1.2, "death": 0.8},   # spin vulnerability
}

DEFAULT_RELIABILITY = 1.0


def _load_reliability() -> dict:
    if os.path.exists(_RELIABILITY_PATH):
        with open(_RELIABILITY_PATH) as f:
            return json.load(f)
    return {}


RELIABILITY = _load_reliability()


class Weighter:
    def weight(self, model_id: str, phase: str) -> float:
        phase_mult = PHASE_RELEVANCE.get(model_id, {}).get(phase, 1.0)
        reliability = RELIABILITY.get(model_id, DEFAULT_RELIABILITY)
        return phase_mult * reliability
