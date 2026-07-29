"""
DecisionEngine
--------------
The single unified entry point. "Unified" means one class, one call signature,
one output schema for both roles - not that bowling and batting numbers get
mixed into the same score. A Context Router picks which 15 models are even
relevant; everything downstream (Normalizer, Weighter, Aggregator,
RuleValidator, TextGenerator) is shared code.

Usage:
    engine = DecisionEngine(config_dir="config", stats_path="data/model_stats.json")
    result = engine.decide(
        role="bowling",                     # or "batting"
        raw_model_outputs={"W1": 10.24, "W2": 0.60, ...},   # the 15 raw scalars
        match_state={"phase": "middle", "overs_bowled_by_current_bowler": 3, ...},
    )
    print(result["text"])
"""

import json
import os

from core.normalizer import Normalizer
from core.weighter import Weighter
from core.aggregator import Aggregator
from core.rule_validator import RuleValidator
from core.text_generator import TextGenerator


class DecisionEngine:
    def __init__(self, config_dir: str, stats_path: str):
        with open(os.path.join(config_dir, "model_config.json")) as f:
            self.model_config = json.load(f)
        with open(os.path.join(config_dir, "actions.json")) as f:
            actions_cfg = json.load(f)
        with open(stats_path) as f:
            model_stats = json.load(f)

        self.action_labels = {k: v for k, v in actions_cfg.items() if k != "rules"}
        self.rules_config = actions_cfg["rules"]

        self.normalizer = Normalizer(model_stats)
        self.weighter = Weighter()
        self.aggregator = Aggregator(self.model_config)
        self.rule_validator = RuleValidator(self.rules_config)
        self.text_generator = TextGenerator(self.action_labels)

    def models_for_role(self, role: str):
        return {mid: cfg for mid, cfg in self.model_config.items() if cfg["role"] == role}

    def decide(self, role: str, raw_model_outputs: dict, match_state: dict) -> dict:
        role_models = self.models_for_role(role)
        phase = match_state.get("phase", "middle")

        # 1) Normalize every raw scalar that was actually supplied for this context.
        #    (In production not every model will have fired for a given ball -
        #    e.g. a matchup model only fires once a specific batter/bowler pair
        #    is known. Missing models are simply skipped, which lowers the
        #    total signal weight rather than crashing.)
        signals = {}
        for model_id, raw_value in raw_model_outputs.items():
            if model_id not in role_models:
                continue
            model_type = role_models[model_id]["type"]
            signals[model_id] = self.normalizer.normalize(model_id, raw_value, model_type)

        # 2) Aggregate into per-action scores with contribution trail
        agg = self.aggregator.aggregate(signals, phase, self.weighter)
        ranked_actions = sorted(agg["action_scores"].items(), key=lambda kv: -kv[1])

        # 3) Validate against hard rules (may block the top pick)
        decision = self.rule_validator.validate(role, ranked_actions, match_state)
        decision["contributions"] = agg["contributions"]
        decision["n_signals"] = len(signals)
        decision["n_models_total"] = len(role_models)

        # 4) Generate coach-readable text
        text = self.text_generator.generate(role, decision)

        return {
            "role": role,
            "phase": phase,
            "ranked_actions": ranked_actions,
            "chosen": decision["chosen"],
            "blocked": decision["blocked"],
            "audit": decision["audit"],
            "contributions": decision["contributions"],
            "signals_used": signals,
            "text": text,
        }
