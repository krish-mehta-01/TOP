"""
Rule Validator
--------------
The Aggregator's weighted scores can pick actions that are technically
"highest scoring" but tactically silly or against the game's rules
(e.g. recommending a bowling change for someone who's only bowled 1 over,
or telling a batter who just walked in to attack immediately).

This layer sits between the Aggregator and the Text Generator. It never
invents a new recommendation; it only blocks or flags ones the scoring
produced, using simple, explainable conditions on the live match_state.

match_state fields used here (all things you already have in your ball-by-ball
data): overs_bowled_by_current_bowler, recent_wicket, balls_faced_by_batter,
phase ("powerplay"/"middle"/"death"), required_run_rate, projected_run_rate.
"""


class RuleValidator:
    def __init__(self, rules_config: dict):
        self.rules = rules_config

    def validate(self, role: str, ranked_actions: list, match_state: dict) -> dict:
        """
        ranked_actions: [(action, score), ...] sorted best first
        returns: {
            "chosen": action_or_None,
            "blocked": [ (action, rule_id), ... ],   # actions skipped BEFORE the chosen one
            "audit": [ {"action", "score", "blocked_by", "chosen"}, ... ]  # every ranked action
        }
        The "audit" list is purely for transparency (e.g. showing a coach why
        the runner-up wasn't picked) - it does not change which action wins;
        "chosen" still stops at the first non-blocked action exactly as before.
        """
        blocked = []
        role_rules = self.rules.get(role, [])

        def is_blocked(action):
            for rule in role_rules:
                if "blocks_keyword" in rule:
                    if rule["blocks_keyword"] not in action:
                        continue
                elif rule["blocks"] != action:
                    continue
                
                cond = rule["condition"]
                if cond == "bowler_is_spin" and match_state.get("bowler_is_spin"):
                    return rule["id"]
                if cond == "bowler_is_pace" and not match_state.get("bowler_is_spin", True):
                    return rule["id"]
                if cond == "recent_wicket" and match_state.get("recent_wicket"):
                    return rule["id"]
                if cond == "overs_bowled_below_minimum" and match_state.get("overs_bowled_by_current_bowler", 99) < 2:
                    return rule["id"]
                if cond == "balls_faced_below_minimum" and match_state.get("balls_faced_by_batter", 99) < 4:
                    return rule["id"]
                if cond == "death_overs_and_rate_gap_high" and match_state.get("phase") == "death" and (
                    match_state.get("required_run_rate", 0) - match_state.get("projected_run_rate", 0) > 3
                ):
                    return rule["id"]
            return None

        chosen = None
        audit = []
        for action, score in ranked_actions:
            rule_id = is_blocked(action)
            is_chosen = False
            if rule_id:
                blocked.append((action, rule_id))
            elif chosen is None:
                chosen = (action, score)
                is_chosen = True
            audit.append({"action": action, "score": score, "blocked_by": rule_id, "chosen": is_chosen})

        return {"chosen": chosen, "blocked": blocked, "audit": audit}
