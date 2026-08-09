"""
Text Generator
--------------
Rewritten for Generative Template Engine (1000+ Scenarios).
Uses a compositional grammar to build advice dynamically based on multiple 
dimensions of the live Match State (Phase, Recent Wickets, Run Momentum, Ball in Over).
"""

import math
import random

ACTION_TARGETS = {
    "change_bowler": "Captain",
    "defensive_field": "Captain",
    "attacking_field": "Captain",
    "bring_spinner": "Captain",
    "maintain_current_bowler": "Captain",
    "use_variations": "Bowler",
    "bowl_yorkers_death": "Bowler",
    "target_matchup": "Bowler",
    "accelerate": "Both Batters",
    "rotate_strike": "Both Batters",
    "build_partnership": "Both Batters",
    "defend_wicket": "Both Batters",
    "target_boundaries": "Striker"
}

PHASE_INSTRUCTIONS = {
    "defensive_field": {
        "powerplay": "drop men to the boundary and employ sweeping sweepers",
        "middle": "protect the boundaries with 5 men out",
        "death": "set a deep square leg trap and push point back"
    },
    "attacking_field": {
        "powerplay": "deploy 2 slips and a gully",
        "middle": "bring mid-off up and keep catching men in the ring",
        "death": "bring long-on up and tempt the big shot"
    },
    "use_variations": {
        "powerplay": "use scrambled seam and cross-seam deliveries",
        "middle": "deploy off-cutters and pace-off deliveries into the pitch",
        "death": "bowl wide off-stump slower bouncers and knuckle balls"
    },
    "change_bowler": {
        "powerplay": "rotate the attack immediately",
        "middle": "bring on a fresh bowler",
        "death": "bring back your premier strike bowler"
    },
    "bring_spinner": {
        "powerplay": "introduce spin early to exploit the dry pitch",
        "middle": "toss it up outside off",
        "death": "dart the ball into the pads"
    },
    "maintain_current_bowler": {
        "powerplay": "let the current bowler finish their spell",
        "middle": "stick with this bowler",
        "death": "trust your death specialist"
    },
    "bowl_yorkers_death": {
        "powerplay": "search for the base of the stumps",
        "middle": "fire in the occasional surprise yorker",
        "death": "execute tailing yorkers at the toes"
    },
    "target_matchup": {
        "powerplay": "exploit the batter's early vulnerability",
        "middle": "target the batter's historical weak zones",
        "death": "keep the ball out of their hitting arc"
    },
    "accelerate": {
        "powerplay": "exploit the fielding restrictions and loft over the infield",
        "middle": "take calculated risks and turn 1s into 2s",
        "death": "clear the front leg and swing for the fences"
    },
    "rotate_strike": {
        "powerplay": "drop and run to upset the bowler's rhythm",
        "middle": "tap the ball into the gaps for easy singles",
        "death": "get the set batter on strike at all costs"
    },
    "build_partnership": {
        "powerplay": "rotate the strike to keep the scoreboard ticking and punish the bad balls",
        "middle": "find the gaps and look for a boundary an over to maintain pressure",
        "death": "maximize every delivery while feeding the set batter the strike"
    },
    "defend_wicket": {
        "powerplay": "survive the immediate threat without getting bogged down",
        "middle": "respect the good deliveries but immediately rotate strike",
        "death": "use the crease to disrupt their length instead of swinging blindly"
    },
    "target_boundaries": {
        "powerplay": "step out and clear the 30-yard circle",
        "middle": "target the shorter boundary with the sweep or reverse-sweep",
        "death": "target cow corner using the depth of the crease"
    }
}

ACTION_JUSTIFICATIONS = {
    "defensive_field": [
        "to cut off the easy doubles.",
        "to force them to take aerial risks.",
        "to squeeze the run rate.",
        "to create scoreboard pressure."
    ],
    "attacking_field": [
        "to create a catching opportunity.",
        "to immediately challenge the new batter.",
        "to exploit the lack of foot movement.",
        "to induce a false shot."
    ],
    "use_variations": [
        "to disrupt the batter's timing.",
        "to extract inconsistent bounce.",
        "to keep the batter guessing.",
        "to deceive them in the flight."
    ],
    "change_bowler": [
        "to break the settled rhythm.",
        "to find an early breakthrough.",
        "to change the angle of attack.",
        "to exploit a favorable matchup."
    ],
    "bring_spinner": [
        "to control the middle overs.",
        "to exploit a turning surface.",
        "to take the pace off the ball.",
        "to target the batter's weakness against spin."
    ],
    "maintain_current_bowler": [
        "to capitalize on their current momentum.",
        "to finish their spell effectively.",
        "to maintain the pressure.",
        "to execute the set plan."
    ],
    "bowl_yorkers_death": [
        "to leave absolutely no room for elevation.",
        "to trap them LBW.",
        "to prevent boundaries at the death.",
        "to target the base of the stumps."
    ],
    "target_matchup": [
        "to exploit their known weakness.",
        "to execute the pre-match plan.",
        "to restrict their scoring options.",
        "to maximize the chance of a wicket."
    ],
    "accelerate": [
        "to maximize the fielding restrictions.",
        "to shift momentum back to the batting side.",
        "to capitalize on the loose deliveries.",
        "to put the bowler under extreme pressure."
    ],
    "rotate_strike": [
        "to avoid facing consecutive dot balls.",
        "to keep the scoreboard ticking.",
        "to frustrate the fielding captain.",
        "to ensure the set batter takes charge."
    ],
    "build_partnership": [
        "to lay a platform without sacrificing the run rate.",
        "to force the bowler to alter their lines constantly.",
        "to wear down the attack before launching.",
        "to maintain a strong foundation for the death overs."
    ],
    "defend_wicket": [
        "to prevent a collapse and build a launchpad.",
        "to respect the good bowling without going entirely defensive.",
        "to rebuild the innings quickly.",
        "to ensure you bat out the full 20 overs at a high tempo."
    ],
    "target_boundaries": [
        "to clear the infield.",
        "to exploit the short boundaries.",
        "to maximize the scoring rate.",
        "to break the bowler's confidence."
    ]
}


class TextGenerator:
    def __init__(self, action_labels: dict):
        self.action_labels = action_labels

    def _get_contextual_prefix(self, match_state: dict, live_state: dict) -> str:
        recent_wicket = match_state.get("recent_wicket", False)
        
        # Depending on role, we look at different state vars
        runs_in_over = live_state.get("runs_conceded_in_over", live_state.get("runs_scored_in_over", 0))
        ball_in_over = live_state.get("ball_in_over", 1)
        
        crr = live_state.get("current_run_rate", 0.0)
        
        if recent_wicket:
            return random.choice([
                "With the new batter at the crease,",
                "Following the crucial breakthrough,",
                "To capitalize on the fallen wicket,"
            ])
            
        if crr > 10.0:
            return random.choice([
                "Capitalizing on this explosive start,",
                "To maintain this dominant run rate,",
                "Keeping the foot on the gas,"
            ])
            
        if ball_in_over >= 5:
            return random.choice([
                "To close out the over strongly,",
                "For the final deliveries of the over,",
                "To finish the over on your terms,"
            ])
            
        if runs_in_over >= 10:
            return random.choice([
                "To stem the bleeding,",
                "Under high run pressure,",
                "To regain control of the over,"
            ])
            
        return random.choice([
            "In this phase,",
            "Given the current match situation,",
            "To dictate the tempo,"
        ])

    def _get_generative_sentence(self, action: str, phase: str, role_label: str, match_state: dict, live_state: dict) -> str:
        # 1. Prefix
        prefix = self._get_contextual_prefix(match_state, live_state)
        
        # 2. Core Instruction
        core_instruction = PHASE_INSTRUCTIONS.get(action, {}).get(phase, role_label.lower())
        
        # 3. Justification
        justifications = ACTION_JUSTIFICATIONS.get(action, ["to execute the tactical plan."])
        # Use a deterministic hash based on ball to keep it stable during a single ball state
        ball_hash = hash(str(live_state.get("over", 0)) + str(live_state.get("ball_in_over", 0)) + action)
        justification = justifications[ball_hash % len(justifications)]
        
        sentence = f"{prefix} {core_instruction} {justification}"
        # Capitalize first letter properly
        return sentence[0].upper() + sentence[1:]

    def generate(self, role: str, decision: dict, match_state: dict, live_state: dict = None) -> str:
        if live_state is None:
            live_state = {}
            
        chosen = decision["chosen"]
        if chosen is None:
            return "No strong recommendation - signals are mixed or all top options were tactically blocked."

        n_signals = decision.get("n_signals", 0)
        n_total = max(decision.get("n_models_total", 1), 1)
        coverage = n_signals / n_total

        valid_actions = [a for a in decision.get("audit", []) if not a.get("blocked_by") and a.get("score", 0) > 0]
        valid_actions.sort(key=lambda x: x["score"], reverse=True)
        
        phase = match_state.get("phase", "middle")

        if not valid_actions:
            action, score = chosen
            label = self.action_labels.get(role, {}).get(action, action)
            context_str = self._get_generative_sentence(action, phase, label, match_state, live_state)
            return f"Recommendation: {context_str}"

        # Group by target
        grouped_advice = {}
        for a in valid_actions[:4]:  
            target = ACTION_TARGETS.get(a["action"], "General")
            if target not in grouped_advice:
                grouped_advice[target] = []
            
            label = self.action_labels.get(role, {}).get(a["action"], a["action"])
            context_str = self._get_generative_sentence(a["action"], phase, label, match_state, live_state)
            grouped_advice[target].append(context_str)

        lines = []
        header = f"Generative Tactical Plan ({phase.upper()} PHASE):"
        if match_state.get("recent_wicket", False):
            header += " [RECENT WICKET - ADAPT STRATEGY]"
        lines.append(header)
        
        for target, advices in grouped_advice.items():
            combined_advice = " AND ".join(advices)
            lines.append(f"• To the {target}: {combined_advice}")

        blocked_note = ""
        if decision["blocked"]:
            shown = decision["blocked"][:3]
            blocked_desc = ", ".join(f"{a} (blocked: {rid})" for a, rid in shown)
            remaining = len(decision["blocked"]) - len(shown)
            if remaining > 0:
                blocked_desc += f", and {remaining} more"
            blocked_note = f"\nNote: {blocked_desc} scored higher but was tactically ruled out."

        coverage_note = ""
        if coverage < 0.6:
            coverage_note = f"\n(Confidence limited: {n_signals}/{n_total} models fired for this state)"

        full_text = "\n".join(lines) + blocked_note + coverage_note
        return full_text
