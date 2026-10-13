#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Lightweight self-test for the DIY rule policy.
"""

from __future__ import annotations

from pprint import pprint

from agent_diy.rule_policy import RulePolicy


def base_summary():
    return {
        "score": 0.0,
        "progress": 0.2,
        "flash_ready": False,
        "hero": {"x": 64.0, "z": 64.0, "escape_bias_dx": 1.0, "escape_bias_dz": 0.0},
        "monsters": [
            {"visible": 1.0, "dx_norm": -0.08, "dz_norm": 0.0, "speed_norm": 0.2, "dist_norm": 0.06},
            {"visible": 0.0, "dx_norm": 0.0, "dz_norm": 0.0, "speed_norm": 0.0, "dist_norm": 1.0},
        ],
        "treasure": {"visible": 1.0, "dx_norm": 0.10, "dz_norm": 0.02, "speed_norm": 0.0, "dist_norm": 0.09},
        "buff": {"visible": 0.0, "dx_norm": 0.0, "dz_norm": 0.0, "speed_norm": 0.0, "dist_norm": 1.0},
        "clearance": [0.9, 0.8, 0.5, 0.2, 0.1, 0.4, 0.7, 0.9],
        "trap": [0.25, 0.0, 0.0, 0.2],
        "stuck": [0.3, 0.2, 0.0, 0.0],
        "safe_treasure": [1.0, 0.7, 0.9, 0.6],
        "legal_actions": list(range(16)),
    }


def run_case(name, mutate):
    policy = RulePolicy()
    summary = base_summary()
    mutate(summary)
    decision = policy.decide(summary, [1] * 16)
    print(f"\n[{name}] action={decision.action} greedy={decision.greedy_action}")
    pprint(decision.debug)
    print("top_prob:", [(idx, round(p, 3)) for idx, p in enumerate(decision.prob) if p > 0.02])


if __name__ == "__main__":
    run_case("escape_near_monster", lambda s: None)

    def flash_escape(summary):
        summary["flash_ready"] = True
        summary["monsters"][0]["dist_norm"] = 0.03
        summary["trap"] = [0.75, 1.0, 0.0, 0.8]

    run_case("flash_escape", flash_escape)

    def chase_treasure(summary):
        summary["monsters"][0]["dist_norm"] = 0.25
        summary["monsters"][0]["dx_norm"] = -0.20
        summary["treasure"]["dx_norm"] = 0.18
        summary["treasure"]["dz_norm"] = 0.0
        summary["safe_treasure"] = [1.0, 0.95, 1.0, 0.9]
        summary["clearance"] = [1.0, 0.8, 0.6, 0.5, 0.4, 0.5, 0.7, 0.9]

    run_case("chase_treasure", chase_treasure)

    def stuck_case(summary):
        summary["flash_ready"] = False
        summary["stuck"] = [0.0, 0.0, 0.8, 1.0]
        summary["hero"]["escape_bias_dx"] = 0.0
        summary["hero"]["escape_bias_dz"] = 1.0
        summary["clearance"] = [0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 1.0, 0.8]
        summary["monsters"][0]["dist_norm"] = 0.14

    run_case("stuck_escape", stuck_case)
