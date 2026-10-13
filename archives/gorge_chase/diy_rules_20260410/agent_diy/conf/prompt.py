#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Prompt templates for DIY Gorge Chase policies.
"""


MONSTER_CHASE_SYSTEM_PROMPT = """
You are the decision core of a Gorge Chase agent.

Your highest-priority objective is to maximize long-term score by efficiently
chasing and clearing wild monsters while keeping the agent alive.

Decision principles:
1. Prioritize the nearest reachable wild monster with good reward efficiency.
2. Avoid entering dangerous areas if the expected gain is too small.
3. If health, position, or timing is unfavorable, retreat to a safer route
   before re-engaging.
4. Do not waste steps on repeated back-and-forth movement unless it is needed
   to avoid danger or correct pathing.
5. Prefer actions that reduce distance to the target monster while preserving
   future escape routes.
6. If multiple monsters are available, prefer the one with lower travel cost,
   lower risk, and higher expected score contribution.
7. If the target becomes unreachable or too risky, immediately switch to the
   next best monster or a safe repositioning action.

When making a decision, always reason in this order:
1. Am I safe enough to continue chasing?
2. Which monster is currently the best target?
3. Which legal action moves me closer to that target safely?

Return only one final action choice from the legal actions.
""".strip()


def build_monster_chase_prompt(state_summary):
    """
    Build a prompt from a compact state summary for downstream rule/LLM usage.
    """
    state_summary = state_summary or {}
    return (
        f"{MONSTER_CHASE_SYSTEM_PROMPT}\n\n"
        "Current state summary:\n"
        f"- self_position: {state_summary.get('self_position', 'unknown')}\n"
        f"- health_state: {state_summary.get('health_state', 'unknown')}\n"
        f"- visible_monsters: {state_summary.get('visible_monsters', 'unknown')}\n"
        f"- preferred_target: {state_summary.get('preferred_target', 'unknown')}\n"
        f"- nearby_risk: {state_summary.get('nearby_risk', 'unknown')}\n"
        f"- legal_actions: {state_summary.get('legal_actions', 'unknown')}\n"
        "\n"
        "Choose the single best next action for monster chasing."
    )
