#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and reward processing for Gorge Chase DIY agent.
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None, state_summary=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None, prompt=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.FEATURE_VECTOR_SHAPE[0],
    legal_action=Config.ACTION_DIM,
    act=1,
    reward=Config.VALUE_DIM,
    reward_sum=Config.VALUE_DIM,
    done=1,
    value=Config.VALUE_DIM,
    next_value=Config.VALUE_DIM,
    advantage=Config.VALUE_DIM,
    prob=Config.ACTION_DIM,
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, next_remain_info, obs, next_obs):
    prev_monster_dist = float((remain_info or {}).get("nearest_monster_dist", 1.0))
    next_monster_dist = float((next_remain_info or {}).get("nearest_monster_dist", 1.0))
    prev_treasure_dist = float((remain_info or {}).get("nearest_treasure_dist", 1.0))
    next_treasure_dist = float((next_remain_info or {}).get("nearest_treasure_dist", 1.0))
    prev_score = float((remain_info or {}).get("score", 0.0))
    next_score = float((next_remain_info or {}).get("score", prev_score))

    reward = 0.01
    reward += 0.08 * (next_monster_dist - prev_monster_dist)

    if next_treasure_dist < 1.0:
        reward += 0.05 * (prev_treasure_dist - next_treasure_dist)

    score_gain = max(0.0, next_score - prev_score)
    reward += min(2.0, score_gain / 100.0)

    if terminated:
        reward -= 2.0
    elif truncated:
        reward += 1.0

    return np.array([reward], dtype=np.float32)


def sample_process(list_game_data):
    if not list_game_data:
        return list_game_data

    running_return = 0.0
    for i in range(len(list_game_data) - 1):
        list_game_data[i].next_value = list_game_data[i + 1].value
    list_game_data[-1].next_value = np.zeros(1, dtype=np.float32)

    for sample in reversed(list_game_data):
        running_return = float(sample.reward[0]) + Config.GAMMA * running_return * (1.0 - float(sample.done[0]))
        sample.reward_sum = np.array([running_return], dtype=np.float32)
        sample.advantage = sample.reward_sum - sample.value

    return list_game_data
