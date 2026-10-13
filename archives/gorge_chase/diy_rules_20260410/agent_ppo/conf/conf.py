#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO.
"""


class Config:
    # hero(6) + monster1(5) + monster2(5) + treasure(5) + buff(5) + local_map(16) + legal_action(16)
    # + directional_clearance(8) + trap(4) + stuck(4) + safe_treasure(4) + progress(2)
    FEATURES = [
        6,
        5,
        5,
        5,
        5,
        16,
        16,
        8,
        4,
        4,
        4,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # 8 move actions + 8 flash actions
    ACTION_NUM = 16

    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
