#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO.
"""


class Config:
    BASE_FEATURES = [
        4,
        5,
        5,
        16,
        16,
        2,
        8,
        1,
        8,
        1,
        1,
        3,
        8,
        4,
        8,
        6,
    ]
    SPATIAL_MAP_SIZE = 21
    SPATIAL_MAP_RADIUS = SPATIAL_MAP_SIZE // 2
    SPATIAL_MAP_CHANNELS = 4
    SPATIAL_FEATURE_LEN = SPATIAL_MAP_SIZE * SPATIAL_MAP_SIZE * SPATIAL_MAP_CHANNELS
    FEATURES = BASE_FEATURES + [SPATIAL_FEATURE_LEN]
    FEATURE_SPLIT_SHAPE = FEATURES
    NUMERICAL_FEATURE_LEN = sum(BASE_FEATURES)
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 16
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.00025
    LEARNING_RATE_END = 0.00005
    LR_DECAY_STEPS = 300000
    BETA_START = 0.012
    BETA_END = 0.0025
    ENTROPY_DECAY_STEPS = 350000
    POLICY_CLIP_PARAM_START = 0.24
    POLICY_CLIP_PARAM_END = 0.18
    CLIP_PARAM = 0.2
    VF_COEF = 0.6
    GRAD_CLIP_RANGE = 0.5
    ADVANTAGE_CLIP = 4.0

    HIDDEN_DIM = 256
    MID_DIM = 128
    LOW_DIM = 64
