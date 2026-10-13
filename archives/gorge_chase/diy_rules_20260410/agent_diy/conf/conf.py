#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase DIY agent.
"""


class Config:
    FEATURE_VECTOR_SHAPE = (80,)
    ACTION_SHAPE = (16,)
    VALUE_SHAPE = (1,)

    ACTION_DIM = ACTION_SHAPE[0]
    VALUE_DIM = VALUE_SHAPE[0]

    GAMMA = 0.99
    START_LR = 1e-4

    HIDDEN_DIM = 64
    MAP_SIZE = 128.0
    MAX_FLASH_CD = 2000.0
    MAX_BUFF_DURATION = 50.0
    MAX_MONSTER_SPEED = 5.0
    MAX_STEP = 1000.0
