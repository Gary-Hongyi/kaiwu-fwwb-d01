#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Minimal neural network model for Gorge Chase DIY agent.
"""


from torch import nn
from agent_diy.conf.conf import Config


class Model(nn.Module):
    def __init__(self, state_shape=None, action_shape=0, softmax=False):
        super().__init__()
        input_dim = int((state_shape or Config.FEATURE_VECTOR_SHAPE)[0])
        hidden_dim = Config.HIDDEN_DIM

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, Config.ACTION_DIM)
        self.value_head = nn.Linear(hidden_dim, Config.VALUE_DIM)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
