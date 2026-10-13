#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Neural network model for Gorge Chase PPO.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_spatial"
        self.device = device

        input_dim = Config.NUMERICAL_FEATURE_LEN
        hidden_dim = Config.HIDDEN_DIM
        mid_dim = Config.MID_DIM
        low_dim = Config.LOW_DIM
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        spatial_channels = Config.SPATIAL_MAP_CHANNELS
        spatial_size = Config.SPATIAL_MAP_SIZE

        self.numerical_backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, low_dim),
            nn.ReLU(),
        )

        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            make_fc_layer(32 * 3 * 3, low_dim),
            nn.ReLU(),
        )

        self.fusion_backbone = nn.Sequential(
            make_fc_layer(low_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, low_dim),
            nn.ReLU(),
        )

        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size

        self.actor_head = make_fc_layer(low_dim, action_num)
        self.critic_head = make_fc_layer(low_dim, value_num)

    def forward(self, obs, inference=False):
        numerical_obs = obs[:, : Config.NUMERICAL_FEATURE_LEN]
        spatial_obs = obs[:, Config.NUMERICAL_FEATURE_LEN :]
        spatial_obs = spatial_obs.view(
            -1, self.spatial_channels, self.spatial_size, self.spatial_size
        )

        numerical_hidden = self.numerical_backbone(numerical_obs)
        spatial_hidden = self.spatial_backbone(spatial_obs)
        hidden = self.fusion_backbone(torch.cat([numerical_hidden, spatial_hidden], dim=1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
