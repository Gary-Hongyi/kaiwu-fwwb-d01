#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_ppo.conf.conf import Args, Config, DimConfig


def fedexing_make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = fedexing_make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.dim_public = Config.DIM_PUBLIC
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.value_head_names = Config.VALUE_HEAD_NAMES
        self.value_head_num = Config.VALUE_HEAD_NUM
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.legal_action_dim = int(np.sum(Config.LEGAL_ACTION_SIZE_LIST))
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_FRD[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_4[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1[0])
        self.single_river_crab_feature_dim = int(DimConfig.DIM_OF_RIVER_CRAB[0])
        self.single_bullet_feature_dim = int(DimConfig.DIM_OF_BULLET_1_9[0])

        self.all_hero_feature_dim = int(np.sum(DimConfig.DIM_OF_HERO_FRD) + np.sum(DimConfig.DIM_OF_HERO_EMY))
        self.all_soldier_feature_dim = int(
            np.sum(DimConfig.DIM_OF_SOLDIER_1_4) + np.sum(DimConfig.DIM_OF_SOLDIER_5_8)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1) + np.sum(DimConfig.DIM_OF_ORGAN_2))
        self.all_bullet_feature_dim = int(
            np.sum(DimConfig.DIM_OF_BULLET_1_9) + np.sum(DimConfig.DIM_OF_BULLET_10)
        )
        structured_dim = (
            self.all_hero_feature_dim
            + self.all_soldier_feature_dim
            + self.single_river_crab_feature_dim
            + self.all_organ_feature_dim
            + self.all_bullet_feature_dim
        )
        assert structured_dim == self.feature_dim, f"{structured_dim=} != {self.feature_dim=}"

        self.position_mlp = MLP([Args.DIM_DISTANCE, 128, 64], "position_mlp")
        self.position_delta_dim = 64 - Args.DIM_DISTANCE
        self.unit_no_pos_mlp = MLP([Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32], "unit_no_pos_mlp")
        self.unit_delta_dim = 64 + 32 - Args.DIM_UNIT

        # Enhanced network architecture with residual connections
        # 增强的网络架构，带有残差连接
        hero_dims = [self.single_hero_feature_dim + self.unit_delta_dim, 512, 384, 256, 128]
        self.hero_mlp = MLP(hero_dims[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = fedexing_make_fc_layer(hero_dims[-2], hero_dims[-1])
        self.hero_emy_fc = fedexing_make_fc_layer(hero_dims[-2], hero_dims[-1])

        soldier_dims = [self.single_soldier_feature_dim + self.unit_delta_dim, 128, 96, 64, 32]
        self.soldier_mlp = MLP(soldier_dims[:-1], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = fedexing_make_fc_layer(soldier_dims[-2], soldier_dims[-1])
        self.soldier_emy_fc = fedexing_make_fc_layer(soldier_dims[-2], soldier_dims[-1])

        self.river_crab_mlp = MLP(
            [self.single_river_crab_feature_dim + self.unit_delta_dim, 128, 64, 32],
            "river_crab_mlp",
        )

        organ_dims = [self.single_organ_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.organ_mlp = MLP(organ_dims[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = fedexing_make_fc_layer(organ_dims[-2], organ_dims[-1])
        self.organ_emy_fc = fedexing_make_fc_layer(organ_dims[-2], organ_dims[-1])

        bullet_dims = [self.single_bullet_feature_dim + self.position_delta_dim, 64, 64, 32]
        self.bullet_mlp = MLP(bullet_dims[:-1], "bullet_mlp", non_linearity_last=True)
        self.bullet_hero_fc = fedexing_make_fc_layer(bullet_dims[-2], bullet_dims[-1])
        self.bullet_organ_fc = fedexing_make_fc_layer(bullet_dims[-2], bullet_dims[-1])

        # Optimized concat layers with better capacity
        # 优化的拼接层，具有更好的容量
        concat_dim = hero_dims[-1] * 2 + soldier_dims[-1] * 2 + 32 + organ_dims[-1] * 2 + bullet_dims[-1] * 2
        self.concat_mlp = MLP([concat_dim, 768, self.lstm_unit_size], "concat_mlp", non_linearity_last=True)
        self.concat_other_mlp = MLP([concat_dim, 640, 512, self.lstm_unit_size], "concat_other_mlp")
        self.lstm_and_linear_mlp = MLP(
            [self.lstm_unit_size + self.lstm_unit_size, 384, self.dim_public],
            "lstm_and_linear_mlp",
            non_linearity_last=True,
        )

        # Layer normalization for better training stability
        # 层归一化以提高训练稳定性
        self.concat_layer_norm = nn.LayerNorm(self.lstm_unit_size)
        self.lstm_layer_norm = nn.LayerNorm(self.lstm_unit_size)

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        self.empty_target_embedding = nn.Parameter(torch.full((self.target_embed_dim,), 0.1))
        self.label_mlp = self._fedexing_make_label_heads()
        self.lstm_tar_embed_mlp = fedexing_make_fc_layer(self.dim_public, self.target_embed_dim)
        self.target_embed_mlp = fedexing_make_fc_layer(32, self.target_embed_dim, use_bias=False)
        self.value_mlp = MLP([self.dim_public, 64, self.value_head_num], "multi_value_mlp")

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        return_heads = usq_reward.float()
        advantage_heads = usq_advantage.float()
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        device = value_result.device
        reward_head_weights = torch.as_tensor(
            Config.REWARD_HEAD_WEIGHTS,
            dtype=advantage_heads.dtype,
            device=device,
        ).reshape(1, -1)
        advantage = torch.sum(advantage_heads * reward_head_weights, dim=1)

        value_loss_head_weights = torch.as_tensor(
            Config.VALUE_LOSS_HEAD_WEIGHTS,
            dtype=value_result.dtype,
            device=device,
        )
        value_loss_head_weights = value_loss_head_weights / torch.clamp(
            value_loss_head_weights.sum(),
            min=torch.tensor(1e-6, dtype=value_loss_head_weights.dtype, device=device),
        )
        value_loss_per_sample_head = 0.5 * torch.square(return_heads - value_result)
        self.value_cost_list = value_loss_per_sample_head.mean(dim=0)
        self.value_cost = torch.sum(self.value_cost_list * value_loss_head_weights)

        label_probability_list = []
        epsilon = 1e-5
        boundary = torch.tensor(1e20, device=device)
        self.policy_cost = torch.tensor(0.0, device=device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )
                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )
                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_probability = label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                ratio = torch.exp(policy_log_p - old_policy_log_p).clamp(0.0, 3.0)

                surr1 = ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                weighted_train = weight_list[task_index].float() * frame_is_train
                temp_policy_loss = -torch.sum(torch.minimum(surr1, surr2) * weighted_train) / torch.maximum(
                    torch.sum(weighted_train),
                    torch.tensor(1.0, device=device),
                )
                self.policy_cost = self.policy_cost + temp_policy_loss

        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                weighted_train = weight_list[task_index].float() * frame_is_train
                temp_entropy_loss = -torch.sum(temp_entropy_loss * weighted_train) / torch.maximum(
                    torch.sum(weighted_train),
                    torch.tensor(1.0, device=device),
                )
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=device))

        self.entropy_cost = torch.tensor(0.0, device=device)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        self.return_heads_mean = return_heads.mean(dim=0)
        self.advantage_heads_mean = advantage_heads.mean(dim=0)
        self.value_heads_mean = value_result.mean(dim=0)

        # Calculate KL divergence for monitoring policy update magnitude
        # 计算 KL 散度以监控策略更新幅度
        with torch.no_grad():
            kl_divergence = torch.tensor(0.0, device=device)
            for task_index in range(len(self.is_reinforce_task_list)):
                if self.is_reinforce_task_list[task_index]:
                    old_probs = old_label_probability_list[task_index]
                    new_probs = label_probability_list[task_index - len(self.is_reinforce_task_list)]
                    # KL(old || new) = sum(old * log(old / new))
                    kl = torch.sum(old_probs * torch.log(old_probs / (new_probs + epsilon) + epsilon), dim=1)
                    kl_divergence = kl_divergence + kl.mean()
            self.kl_divergence = kl_divergence / max(sum(self.is_reinforce_task_list), 1)

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
            self.value_cost_list,
            self.return_heads_mean,
            self.advantage_heads_mean,
            self.value_heads_mean,
            self.kl_divergence,
        ]

    def _fedexing_build_target_bias(self, target_raw_list, device, batch_size):
        bias_list = [torch.zeros(batch_size, device=device)]
        for target_idx, raw_feature in enumerate(target_raw_list[1:], start=1):
            if raw_feature is None:
                bias_list.append(torch.zeros(batch_size, device=device))
                continue

            present = (raw_feature.abs().sum(dim=1) > 0).float()
            hp_ratio = self._fedexing_unit_hp_ratio(raw_feature).clamp(0.0, 1.0)
            distance_ratio = self._fedexing_unit_distance_ratio(raw_feature).clamp(0.0, 1.0)

            if target_idx == 2:
                target_bias = 0.08 * (1.0 - hp_ratio)
            elif 3 <= target_idx <= 6:
                target_bias = 0.04 * (1.0 - hp_ratio) - 0.12 * distance_ratio
            elif target_idx == 7:
                target_bias = 0.06 * (1.0 - hp_ratio) - 0.08 * distance_ratio
            else:
                target_bias = 0.08 * (1.0 - hp_ratio) - 0.10 * distance_ratio

            target_bias = target_bias + (1.0 - present) * -2.0
            bias_list.append(target_bias)
        return torch.stack(bias_list, dim=1)

    def _fedexing_encode_structured_feature(self, feature_vec):
        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.single_river_crab_feature_dim,
                self.all_organ_feature_dim,
                self.all_bullet_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(
            [int(np.sum(DimConfig.DIM_OF_HERO_FRD)), int(np.sum(DimConfig.DIM_OF_HERO_EMY))],
            dim=1,
        )
        soldier_vec_list = feature_vec_split_list[1].split(
            [int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)), int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8))],
            dim=1,
        )
        organ_vec_list = feature_vec_split_list[3].split(
            [int(np.sum(DimConfig.DIM_OF_ORGAN_1)), int(np.sum(DimConfig.DIM_OF_ORGAN_2))],
            dim=1,
        )
        bullet_vec_list = feature_vec_split_list[4].split(
            [int(np.sum(DimConfig.DIM_OF_BULLET_1_9)), int(np.sum(DimConfig.DIM_OF_BULLET_10))],
            dim=1,
        )

        hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        soldier_frd = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_4, dim=1)
        soldier_emy = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_5_8, dim=1)
        organ_frd = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1, dim=1)
        organ_emy = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_2, dim=1)
        bullet_hero = bullet_vec_list[0].split(DimConfig.DIM_OF_BULLET_1_9, dim=1)
        bullet_organ = bullet_vec_list[1].split(DimConfig.DIM_OF_BULLET_10, dim=1)

        target_embed_list = []
        target_raw_list = []

        hero_emy_result_list = []
        for x in hero_emy:
            fc_out = self.hero_emy_fc(self.process_sub_feature(x, self.hero_mlp, True))
            _, target_part = fc_out.split([96, 32], dim=1)
            target_embed_list.append(target_part)
            target_raw_list.append(x)
            hero_emy_result_list.append(fc_out)
        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)

        hero_frd_result_list = []
        for x in hero_frd:
            fc_out = self.hero_frd_fc(self.process_sub_feature(x, self.hero_mlp, True))
            _, target_part = fc_out.split([96, 32], dim=1)
            target_embed_list.append(target_part)
            target_raw_list.append(x)
            hero_frd_result_list.append(fc_out)
        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)

        soldier_frd_result_list = []
        for x in soldier_frd:
            soldier_frd_result_list.append(self.soldier_frd_fc(self.process_sub_feature(x, self.soldier_mlp, True)))
        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32)
        soldier_frd_concat_result, _ = soldier_frd_concat_result.max(dim=1)

        soldier_emy_result_list = []
        for x in soldier_emy:
            fc_out = self.soldier_emy_fc(self.process_sub_feature(x, self.soldier_mlp, True))
            target_embed_list.append(fc_out)
            target_raw_list.append(x)
            soldier_emy_result_list.append(fc_out)
        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32)
        soldier_emy_concat_result, _ = soldier_emy_concat_result.max(dim=1)

        river_crab_result = self.process_sub_feature(feature_vec_split_list[2], self.river_crab_mlp, True)

        organ_frd_result_list = []
        for x in organ_frd:
            organ_frd_result_list.append(self.organ_frd_fc(self.process_sub_feature(x, self.organ_mlp, True)))
        organ_frd_concat_result = torch.cat(organ_frd_result_list, dim=1)

        organ_emy_result_list = []
        for x in organ_emy:
            fc_out = self.organ_emy_fc(self.process_sub_feature(x, self.organ_mlp, True))
            target_embed_list.append(fc_out)
            target_raw_list.append(x)
            organ_emy_result_list.append(fc_out)
        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)

        bullet_hero_result_list = []
        for x in bullet_hero:
            bullet_hero_result_list.append(self.bullet_hero_fc(self.process_sub_feature(x, self.bullet_mlp, False)))
        bullet_hero_concat_result = torch.cat(bullet_hero_result_list, dim=1).reshape(-1, Args.BULLET_MAX_NUM - 1, 32)
        bullet_hero_concat_result, _ = bullet_hero_concat_result.max(dim=1)

        bullet_organ_result_list = []
        for x in bullet_organ:
            bullet_organ_result_list.append(self.bullet_organ_fc(self.process_sub_feature(x, self.bullet_mlp, False)))
        bullet_organ_concat_result = torch.cat(bullet_organ_result_list, dim=1)

        empty_target = self.empty_target_embedding.reshape(1, -1).expand(feature_vec.shape[0], -1)
        target_embed_list.insert(0, empty_target)
        target_raw_list.insert(0, None)
        target_embed_list.append(river_crab_result)
        target_raw_list.append(feature_vec_split_list[2])
        target_embedding = torch.stack(target_embed_list, dim=1)
        target_bias = self._fedexing_build_target_bias(target_raw_list, feature_vec.device, feature_vec.shape[0])

        return (
            hero_frd_concat_result,
            hero_emy_concat_result,
            soldier_frd_concat_result,
            soldier_emy_concat_result,
            river_crab_result,
            organ_frd_concat_result,
            organ_emy_concat_result,
            bullet_hero_concat_result,
            bullet_organ_concat_result,
            target_embedding,
            target_bias,
        )

    def _fedexing_make_label_heads(self):
        return ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.dim_public, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )

    def _fedexing_output_heads(self, feature_vec, public_result, target_embedding, target_bias):
        return self._fedexing_run_single_head(
            public_result,
            target_embedding,
            target_bias,
            self.label_mlp,
            self.lstm_tar_embed_mlp,
            self.target_embed_mlp,
            self.value_mlp,
        )

    def _fedexing_run_single_head(
        self,
        public_result,
        target_embedding,
        target_bias,
        label_mlp,
        lstm_tar_embed_mlp,
        target_embed_mlp,
        value_mlp,
    ):
        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(label_mlp[f"hero_label{label_index}_mlp"](public_result))
        result_list.append(
            self._fedexing_target_logits(public_result, target_embedding, target_bias, lstm_tar_embed_mlp, target_embed_mlp)
        )
        result_list.append(value_mlp(public_result))
        return result_list

    def _fedexing_target_logits(self, public_result, target_embedding, target_bias, lstm_tar_embed_mlp, target_embed_mlp):
        public_target = lstm_tar_embed_mlp(public_result).unsqueeze(1)
        target_query = target_embed_mlp(target_embedding)
        return torch.sum(target_query * public_target, dim=-1) / math.sqrt(self.target_embed_dim) + target_bias

    def _fedexing_unit_distance_ratio(self, feature):
        return feature[:, -Args.DIM_DISTANCE + Args.DIM_RPOS - 1]

    def _fedexing_unit_hp_ratio(self, feature):
        return feature[:, -Args.DIM_UNIT]

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        feature_vec = feature_vec.reshape(-1, self.feature_dim)

        (
            hero_frd_concat_result,
            hero_emy_concat_result,
            soldier_frd_concat_result,
            soldier_emy_concat_result,
            river_crab_result,
            organ_frd_concat_result,
            organ_emy_concat_result,
            bullet_hero_concat_result,
            bullet_organ_concat_result,
            target_embedding,
            target_bias,
        ) = self._fedexing_encode_structured_feature(feature_vec)

        concat_result = torch.cat(
            [
                hero_frd_concat_result,
                hero_emy_concat_result,
                soldier_frd_concat_result,
                soldier_emy_concat_result,
                river_crab_result,
                organ_frd_concat_result,
                organ_emy_concat_result,
                bullet_hero_concat_result,
                bullet_organ_concat_result,
            ],
            dim=1,
        )

        time_steps = 1 if inference else self.lstm_time_steps
        if feature_vec.shape[0] % time_steps != 0:
            time_steps = 1

        # Enhanced forward pass with layer normalization
        # 增强的前向传播，带有层归一化
        fc_public_result = self.concat_mlp(concat_result)
        fc_public_result = self.concat_layer_norm(fc_public_result)  # Apply layer norm

        reshape_fc_public_result = fc_public_result.reshape(-1, time_steps, self.lstm_unit_size)
        lstm_outputs, state = self.lstm(
            reshape_fc_public_result,
            (
                lstm_hidden_init.reshape(-1, self.lstm_unit_size).unsqueeze(0).contiguous(),
                lstm_cell_init.reshape(-1, self.lstm_unit_size).unsqueeze(0).contiguous(),
            ),
        )
        self.lstm_hidden_output = state[0]
        self.lstm_cell_output = state[1]
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)
        lstm_outputs = self.lstm_layer_norm(lstm_outputs)  # Apply layer norm

        public_mlp_result = self.concat_other_mlp(concat_result)
        public_result = self.lstm_and_linear_mlp(torch.cat([lstm_outputs, public_mlp_result], dim=-1))

        result_list = self._fedexing_output_heads(feature_vec, public_result, target_embedding, target_bias)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        return result_list

    def process_sub_feature(self, x, mlp, is_unit: bool):
        ret = [x]
        dim_suffix = Args.DIM_DISTANCE
        if is_unit:
            unit_no_pos = self.unit_no_pos_mlp(x[..., -Args.DIM_UNIT : -Args.DIM_DISTANCE])
            ret.append(unit_no_pos)
            dim_suffix = Args.DIM_UNIT
        pos = self.position_mlp(x[..., -Args.DIM_DISTANCE :])
        ret.append(pos)
        ret[0] = x[..., :-dim_suffix]
        return mlp(torch.cat(ret, dim=-1))

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()
