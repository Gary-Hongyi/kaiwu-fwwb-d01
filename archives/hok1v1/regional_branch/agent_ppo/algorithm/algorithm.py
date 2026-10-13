#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0

    def _fedexing_legal_action_monitor_stats(self, legal_action):
        legal_mask = (legal_action > 0.5).float()
        legal_parts = torch.split(legal_mask, list(Config.LABEL_SIZE_LIST), dim=1)
        legal_counts = [part.sum(dim=1) for part in legal_parts]
        empty_head_count = sum((count <= 0).sum() for count in legal_counts)
        entropy_per_frame = torch.zeros_like(legal_counts[0])
        for count in legal_counts:
            entropy_per_frame = entropy_per_frame + torch.log(torch.clamp(count, min=1.0))

        button_mask = legal_parts[0]
        target_mask = legal_parts[-1]
        normal_attack_ratio = 0.0
        tower_target_ratio = 0.0
        if button_mask.shape[1] > 3:
            normal_attack_ratio = float((button_mask[:, 3] > 0).float().mean().item())
        if target_mask.shape[1] > 7:
            tower_target_ratio = float((target_mask[:, 7] > 0).float().mean().item())

        return {
            "legal_action_ratio": round(float(legal_mask.mean().item()), 4),
            "illegal_action_count": int(empty_head_count.item()),
            "action_entropy": round(float(entropy_per_frame.mean().item()), 4),
            "legal_button_count": round(float(legal_counts[0].mean().item()), 3),
            "legal_target_count": round(float(legal_counts[-1].mean().item()), 3),
            "normal_attack_legal_ratio": round(normal_attack_ratio, 4),
            "compact_tower_target_legal_ratio": round(tower_target_ratio, 4),
        }

    def _fedexing_global_step(self):
        # Resume-safe global step: read Adam's internal per-parameter step counter
        # (restored together with optimizer state on resume). Falls back to the
        # local train_step if optimizer state is not populated yet.
        # 全局步（resume 安全）：读 Adam 优化器内部 step 计数（续训随优化器状态恢复）；
        # 优化器状态还没建立时回退到本地 train_step。
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p)
                if state and "step" in state:
                    step = state["step"]
                    return float(step.item() if hasattr(step, "item") else step)
        return float(self.train_step)

    def _fedexing_update_var_beta(self):
        # Linearly anneal the entropy coefficient (beta) from BETA_START to
        # BETA_END over BETA_ANNEAL_STEPS, beginning at BETA_ANNEAL_START_STEP.
        # Keyed on the GLOBAL step so it survives process restarts / resumes.
        # 熵系数 beta 线性退火，按全局步计，重启/续训不重置。
        step = self._fedexing_global_step()
        start = Config.BETA_START
        end = Config.BETA_END
        anneal_start = Config.BETA_ANNEAL_START_STEP
        anneal_steps = max(1, Config.BETA_ANNEAL_STEPS)
        progress = min(1.0, max(0.0, (step - anneal_start) / anneal_steps))
        self.model.var_beta = start + (end - start) * progress
        return self.model.var_beta

    def learn(self, list_sample_data):
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        # Extract sample field from SampleData objects and stack into tensor
        # 从 SampleData 对象中提取 sample 字段并 stack 成 tensor
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        current_var_beta = self._fedexing_update_var_beta()

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 梯度剪裁
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            elif isinstance(info, torch.Tensor) and info.ndim > 0:
                _info = info.detach().cpu().tolist()
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            (
                _,
                (value_loss, policy_loss, entropy_loss),
                value_loss_heads,
                return_heads,
                advantage_heads,
                value_heads,
                kl_divergence,
            ) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            for idx, name in enumerate(Config.VALUE_HEAD_NAMES):
                results[f"value_loss_{name}"] = round(value_loss_heads[idx], 4)
                results[f"return_{name}"] = round(return_heads[idx], 4)
                results[f"advantage_{name}"] = round(advantage_heads[idx], 4)
                results[f"value_pred_{name}"] = round(value_heads[idx], 4)

            # Enhanced monitoring metrics for Taichu-2025
            # Taichu-2025 增强监控指标
            if Config.USE_GRAD_CLIP:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, float('inf'))
                results["gradient_norm"] = round(float(grad_norm), 4)

            results["learning_rate"] = round(self.optimizer.param_groups[0]['lr'], 8)
            results["var_beta"] = round(current_var_beta, 6)
            results["beta_step"] = round(self._fedexing_global_step())
            results["kl_divergence"] = round(kl_divergence, 4)
            results.update(self._fedexing_legal_action_monitor_stats(legal_action))

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
