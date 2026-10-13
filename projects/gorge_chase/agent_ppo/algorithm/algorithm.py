#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO algorithm implementation for Gorge Chase PPO.
"""

import math
import os
import time

import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.beta_start = Config.BETA_START
        self.beta_end = Config.BETA_END
        self.entropy_decay_steps = Config.ENTROPY_DECAY_STEPS
        self.vf_coef = Config.VF_COEF
        self.value_clip_param = 0.2
        self.policy_clip_start = Config.POLICY_CLIP_PARAM_START
        self.policy_clip_end = Config.POLICY_CLIP_PARAM_END
        self.policy_clip_decay_steps = Config.ENTROPY_DECAY_STEPS

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        raw_advantage = advantage.clone()
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        advantage = advantage.clamp(-Config.ADVANTAGE_CLIP, Config.ADVANTAGE_CLIP)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        entropy_ratio = min(1.0, self.train_step / max(1, self.entropy_decay_steps))
        self.var_beta = self.beta_end + (self.beta_start - self.beta_end) * (1.0 - entropy_ratio)

        lr_ratio = min(1.0, self.train_step / max(1, Config.LR_DECAY_STEPS))
        current_lr = Config.LEARNING_RATE_END + (
            (Config.INIT_LEARNING_RATE_START - Config.LEARNING_RATE_END)
            * (1 + math.cos(math.pi * lr_ratio))
            / 2
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        policy_clip_ratio = min(1.0, self.train_step / max(1, self.policy_clip_decay_steps))
        policy_clip_param = self.policy_clip_start + (
            self.policy_clip_end - self.policy_clip_start
        ) * policy_clip_ratio

        logits, value_pred = self.model(obs)
        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
            raw_advantage=raw_advantage,
            policy_clip_param=policy_clip_param,
        )

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
                "beta": round(self.var_beta, 6),
                "lr": round(current_lr, 8),
                "policy_clip": round(policy_clip_param, 4),
                "approx_kl": round(float(info_list[3]), 6),
                "clip_frac": round(float(info_list[4]), 4),
                "explained_variance": round(float(info_list[5]), 4),
                "advantage_mean": round(float(info_list[6]), 4),
                "advantage_std": round(float(info_list[7]), 4),
                "value_pred_mean": round(float(info_list[8]), 4),
                "ratio_mean": round(float(info_list[9]), 4),
                "reward_sum_mean": round(float(info_list[10]), 4),
                "grad_norm": round(float(grad_norm), 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} beta:{results['beta']} lr:{results['lr']} "
                f"policy_clip:{results['policy_clip']} "
                f"policy_loss:{results['policy_loss']} value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} approx_kl:{results['approx_kl']} "
                f"clip_frac:{results['clip_frac']} exp_var:{results['explained_variance']} "
                f"adv_mean:{results['advantage_mean']} grad_norm:{results['grad_norm']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
        raw_advantage,
        policy_clip_param,
    ):
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - policy_clip_param, 1 + policy_clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        value_clip = old_value + (value_pred - old_value).clamp(
            -self.value_clip_param, self.value_clip_param
        )
        value_loss = 0.5 * torch.maximum(
            torch.square(reward_sum - value_pred),
            torch.square(reward_sum - value_clip),
        ).mean()

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        ratio_safe = ratio.clamp(min=1e-9)
        approx_kl = ((ratio_safe - 1.0) - torch.log(ratio_safe)).mean()
        clip_frac = (torch.abs(ratio - 1.0) > policy_clip_param).float().mean()

        reward_var = torch.var(reward_sum, unbiased=False)
        value_error_var = torch.var(reward_sum - value_pred, unbiased=False)
        explained_variance = 1.0 - value_error_var / (reward_var + 1e-8)

        return total_loss, [
            value_loss,
            policy_loss,
            entropy_loss,
            approx_kl.detach().item(),
            clip_frac.detach().item(),
            explained_variance.detach().item(),
            raw_advantage.mean().detach().item(),
            raw_advantage.std(unbiased=False).detach().item(),
            value_pred.mean().detach().item(),
            ratio.mean().detach().item(),
            reward_sum.mean().detach().item(),
        ]

    def _masked_softmax(self, logits, legal_action):
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
