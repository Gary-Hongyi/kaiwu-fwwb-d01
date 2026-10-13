#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    Enhanced monitoring panel with Taichu-2025 standards
    增强的监控面板，符合 Taichu-2025 标准
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1 - Taichu-2025")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .add_metric(
            metrics_name="reward_health",
            expr="round(avg(reward_health{}), 0.01)",
        )
        .add_metric(
            metrics_name="reward_tower",
            expr="round(avg(reward_tower{}), 0.01)",
        )
        .add_metric(
            metrics_name="reward_economy",
            expr="round(avg(reward_economy{}), 0.01)",
        )
        .add_metric(
            metrics_name="reward_state",
            expr="round(avg(reward_state{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="Legal Action 统计",
            name_en="legal_action_stats",
            type="line",
        )
        .add_metric(
            metrics_name="legal_action_ratio",
            expr="round(avg(legal_action_ratio{}), 0.001)",
        )
        .add_metric(
            metrics_name="illegal_action_count",
            expr="round(sum(illegal_action_count{}), 1)",
        )
        .add_metric(
            metrics_name="action_entropy",
            expr="round(avg(action_entropy{}), 0.01)",
        )
        .add_metric(
            metrics_name="legal_button_count",
            expr="round(avg(legal_button_count{}), 0.01)",
        )
        .add_metric(
            metrics_name="legal_target_count",
            expr="round(avg(legal_target_count{}), 0.01)",
        )
        .add_metric(
            metrics_name="normal_attack_legal_ratio",
            expr="round(avg(normal_attack_legal_ratio{}), 0.001)",
        )
        .add_metric(
            metrics_name="compact_tower_target_legal_ratio",
            expr="round(avg(compact_tower_target_legal_ratio{}), 0.001)",
        )
        .add_metric(
            metrics_name="raw_legal_action_valid_ratio",
            expr="round(avg(raw_legal_action_valid_ratio{}), 0.001)",
        )
        .add_metric(
            metrics_name="raw_legal_action_density",
            expr="round(avg(raw_legal_action_density{}), 0.001)",
        )
        .add_metric(
            metrics_name="raw_legal_button_count",
            expr="round(avg(raw_legal_button_count{}), 0.01)",
        )
        .add_metric(
            metrics_name="raw_legal_target_count",
            expr="round(avg(raw_legal_target_count{}), 0.01)",
        )
        .add_metric(
            metrics_name="raw_normal_attack_legal_ratio",
            expr="round(avg(raw_normal_attack_legal_ratio{}), 0.001)",
        )
        .add_metric(
            metrics_name="raw_tower_target_legal_ratio",
            expr="round(avg(raw_tower_target_legal_ratio{}), 0.001)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_farm",
            expr="round(avg(value_loss_farm{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_kda",
            expr="round(avg(value_loss_kda{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_damage",
            expr="round(avg(value_loss_damage{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_push",
            expr="round(avg(value_loss_push{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_loss_state",
            expr="round(avg(value_loss_state{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值预测",
            name_en="value_pred",
            type="line",
        )
        .add_metric(
            metrics_name="value_pred_farm",
            expr="round(avg(value_pred_farm{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_pred_kda",
            expr="round(avg(value_pred_kda{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_pred_damage",
            expr="round(avg(value_pred_damage{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_pred_push",
            expr="round(avg(value_pred_push{}), 0.01)",
        )
        .add_metric(
            metrics_name="value_pred_state",
            expr="round(avg(value_pred_state{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="回报目标",
            name_en="return_heads",
            type="line",
        )
        .add_metric(
            metrics_name="return_farm",
            expr="round(avg(return_farm{}), 0.01)",
        )
        .add_metric(
            metrics_name="return_kda",
            expr="round(avg(return_kda{}), 0.01)",
        )
        .add_metric(
            metrics_name="return_damage",
            expr="round(avg(return_damage{}), 0.01)",
        )
        .add_metric(
            metrics_name="return_push",
            expr="round(avg(return_push{}), 0.01)",
        )
        .add_metric(
            metrics_name="return_state",
            expr="round(avg(return_state{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="优势函数",
            name_en="advantage_heads",
            type="line",
        )
        .add_metric(
            metrics_name="advantage_farm",
            expr="round(avg(advantage_farm{}), 0.01)",
        )
        .add_metric(
            metrics_name="advantage_kda",
            expr="round(avg(advantage_kda{}), 0.01)",
        )
        .add_metric(
            metrics_name="advantage_damage",
            expr="round(avg(advantage_damage{}), 0.01)",
        )
        .add_metric(
            metrics_name="advantage_push",
            expr="round(avg(advantage_push{}), 0.01)",
        )
        .add_metric(
            metrics_name="advantage_state",
            expr="round(avg(advantage_state{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="网络性能指标",
            name_en="network_performance",
            type="line",
        )
        .add_metric(
            metrics_name="gradient_norm",
            expr="round(avg(gradient_norm{}), 0.001)",
        )
        .add_metric(
            metrics_name="learning_rate",
            expr="round(avg(learning_rate{}), 0.000001)",
        )
        .add_metric(
            metrics_name="kl_divergence",
            expr="round(avg(kl_divergence{}), 0.001)",
        )
        .end_panel()
        .add_panel(
            name="Taichu-2025 特征统计",
            name_en="taichu_feature_stats",
            type="line",
        )
        .add_metric(
            metrics_name="hero_feature_norm",
            expr="round(avg(hero_feature_norm{}), 0.01)",
        )
        .add_metric(
            metrics_name="soldier_feature_norm",
            expr="round(avg(soldier_feature_norm{}), 0.01)",
        )
        .add_metric(
            metrics_name="target_attention_entropy",
            expr="round(avg(target_attention_entropy{}), 0.01)",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
