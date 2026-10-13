#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(monitor, name, metric, expr=None):
    return (
        monitor.add_panel(
            name=name,
            name_en=metric,
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name=metric,
            expr=expr or f"round(avg({metric}{{}}), 0.0001)",
        )
        .end_panel()
    )


def build_monitor():
    monitor = MonitorConfigBuilder()

    monitor = (
        monitor.title("DIY 1v1 Training")
        .add_group(
            group_name="Training",
            group_name_en="training",
        )
    )
    for name, metric in [
        ("Episode Reward", "reward"),
        ("Episode Frame", "episode_frame"),
        ("Episode Count", "episode_cnt"),
        ("Is Eval", "is_eval"),
    ]:
        monitor = _add_line_panel(monitor, name, metric)
    monitor = monitor.end_group()

    monitor = monitor.add_group(
        group_name="Behavior",
        group_name_en="behavior",
    )
    for name, metric in [
        ("Move Rate", "move_rate"),
        ("No-op Rate", "noop_rate"),
        ("Attack Rate", "attack_rate"),
        ("Skill Rate", "skill_rate"),
        ("Other Action Rate", "other_action_rate"),
        ("Force Move Count", "force_move_mask_count"),
    ]:
        monitor = _add_line_panel(monitor, name, metric)
    monitor = monitor.end_group()

    monitor = monitor.add_group(
        group_name="Progress",
        group_name_en="progress",
    )
    for name, metric in [
        ("Progress", "progress"),
        ("Progress Delta", "progress_delta"),
        ("Max Progress", "max_progress"),
    ]:
        monitor = _add_line_panel(monitor, name, metric)
    monitor = monitor.end_group()

    monitor = monitor.add_group(
        group_name="Reward Items",
        group_name_en="reward_items",
    )
    for name, metric in [
        ("Forward Reward", "forward_weight"),
        ("HP Reward", "hp_point_weight"),
        ("Tower HP Reward", "tower_hp_point_weight"),
        ("Money Reward", "money_weight"),
        ("Exp Reward", "exp_weight"),
        ("EP Reward", "ep_rate_weight"),
        ("Kill Reward", "kill_weight"),
        ("Death Reward", "death_weight"),
        ("Last Hit Reward", "last_hit_weight"),
    ]:
        monitor = _add_line_panel(monitor, name, metric)
    monitor = monitor.end_group()

    monitor = monitor.add_group(
        group_name="Learner Loss",
        group_name_en="learner_loss",
    )
    for name, metric in [
        ("Total Loss", "total_loss"),
        ("Value Loss", "value_loss"),
        ("Policy Loss", "policy_loss"),
        ("Entropy Loss", "entropy_loss"),
    ]:
        monitor = _add_line_panel(monitor, name, metric, expr=f"round(avg({metric}{{}}), 0.01)")
    monitor = monitor.end_group()

    return monitor.build()
