#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Patch: berserk_tower_chip v2.1 - Hero-specific feature computation
补丁：狂暴塔外磨塔 v2.1 - 英雄专用特征计算
"""

import math


# Hero config IDs
# 英雄配置ID
HERO_LUBAN = 112
HERO_DIRENJIE = 133

# Tower attack range
# 防御塔攻击范围
TOWER_ATTACK_RANGE = 8000


class HeroSpecificFeatures:
    """Compute hero-specific features for Luban (112) and Di Renjie (133).
    计算鲁班七号和狄仁杰的英雄专用特征。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.attack_count = 0
        self.skill_used_this_frame = False
        self.frames_since_skill = 99
        self.prev_frame_no = -1
        self.hero_config_id = -1

    def update(self, frame_state, hero_config_id, main_camp):
        """Update hero-specific state from frame data.
        从帧数据更新英雄专用状态。
        """
        frame_no = frame_state.get("frame_no", 0)
        if self.prev_frame_no >= 0 and frame_no > self.prev_frame_no:
            self.frames_since_skill += 1
        self.prev_frame_no = frame_no

        self.hero_config_id = hero_config_id

    def record_attack(self):
        self.attack_count += 1

    def record_skill_use(self):
        self.skill_used_this_frame = True
        self.frames_since_skill = 0

    def get_luban_features(self, frame_state, main_camp):
        """Compute Luban-specific features.
        计算鲁班七号专用特征。
        
        Returns dict with:
        - sweep_ready_est: estimated sweep ready (0-1)
        - sweep_triggerable_by_skill: can trigger sweep with next skill
        - safe_to_sweep: safe to use sweep
        - sweep_can_hit_hero: sweep can hit enemy hero
        - sweep_can_hit_minion_count: estimated sweep-hit minions
        - sweep_can_hit_tower: sweep can hit tower
        """
        features = {
            "sweep_ready_est": 0.0,
            "sweep_triggerable_by_skill": False,
            "safe_to_sweep": False,
            "sweep_can_hit_hero": False,
            "sweep_can_hit_minion_count": 0,
            "sweep_can_hit_tower": False,
        }

        self_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == main_camp and hero["hp"] > 0:
                self_hero = hero
            elif hero["camp"] != main_camp and hero["hp"] > 0:
                enemy_hero = hero

        if self_hero is None:
            return features

        # Estimate sweep ready: based on attack count (every 5th attack triggers sweep)
        # or if a skill was recently used (skill triggers sweep)
        # 估计扫射就绪：基于攻击计数（每5次普攻触发扫射）或技能后触发
        attack_mod = self.attack_count % 5
        if attack_mod == 4:
            features["sweep_ready_est"] = 0.9
        elif attack_mod == 3:
            features["sweep_ready_est"] = 0.6
        elif self.frames_since_skill < 3:
            features["sweep_ready_est"] = 0.8
        else:
            features["sweep_ready_est"] = 0.2

        features["sweep_triggerable_by_skill"] = (
            features["sweep_ready_est"] < 0.5 and self.frames_since_skill > 10
        )

        # Safety check: hero HP > 35% and not about to die
        # 安全检查：英雄血量 > 35% 且不会被秒杀
        my_hp_pct = self_hero["hp"] / max(self_hero["max_hp"], 1)
        enemy_nearby = False
        if enemy_hero and enemy_hero["hp"] > 0:
            my_loc = self_hero["location"]
            en_loc = enemy_hero["location"]
            dist = math.sqrt((my_loc["x"] - en_loc["x"]) ** 2 + (my_loc["z"] - en_loc["z"]) ** 2)
            enemy_nearby = dist < 450

        features["safe_to_sweep"] = (
            my_hp_pct > 0.35
            and not (enemy_nearby and my_hp_pct < 0.4)
        )

        # Check sweep target coverage
        # 检查扫射目标覆盖
        if enemy_hero:
            features["sweep_can_hit_hero"] = self._can_hit_in_front(
                self_hero["location"], enemy_hero["location"]
            )

        enemy_minion_count = 0
        enemy_tower = None
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != main_camp:
                if npc.get("sub_type") == 11 and npc.get("hp", 0) > 0:
                    if self._can_hit_in_front(self_hero["location"], npc["location"]):
                        enemy_minion_count += 1
                elif npc.get("sub_type") == 21 and npc.get("hp", 0) > 0:
                    enemy_tower = npc

        features["sweep_can_hit_minion_count"] = enemy_minion_count

        if enemy_tower:
            features["sweep_can_hit_tower"] = self._can_hit_in_front(
                self_hero["location"], enemy_tower["location"]
            )

        return features

    def get_direnjie_features(self, frame_state, main_camp):
        """Compute Di Renjie-specific features.
        计算狄仁杰专用特征。
        
        Returns dict with:
        - passive_stack_est: estimated passive stacks (0-5)
        - next_attack_is_card_est: estimated next attack has enhanced card
        - passive_high: passive stacks >= 3
        - card_can_attack_tower_outside: enhanced card can hit tower from outside
        - card_can_pierce_minion_to_hero: card can pierce minion to hit hero
        """
        features = {
            "passive_stack_est": 0,
            "next_attack_is_card_est": 0.0,
            "passive_high": False,
            "card_can_attack_tower_outside": False,
            "card_can_pierce_minion_to_hero": False,
        }

        self_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == main_camp and hero["hp"] > 0:
                self_hero = hero
            elif hero["camp"] != main_camp and hero["hp"] > 0:
                enemy_hero = hero

        if self_hero is None:
            return features

        # Estimate passive stacks: based on recent attack count
        # 估计被动层数：基于近期攻击次数
        passive_stacks = min(5, self.attack_count % 6)
        features["passive_stack_est"] = passive_stacks
        features["passive_high"] = passive_stacks >= 3

        # Estimate next attack is enhanced card
        # 估计下一次攻击是强化牌
        if passive_stacks >= 3:
            features["next_attack_is_card_est"] = 0.4 + 0.1 * (passive_stacks - 3)
        else:
            features["next_attack_is_card_est"] = 0.1

        # Check tower outside attack capability
        # 检查塔外攻击能力
        enemy_tower = None
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != main_camp and npc.get("sub_type") == 21 and npc.get("hp", 0) > 0:
                enemy_tower = npc
                break

        if enemy_tower and self_hero:
            my_loc = self_hero["location"]
            t_loc = enemy_tower["location"]
            dist = math.sqrt((my_loc["x"] - t_loc["x"]) ** 2 + (my_loc["z"] - t_loc["z"]) ** 2)
            if dist > TOWER_ATTACK_RANGE:
                features["card_can_attack_tower_outside"] = features["next_attack_is_card_est"] > 0.3

        # Check card pierce capability
        # 检查强化牌穿透能力
        if enemy_hero and self_hero:
            can_pierce = False
            for npc in frame_state.get("npc_states", []):
                if npc.get("camp") != main_camp and npc.get("sub_type") == 11 and npc.get("hp", 0) > 0:
                    if self._targets_aligned(
                        self_hero["location"],
                        npc["location"],
                        enemy_hero["location"],
                    ):
                        can_pierce = True
                        break
            features["card_can_pierce_minion_to_hero"] = can_pierce

        return features

    def _can_hit_in_front(self, hero_loc, target_loc, max_angle=60):
        """Check if target is in front of hero within max_angle degrees.
        检查目标是否在英雄前方指定角度范围内。
        """
        dx = target_loc["x"] - hero_loc["x"]
        dz = target_loc["z"] - hero_loc["z"]
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 1:
            return False
        cos_angle = abs(dz) / dist
        angle = math.degrees(math.acos(min(1.0, cos_angle)))
        return angle < max_angle

    def _targets_aligned(self, origin, target1, target2):
        """Check if target1 and target2 are aligned from origin.
        检查从origin出发，两个目标是否大致对齐。
        """
        dx1 = target1["x"] - origin["x"]
        dz1 = target1["z"] - origin["z"]
        dx2 = target2["x"] - origin["x"]
        dz2 = target2["z"] - origin["z"]

        dist1 = math.sqrt(dx1 * dx1 + dz1 * dz1)
        dist2 = math.sqrt(dx2 * dx2 + dz2 * dz2)

        if dist1 < 1 or dist2 < 1:
            return False

        dot = (dx1 * dx2 + dz1 * dz2) / (dist1 * dist2)
        return dot > 0.85
