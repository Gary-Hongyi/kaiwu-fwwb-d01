#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
9-reward-type reward manager from taichu_code, adapted for current protocol.
"""

import math
from agent_ppo.conf.conf import GameConfig


def _fedexing_get_first(data, *keys, default=0):
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]
    return default


def fedexing_init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = FedExingRewardStruct(weight)
    return calc_frame_map


class FedExingGameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = fedexing_init_calc_frame_map()
        self.m_main_calc_frame_map = fedexing_init_calc_frame_map()
        self.m_enemy_calc_frame_map = fedexing_init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self._win_loss_given = False
        self._frenzy_was_active = False
        self._frenzy_activation_frame = -1
        self._frenzy_hurt_at_activation = 0
        self._last_lane_progress = None
        self._last_enemy_tower_hp_ratio = None
        self._last_enemy_tower_hp_for_idle = None
        self._grass_idle_count = 0
        self._tower_edge_idle_count = 0
        self.init_max_exp_of_each_hero()

    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        level = max(this_hero_info.get("level", 1), 1)
        exp = this_hero_info.get("exp", 0)
        for i in range(1, level):
            exp_sum += self.m_each_level_max_exp.get(i, 0)
        exp_sum += exp
        return exp_sum

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        mp = (main_tower["location"]["x"], main_tower["location"]["z"])
        ep = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hp = (main_hero["location"]["x"], main_hero["location"]["z"])
        dist_h2e = math.dist(hp, ep)
        dist_m2e = math.dist(mp, ep)
        if dist_m2e <= 0:
            return 0
        if main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1) > 0.99 and dist_h2e > dist_m2e:
            return (dist_m2e - dist_h2e) / dist_m2e
        return 0

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id", hero.get("player_id")) == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        if main_camp >= 0:
            self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        if enemy_camp >= 0:
            self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            m = self.m_main_calc_frame_map[reward_name]
            e = self.m_enemy_calc_frame_map[reward_name]
            if reward_name == "hp_point":
                if m.last_frame_value == 0.0 and e.last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif m.last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - e.cur_frame_value
                    reward_struct.last_frame_value = 0 - e.last_frame_value
                elif e.last_frame_value == 0.0:
                    reward_struct.cur_frame_value = m.cur_frame_value - 0
                    reward_struct.last_frame_value = m.last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = m.cur_frame_value - e.cur_frame_value
                    reward_struct.last_frame_value = m.last_frame_value - e.last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = m.cur_frame_value
                reward_struct.last_frame_value = m.last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data.get("hero_states", []):
                    if hero.get("runtime_id", hero.get("player_id")) == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero.get("level", 1) >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = m.cur_frame_value - e.cur_frame_value
                    reward_struct.last_frame_value = m.last_frame_value - e.last_frame_value
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = m.cur_frame_value
                if GameConfig.REMOVE_FORWARD_AFTER is not None:
                    reward_struct.value *= (frame_no <= GameConfig.REMOVE_FORWARD_AFTER)
            elif reward_name == "last_hit":
                reward_struct.value = m.cur_frame_value
            elif reward_name == "win_loss":
                reward_struct.value = 0.0
                if self._win_loss_given:
                    pass
                else:
                    for organ in frame_data.get("npc_states", []):
                        if organ.get("sub_type") == 21:
                            if organ.get("camp") == self.main_hero_camp:
                                if organ.get("hp", 0) <= 100:
                                    reward_struct.value = -1.0
                                    self._win_loss_given = True
                            else:
                                if organ.get("hp", 0) <= 100:
                                    reward_struct.value = 1.0
                                    self._win_loss_given = True
            elif reward_name == "frenzy_use":
                reward_struct.value = self._evaluate_frenzy_use(frame_data)
            elif reward_name == "damage_taken":
                # 只看己方受伤增量（不做敌我差值），受伤越多值越大
                reward_struct.cur_frame_value = m.cur_frame_value
                reward_struct.last_frame_value = m.last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "lane_progress":
                reward_struct.value = self._evaluate_lane_progress(frame_data)
            elif reward_name == "minion_engage":
                reward_struct.value = self._evaluate_minion_engage(frame_data)
            elif reward_name == "base_idle":
                reward_struct.value = self._evaluate_base_idle(frame_data)
            elif reward_name == "danger_zone":
                reward_struct.value = self._evaluate_danger_zone(frame_data)
            elif reward_name == "tower_damage":
                reward_struct.value = self._evaluate_tower_damage(frame_data)
            elif reward_name == "push_window":
                reward_struct.value = self._evaluate_push_window(frame_data)
            elif reward_name == "grass_idle":
                reward_struct.value = self._evaluate_grass_idle(frame_data)
            elif reward_name == "tower_edge_idle":
                reward_struct.value = self._evaluate_tower_edge_idle(frame_data)
            else:
                reward_struct.cur_frame_value = m.cur_frame_value - e.cur_frame_value
                reward_struct.last_frame_value = m.last_frame_value - e.last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            time_scale = 1.0
            if self.time_scale_arg > 0:
                if reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                    time_scale = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

            reward_dict[reward_name + "_origin"] = reward_struct.value
            reward_dict[reward_name + "_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[reward_name + "_weight"]

        reward_dict["reward_sum"] = reward_sum

    def _evaluate_frenzy_use(self, frame_data):
        """评估狂暴释放质量。

        检测狂暴buff激活（buff_id 801100），在激活帧判断释放条件：
        - 有效释放（+1）：敌方英雄可见且在有效范围内，自身血量>35%，
          处于对拼/追杀/击杀窗口
        - 无效释放（-1）：开局释放、敌人距离过远、逃跑路上释放、
          敌塔下无收益释放

        狂暴持续期间打出伤害/击杀额外给予正向奖励（通过 hurt_to_hero 和 kill 已覆盖）。
        """
        main_hero = None
        enemy_hero = None
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id", hero.get("player_id")) == self.main_hero_player_id:
                main_hero = hero
            else:
                enemy_hero = hero

        if main_hero is None:
            return 0.0

        # 检测狂暴buff是否激活（buff_id 801100 对应狂暴技能80110）
        frenzy_active = False
        buff_skills = main_hero.get("buff_skills", [])
        if isinstance(buff_skills, list):
            for buff in buff_skills:
                buff_id = buff.get("buff_id", buff) if isinstance(buff, dict) else buff
                if int(buff_id) == 801100:
                    frenzy_active = True
                    break

        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))

        # 检测狂暴刚刚激活（上一帧没有，这一帧有）
        just_activated = frenzy_active and not self._frenzy_was_active
        self._frenzy_was_active = frenzy_active

        if not just_activated:
            return 0.0

        # 狂暴刚激活，评估释放质量
        self._frenzy_activation_frame = frame_no
        self._frenzy_hurt_at_activation = _fedexing_get_first(
            main_hero, "total_hurt_to_hero", "totalHurtToHero"
        )

        main_hp = main_hero.get("hp", 0)
        main_max_hp = max(main_hero.get("max_hp", 1), 1)
        main_hp_ratio = main_hp / main_max_hp

        # 惩罚条件：开局释放（前90帧约3秒）
        if frame_no < 90:
            return -1.0

        # 惩罚条件：自身血量过低（逃跑中释放）
        if main_hp_ratio <= 0.35:
            return -1.0

        # 惩罚条件：敌方英雄不可见或不存在
        if enemy_hero is None:
            return -1.0
        enemy_hp = enemy_hero.get("hp", 0)
        if enemy_hp <= 0:
            return -1.0

        # 计算与敌方英雄距离
        main_loc = main_hero.get("location", {})
        enemy_loc = enemy_hero.get("location", {})
        if not main_loc or not enemy_loc:
            return -1.0

        mx, mz = main_loc.get("x", 0), main_loc.get("z", 0)
        ex, ez = enemy_loc.get("x", 0), enemy_loc.get("z", 0)
        dist_to_enemy = math.dist((mx, mz), (ex, ez))

        # 惩罚条件：敌人距离过远（超过8000，普攻/技能打不到）
        if dist_to_enemy > 8000:
            return -1.0

        # 惩罚条件：在敌方塔下且敌方血量不低（无击杀收益）
        enemy_max_hp = max(enemy_hero.get("max_hp", 1), 1)
        enemy_hp_ratio = enemy_hp / enemy_max_hp
        for organ in frame_data.get("npc_states", []):
            if organ.get("sub_type") == 21 and organ.get("camp") != self.main_hero_camp:
                tower_loc = organ.get("location", {})
                if tower_loc:
                    tx, tz = tower_loc.get("x", 0), tower_loc.get("z", 0)
                    dist_to_tower = math.dist((mx, mz), (tx, tz))
                    if dist_to_tower < 7000 and organ.get("hp", 0) > 0:
                        # 在敌塔下，只有击杀窗口才允许
                        if enemy_hp_ratio >= 0.35:
                            return -1.0

        # 有效释放：敌方在范围内，自身血量安全，处于战斗状态
        return 1.0

    @staticmethod
    def _fedexing_point(actor):
        loc = actor.get("location") if isinstance(actor, dict) else None
        if isinstance(loc, dict):
            return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            return float(loc[0]), float(loc[1])
        return None

    @staticmethod
    def _fedexing_dist(p1, p2):
        if p1 is None or p2 is None:
            return float("inf")
        return math.dist(p1, p2)

    @staticmethod
    def _fedexing_clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def _fedexing_hp_ratio(actor):
        if not isinstance(actor, dict):
            return 0.0
        hp = float(_fedexing_get_first(actor, "hp", default=0) or 0)
        max_hp = float(_fedexing_get_first(actor, "max_hp", "maxHp", default=1) or 1)
        return max(0.0, min(1.0, hp / max(max_hp, 1.0)))

    @staticmethod
    def _fedexing_is_alive(actor):
        return isinstance(actor, dict) and float(actor.get("hp", 0) or 0) > 0

    def _fedexing_main_enemy_heroes(self, frame_data):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id", hero.get("player_id")) == self.main_hero_player_id:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _fedexing_structures(self, frame_data, camp):
        own_crystal, enemy_crystal, enemy_tower = None, None, None
        for npc in frame_data.get("npc_states", []):
            sub_type = npc.get("sub_type")
            npc_camp = npc.get("camp")
            if sub_type == 24:
                if npc_camp == camp:
                    own_crystal = npc
                else:
                    enemy_crystal = npc
            elif sub_type == 21 and npc_camp != camp and npc.get("hp", 0) > 0:
                enemy_tower = npc
        return own_crystal, enemy_crystal, enemy_tower

    @staticmethod
    def _fedexing_minions(frame_data, camp, enemy=False):
        minions = []
        for npc in frame_data.get("npc_states", []):
            if npc.get("sub_type") != 11 or npc.get("hp", 0) <= 0:
                continue
            is_enemy = npc.get("camp") != camp
            if is_enemy == enemy:
                minions.append(npc)
        return minions

    def _fedexing_has_minion_near(self, minions, point, radius):
        return any(self._fedexing_dist(self._fedexing_point(minion), point) <= radius for minion in minions)

    def _fedexing_enemy_tower_hp_ratio(self, frame_data, camp):
        for npc in frame_data.get("npc_states", []):
            if npc.get("sub_type") != 21 or npc.get("camp") == camp:
                continue
            max_hp = float(npc.get("max_hp", npc.get("maxHp", 1)) or 1)
            hp = float(npc.get("hp", 0) or 0)
            return self._fedexing_clamp(hp / max(max_hp, 1.0), 0.0, 1.0)
        return None

    def _fedexing_safe_push_context(self, frame_data):
        main_hero, enemy_hero = self._fedexing_main_enemy_heroes(frame_data)
        context = {
            "main_hero": main_hero,
            "enemy_hero": enemy_hero,
            "enemy_tower": None,
            "tower_pos": None,
            "hero_pos": None,
            "tower_dist": float("inf"),
            "tower_range": 6000,
            "tower_hp_ratio": None,
            "ally_cover": False,
            "enemy_threat": False,
        }
        if not self._fedexing_is_alive(main_hero):
            return False, context

        camp = main_hero.get("camp", self.main_hero_camp)
        _, _, enemy_tower = self._fedexing_structures(frame_data, camp)
        if enemy_tower is None:
            return False, context

        hero_pos = self._fedexing_point(main_hero)
        tower_pos = self._fedexing_point(enemy_tower)
        tower_range = int(enemy_tower.get("attack_range", 6000) or 6000)
        tower_hp_ratio = self._fedexing_enemy_tower_hp_ratio(frame_data, camp)
        tower_dist = self._fedexing_dist(hero_pos, tower_pos)
        ally_minions = self._fedexing_minions(frame_data, camp, enemy=False)
        ally_cover = self._fedexing_has_minion_near(ally_minions, tower_pos, tower_range + 1800)
        ally_cover = ally_cover or self._fedexing_has_minion_near(ally_minions, hero_pos, 5500)

        main_hp_ratio = self._fedexing_hp_ratio(main_hero)
        enemy_dist = float("inf")
        enemy_hp_ratio = 0.0
        enemy_threat = False
        if self._fedexing_is_alive(enemy_hero):
            enemy_dist = self._fedexing_dist(hero_pos, self._fedexing_point(enemy_hero))
            enemy_hp_ratio = self._fedexing_hp_ratio(enemy_hero)
            enemy_threat = enemy_dist <= 6000 and not (enemy_hp_ratio < 0.35 and main_hp_ratio > 0.55)

        context.update(
            enemy_tower=enemy_tower,
            tower_pos=tower_pos,
            hero_pos=hero_pos,
            tower_dist=tower_dist,
            tower_range=tower_range,
            tower_hp_ratio=tower_hp_ratio,
            ally_cover=ally_cover,
            enemy_threat=enemy_threat,
            main_hp_ratio=main_hp_ratio,
            enemy_dist=enemy_dist,
            enemy_hp_ratio=enemy_hp_ratio,
        )
        safe = (
            tower_hp_ratio is not None
            and tower_hp_ratio > 0
            and main_hp_ratio >= 0.45
            and ally_cover
            and tower_dist <= tower_range + 2800
            and not enemy_threat
        )
        return safe, context

    @staticmethod
    def _fedexing_projection_progress(point, start, end):
        if point is None or start is None or end is None:
            return 0.0
        sx, sz = start
        ex, ez = end
        px, pz = point
        vx, vz = ex - sx, ez - sz
        denom = vx * vx + vz * vz
        if denom <= 1e-6:
            return 0.0
        return ((px - sx) * vx + (pz - sz) * vz) / denom

    def _evaluate_lane_progress(self, frame_data):
        main_hero, _ = self._fedexing_main_enemy_heroes(frame_data)
        if main_hero is None or main_hero.get("hp", 0) <= 0:
            return 0.0
        camp = main_hero.get("camp", self.main_hero_camp)
        own_crystal, enemy_crystal, enemy_tower = self._fedexing_structures(frame_data, camp)
        start = self._fedexing_point(own_crystal)
        end = self._fedexing_point(enemy_tower) or self._fedexing_point(enemy_crystal)
        pos = self._fedexing_point(main_hero)
        raw_progress = self._fedexing_projection_progress(pos, start, end)
        progress = self._fedexing_clamp(raw_progress, 0.0, 0.65)
        if self._last_lane_progress is None:
            self._last_lane_progress = progress
            return 0.0

        delta = progress - self._last_lane_progress
        self._last_lane_progress = progress

        hp_ratio = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hp_ratio < 0.35 and delta < 0:
            return 0.0

        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        if frame_no > 4500:
            delta *= 0.25
        elif frame_no > 2500:
            delta *= 0.5
        return self._fedexing_clamp(delta * 5.0, -0.05, 0.08)

    def _evaluate_minion_engage(self, frame_data):
        main_hero, _ = self._fedexing_main_enemy_heroes(frame_data)
        if main_hero is None or main_hero.get("hp", 0) <= 0:
            return 0.0
        hp_ratio = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hp_ratio < 0.25:
            return 0.0
        camp = main_hero.get("camp", self.main_hero_camp)
        pos = self._fedexing_point(main_hero)
        enemy_minions = self._fedexing_minions(frame_data, camp, enemy=True)
        if not enemy_minions:
            return 0.0
        nearest = min(self._fedexing_dist(pos, self._fedexing_point(minion)) for minion in enemy_minions)
        if nearest <= 4500:
            return 0.08
        if nearest <= 8000:
            return 0.04
        return 0.0

    def _evaluate_base_idle(self, frame_data):
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        if frame_no < 180:
            return 0.0
        main_hero, enemy_hero = self._fedexing_main_enemy_heroes(frame_data)
        if main_hero is None or main_hero.get("hp", 0) <= 0:
            return 0.0
        hp_ratio = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        if hp_ratio < 0.35:
            return 0.0
        camp = main_hero.get("camp", self.main_hero_camp)
        own_crystal, _, _ = self._fedexing_structures(frame_data, camp)
        pos = self._fedexing_point(main_hero)
        own_crystal_pos = self._fedexing_point(own_crystal)
        if enemy_hero is not None and enemy_hero.get("hp", 0) > 0:
            if self._fedexing_dist(pos, self._fedexing_point(enemy_hero)) <= 7000:
                return 0.0
        dist_to_base = self._fedexing_dist(pos, own_crystal_pos)
        if dist_to_base <= 9000:
            return -0.12
        if dist_to_base <= 14000:
            return -0.06
        return 0.0

    def _evaluate_danger_zone(self, frame_data):
        main_hero, enemy_hero = self._fedexing_main_enemy_heroes(frame_data)
        if main_hero is None or main_hero.get("hp", 0) <= 0:
            return 0.0
        camp = main_hero.get("camp", self.main_hero_camp)
        pos = self._fedexing_point(main_hero)
        hp_ratio = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1)
        own_crystal, enemy_crystal, enemy_tower = self._fedexing_structures(frame_data, camp)
        ally_minions = self._fedexing_minions(frame_data, camp, enemy=False)
        penalty = 0.0
        safe_push, _ = self._fedexing_safe_push_context(frame_data)

        if enemy_tower is not None:
            tower_pos = self._fedexing_point(enemy_tower)
            tower_range = enemy_tower.get("attack_range", 6000) or 6000
            ally_cover = self._fedexing_has_minion_near(ally_minions, tower_pos, tower_range + 1200)
            tower_dist = self._fedexing_dist(pos, tower_pos)
            kill_window = False
            if enemy_hero is not None and enemy_hero.get("hp", 0) > 0:
                enemy_hp_ratio = enemy_hero.get("hp", 0) / max(enemy_hero.get("max_hp", 1), 1)
                kill_window = enemy_hp_ratio < 0.25 and hp_ratio > 0.55
            if tower_dist <= tower_range + 800 and (not ally_cover) and (not kill_window) and (not safe_push):
                penalty -= 0.22
            if tower_dist <= tower_range + 400 and hp_ratio < 0.45:
                penalty -= 0.15

        enemy_crystal_pos = self._fedexing_point(enemy_crystal)
        if enemy_crystal_pos is not None:
            crystal_dist = self._fedexing_dist(pos, enemy_crystal_pos)
            ally_near_crystal = self._fedexing_has_minion_near(ally_minions, enemy_crystal_pos, 9000)
            if crystal_dist <= 16000 and enemy_tower is not None and not ally_near_crystal:
                penalty -= 0.25

        start = self._fedexing_point(own_crystal)
        end = self._fedexing_point(enemy_crystal) or self._fedexing_point(enemy_tower)
        progress = self._fedexing_projection_progress(pos, start, end)
        if enemy_tower is not None and progress > 0.85:
            penalty -= 0.12

        return self._fedexing_clamp(penalty, -0.35, 0.0)

    def _evaluate_tower_damage(self, frame_data):
        main_hero, _ = self._fedexing_main_enemy_heroes(frame_data)
        if main_hero is None:
            return 0.0
        camp = main_hero.get("camp", self.main_hero_camp)
        cur_ratio = self._fedexing_enemy_tower_hp_ratio(frame_data, camp)
        if cur_ratio is None:
            cur_ratio = 0.0 if self._last_enemy_tower_hp_ratio is not None else None
        if cur_ratio is None:
            return 0.0
        if self._last_enemy_tower_hp_ratio is None:
            self._last_enemy_tower_hp_ratio = cur_ratio
            return 0.0

        delta = self._last_enemy_tower_hp_ratio - cur_ratio
        self._last_enemy_tower_hp_ratio = cur_ratio
        if delta <= 1e-5:
            return 0.0
        safe_push, _ = self._fedexing_safe_push_context(frame_data)
        scale = 14.0 if safe_push else 10.0
        return self._fedexing_clamp(delta * scale, 0.0, 0.6)

    def _evaluate_push_window(self, frame_data):
        safe_push, context = self._fedexing_safe_push_context(frame_data)
        if not safe_push:
            return 0.0
        # This is deliberately small. It says "stand here and attack now" without
        # becoming more attractive than actual tower damage.
        tower_margin = context["tower_dist"] - context["tower_range"]
        if tower_margin <= 800:
            return 0.05
        return 0.03

    def _evaluate_grass_idle(self, frame_data):
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        main_hero, enemy_hero = self._fedexing_main_enemy_heroes(frame_data)
        if frame_no < 240 or not self._fedexing_is_alive(main_hero):
            self._grass_idle_count = 0
            return 0.0

        in_grass = bool(
            _fedexing_get_first(main_hero, "is_in_grass", "isInGrass", "flag_in_grass", default=False)
        )
        if not in_grass:
            self._grass_idle_count = 0
            return 0.0

        hp_ratio = self._fedexing_hp_ratio(main_hero)
        if hp_ratio < 0.35:
            self._grass_idle_count = 0
            return 0.0

        pos = self._fedexing_point(main_hero)
        if self._fedexing_is_alive(enemy_hero):
            if self._fedexing_dist(pos, self._fedexing_point(enemy_hero)) <= 5000:
                self._grass_idle_count = 0
                return 0.0

        safe_push, _ = self._fedexing_safe_push_context(frame_data)
        if safe_push:
            self._grass_idle_count += 2
        else:
            self._grass_idle_count += 1
        if self._grass_idle_count <= 45:
            return 0.0
        return -0.08

    def _evaluate_tower_edge_idle(self, frame_data):
        safe_push, context = self._fedexing_safe_push_context(frame_data)
        if not safe_push:
            self._tower_edge_idle_count = 0
            self._last_enemy_tower_hp_for_idle = None
            return 0.0

        cur_ratio = context.get("tower_hp_ratio")
        if cur_ratio is None:
            return 0.0
        if self._last_enemy_tower_hp_for_idle is None:
            self._last_enemy_tower_hp_for_idle = cur_ratio
            self._tower_edge_idle_count = 0
            return 0.0
        if cur_ratio < self._last_enemy_tower_hp_for_idle - 1e-5:
            self._last_enemy_tower_hp_for_idle = cur_ratio
            self._tower_edge_idle_count = 0
            return 0.0

        self._last_enemy_tower_hp_for_idle = cur_ratio
        self._tower_edge_idle_count += 1
        if self._tower_edge_idle_count <= 45:
            return 0.0
        if context["tower_dist"] <= context["tower_range"] + 900:
            return -0.10
        return -0.06

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        if not frame_data:
            self.m_reward_value.clear()
            for reward_name in self.m_cur_calc_frame_map:
                self.m_reward_value[reward_name + "_origin"] = 0.0
                self.m_reward_value[reward_name + "_weight"] = 0.0
            self.m_reward_value["reward_sum"] = 0.0
            return self.m_reward_value

        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if hero["camp"] == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        if main_hero is None:
            return  # no hero found for this camp yet

        main_hero_hp = main_hero.get("hp", 0)
        main_hero_max_hp = max(main_hero.get("max_hp", 1), 1)
        main_hero_ep = main_hero.get("ep", 0)
        main_hero_max_ep = max(main_hero.get("max_ep", 0), 0)

        main_tower, enemy_tower = None, None
        for organ in frame_data.get("npc_states", []):
            organ_camp = organ.get("camp", 0)
            if organ.get("sub_type") == 21:
                if organ_camp == camp:
                    main_tower = organ
                else:
                    enemy_tower = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if reward_name == "money":
                reward_struct.cur_frame_value = _fedexing_get_first(main_hero, "money_cnt", "moneyCnt", "money")
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero.get("kill_cnt", 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero.get("dead_cnt", 0)
            elif reward_name == "tower_hp_point":
                if main_tower and main_tower.get("max_hp", 0) > 0:
                    reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
                else:
                    reward_struct.cur_frame_value = 0
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data.get("frame_action", {})
                if frame_action and "dead_action" in frame_action:
                    for dead_action in frame_action["dead_action"]:
                        killer = dead_action.get("killer", {})
                        death = dead_action.get("death", {})
                        if killer.get("runtime_id") == main_hero.get("runtime_id") and death.get("sub_type") == 11:
                            reward_struct.cur_frame_value += 1.0
                        elif enemy_hero and killer.get("runtime_id") == enemy_hero.get("runtime_id") and death.get("sub_type") == 11:
                            reward_struct.cur_frame_value -= 1.0
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "hurt_to_hero":
                reward_struct.cur_frame_value = _fedexing_get_first(main_hero, "total_hurt_to_hero", "totalHurtToHero")
            elif reward_name == "damage_taken":
                # 敌方英雄的 total_hurt_to_hero 就是对己方英雄造成的伤害
                # 用敌方的这个字段，确保只计算英雄间伤害
                if enemy_hero:
                    reward_struct.cur_frame_value = _fedexing_get_first(enemy_hero, "total_hurt_to_hero", "totalHurtToHero") / max(main_hero_max_hp, 1)
                else:
                    reward_struct.cur_frame_value = 0
            elif reward_name == "forward":
                if main_tower and enemy_tower:
                    reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
                else:
                    reward_struct.cur_frame_value = 0


class FedExingRewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
