#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward design for Gorge Chase PPO.
"""

import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_STEP = 2000.0
MAX_SCORE = 3000.0
MAX_DISTANCE = MAP_SIZE * 1.42
MAX_PROBE_STEPS = 4
STUCK_DISTANCE = 0.8
POSITION_HISTORY = 6


MOVE_DIRS = [
    (1.0, 0.0),
    (1.0, -1.0),
    (0.0, -1.0),
    (-1.0, -1.0),
    (-1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
]


GRID_DIRS = [
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _distance(x1, z1, x2, z2):
    dx = x1 - x2
    dz = z1 - z2
    return float(np.sqrt(dx * dx + dz * dz))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 1.0
        self.last_treasure_dist_norm = 1.0
        self.last_score = 0.0
        self.position_history = []
        self.last_stuck_ratio = 0.0

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation.get("map_info")
        legal_act_raw = observation.get("legal_action") or observation.get("legal_act") or []

        self.step_no = int(observation.get("step_no", 0))
        self.max_step = int(env_info.get("max_step", 1000))
        total_score = float(env_info.get("total_score", 0.0))

        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        hero_x = float(hero_pos.get("x", 0.0))
        hero_z = float(hero_pos.get("z", 0.0))
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))

        monsters = self._normalize_entities(frame_state.get("monsters", []))
        treasures = self._normalize_entities(frame_state.get("treasures", frame_state.get("treasure", [])))
        buffs = self._normalize_entities(frame_state.get("buffs", frame_state.get("buff", [])))

        nearest_monsters = self._nearest_entities(hero_x, hero_z, monsters, limit=2)
        while len(nearest_monsters) < 2:
            nearest_monsters.append(self._empty_entity())
        nearest_treasure = self._nearest_entities(hero_x, hero_z, treasures, limit=1)[0] if treasures else self._empty_entity()
        nearest_buff = self._nearest_entities(hero_x, hero_z, buffs, limit=1)[0] if buffs else self._empty_entity()

        legal_action = self._extract_legal_action(legal_act_raw)
        local_map_feature = self._local_map_feature(map_info)
        clearance_feature = self._directional_clearance_feature(map_info)
        trap_feature = self._trap_feature(clearance_feature)
        stuck_feature = self._stuck_feature(hero_x, hero_z)
        safe_treasure_feature = self._safe_treasure_feature(
            nearest_treasure, nearest_monsters[0], clearance_feature, trap_feature
        )

        hero_feat = np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                _norm(flash_cd, MAX_FLASH_CD),
                _norm(buff_remain, MAX_BUFF_DURATION),
                _norm(self.step_no, max(1.0, float(self.max_step))),
                min(1.0, total_score / MAX_SCORE),
            ],
            dtype=np.float32,
        )

        feature = np.concatenate(
            [
                hero_feat,
                self._entity_feature(nearest_monsters[0], include_speed=True),
                self._entity_feature(nearest_monsters[1], include_speed=True),
                self._entity_feature(nearest_treasure, include_speed=False),
                self._entity_feature(nearest_buff, include_speed=False),
                local_map_feature,
                np.array(legal_action, dtype=np.float32),
                clearance_feature,
                trap_feature,
                stuck_feature,
                safe_treasure_feature,
                np.array(
                    [
                        _norm(self.step_no, max(1.0, float(self.max_step))),
                        1.0 - _norm(self.step_no, max(1.0, float(self.max_step))),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        reward = [self._shape_reward(total_score, nearest_monsters[0], nearest_treasure, nearest_buff)]
        return feature, legal_action, reward

    def _shape_reward(self, total_score, nearest_monster, nearest_treasure, nearest_buff):
        survive_reward = 0.015

        cur_monster_dist = nearest_monster["dist_norm"]
        monster_delta = cur_monster_dist - self.last_min_monster_dist_norm
        danger_penalty = -0.12 if cur_monster_dist < 0.06 else 0.0
        distance_reward = 0.08 * monster_delta

        cur_treasure_dist = nearest_treasure["dist_norm"]
        treasure_progress = 0.0
        if nearest_treasure["visible"] > 0:
            treasure_progress = 0.05 * (self.last_treasure_dist_norm - cur_treasure_dist)

        score_delta = total_score - self.last_score
        treasure_bonus = max(0.0, score_delta) / 100.0

        buff_bonus = 0.0
        if nearest_buff["visible"] > 0 and cur_monster_dist < 0.18:
            buff_bonus = 0.03 * (1.0 - nearest_buff["dist_norm"])

        stuck_penalty = -0.02 * self.last_stuck_ratio

        self.last_min_monster_dist_norm = cur_monster_dist
        self.last_treasure_dist_norm = cur_treasure_dist if nearest_treasure["visible"] > 0 else 1.0
        self.last_score = total_score

        return float(
            survive_reward
            + distance_reward
            + treasure_progress
            + treasure_bonus
            + buff_bonus
            + danger_penalty
            + stuck_penalty
        )

    def _extract_legal_action(self, legal_act_raw):
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                legal_action = [int(legal_act_raw[i]) if i < len(legal_act_raw) else 0 for i in range(16)]
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if i in valid_set else 0 for i in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16
        return legal_action

    def _normalize_entities(self, entities):
        result = []
        for entity in entities or []:
            pos = entity.get("pos", {})
            result.append(
                {
                    "visible": float(entity.get("is_in_view", 1)),
                    "x": float(pos.get("x", 0.0)),
                    "z": float(pos.get("z", 0.0)),
                    "speed": float(entity.get("speed", 0.0)),
                }
            )
        return result

    def _nearest_entities(self, hero_x, hero_z, entities, limit=1):
        scored = []
        for entity in entities:
            if entity["visible"] <= 0:
                continue
            dx = entity["x"] - hero_x
            dz = entity["z"] - hero_z
            dist = float(np.sqrt(dx * dx + dz * dz))
            scored.append(
                {
                    "visible": 1.0,
                    "dx_norm": np.clip(dx / MAP_SIZE, -1.0, 1.0),
                    "dz_norm": np.clip(dz / MAP_SIZE, -1.0, 1.0),
                    "speed_norm": _norm(entity["speed"], MAX_MONSTER_SPEED),
                    "dist_norm": _norm(dist, MAX_DISTANCE),
                }
            )
        scored.sort(key=lambda item: item["dist_norm"])
        return scored[:limit]

    def _empty_entity(self):
        return {"visible": 0.0, "dx_norm": 0.0, "dz_norm": 0.0, "speed_norm": 0.0, "dist_norm": 1.0}

    def _entity_feature(self, entity, include_speed=False):
        if include_speed:
            return np.array(
                [
                    entity["visible"],
                    entity["dx_norm"],
                    entity["dz_norm"],
                    entity["speed_norm"],
                    entity["dist_norm"],
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                entity["visible"],
                entity["dx_norm"],
                entity["dz_norm"],
                entity["dist_norm"],
                1.0 - entity["dist_norm"],
            ],
            dtype=np.float32,
        )

    def _local_map_feature(self, map_info):
        feature = np.zeros(16, dtype=np.float32)
        if map_info is None or len(map_info) == 0:
            return feature
        center = len(map_info) // 2
        flat_idx = 0
        for row in range(center - 2, center + 2):
            for col in range(center - 2, center + 2):
                if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                    feature[flat_idx] = float(map_info[row][col] != 0)
                flat_idx += 1
        return feature

    def _directional_clearance_feature(self, map_info):
        feature = np.zeros(8, dtype=np.float32)
        if map_info is None or len(map_info) == 0:
            return feature

        rows = len(map_info)
        cols = len(map_info[0])
        center_r = rows // 2
        center_c = cols // 2

        for idx, (dr, dc) in enumerate(GRID_DIRS):
            clear_steps = 0
            for step in range(1, MAX_PROBE_STEPS + 1):
                rr = center_r + dr * step
                cc = center_c + dc * step
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break
                if map_info[rr][cc] != 0:
                    break
                clear_steps += 1
            feature[idx] = clear_steps / float(MAX_PROBE_STEPS)
        return feature

    def _trap_feature(self, clearance_feature):
        orth = [clearance_feature[0], clearance_feature[2], clearance_feature[4], clearance_feature[6]]
        blocked = [1 if c <= 0.25 else 0 for c in orth]
        blocked_cnt = sum(blocked)

        is_dead_end = 1.0 if blocked_cnt >= 3 else 0.0
        is_corner = 1.0 if blocked_cnt == 2 and self._has_adjacent_pair(blocked) else 0.0
        narrowness = 1.0 - float(np.mean(orth))
        blocked_ratio = blocked_cnt / 4.0

        return np.array(
            [
                blocked_ratio,
                is_dead_end,
                is_corner,
                narrowness,
            ],
            dtype=np.float32,
        )

    def _stuck_feature(self, hero_x, hero_z):
        self.position_history.append((hero_x, hero_z))
        if len(self.position_history) > POSITION_HISTORY:
            self.position_history.pop(0)

        move_dist = 0.0
        if len(self.position_history) >= 2:
            prev_x, prev_z = self.position_history[-2]
            move_dist = _distance(hero_x, hero_z, prev_x, prev_z)

        recent_moves = []
        for idx in range(1, len(self.position_history)):
            x1, z1 = self.position_history[idx - 1]
            x2, z2 = self.position_history[idx]
            recent_moves.append(_distance(x1, z1, x2, z2))

        avg_move = float(np.mean(recent_moves[-4:])) if recent_moves else 0.0

        still_tail = 0
        for dist in reversed(recent_moves):
            if dist < STUCK_DISTANCE:
                still_tail += 1
            else:
                break
        stuck_ratio = min(1.0, still_tail / 4.0)

        oscillate = 0.0
        if len(self.position_history) >= 4:
            x_prev2, z_prev2 = self.position_history[-3]
            x_prev1, z_prev1 = self.position_history[-2]
            x_prev3, z_prev3 = self.position_history[-4]
            if _distance(hero_x, hero_z, x_prev2, z_prev2) < STUCK_DISTANCE and _distance(
                x_prev1, z_prev1, x_prev3, z_prev3
            ) < STUCK_DISTANCE:
                oscillate = 1.0

        self.last_stuck_ratio = stuck_ratio

        return np.array(
            [
                _norm(move_dist, 4.0),
                _norm(avg_move, 4.0),
                stuck_ratio,
                oscillate,
            ],
            dtype=np.float32,
        )

    def _safe_treasure_feature(self, nearest_treasure, nearest_monster, clearance_feature, trap_feature):
        if nearest_treasure["visible"] <= 0:
            return np.zeros(4, dtype=np.float32)

        treasure_dist = nearest_treasure["dist_norm"]
        monster_dist = nearest_monster["dist_norm"]
        monster_margin = float(np.clip(monster_dist - treasure_dist, -1.0, 1.0))
        monster_margin_norm = 0.5 + 0.5 * monster_margin

        treasure_dir_idx = self._nearest_direction_idx(nearest_treasure["dx_norm"], nearest_treasure["dz_norm"])
        treasure_clearance = clearance_feature[treasure_dir_idx] if treasure_dir_idx >= 0 else 0.0

        is_safe = 1.0 if monster_margin > 0.05 and treasure_clearance > 0.25 and trap_feature[1] < 0.5 else 0.0

        return np.array(
            [
                1.0,
                monster_margin_norm,
                treasure_clearance,
                is_safe * (1.0 - treasure_dist),
            ],
            dtype=np.float32,
        )

    def _nearest_direction_idx(self, dx_norm, dz_norm):
        if abs(dx_norm) < 1e-6 and abs(dz_norm) < 1e-6:
            return -1
        norm = np.sqrt(dx_norm * dx_norm + dz_norm * dz_norm)
        ux = dx_norm / norm
        uz = dz_norm / norm

        best_idx = 0
        best_dot = -1e9
        for idx, (mx, mz) in enumerate(MOVE_DIRS):
            m_norm = np.sqrt(mx * mx + mz * mz)
            dot = ux * (mx / m_norm) + uz * (mz / m_norm)
            if dot > best_dot:
                best_dot = dot
                best_idx = idx
        return best_idx

    def _has_adjacent_pair(self, blocked):
        for idx in range(len(blocked)):
            if blocked[idx] and blocked[(idx + 1) % len(blocked)]:
                return True
        return False
