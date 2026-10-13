#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and phase-aware reward design for Gorge Chase PPO.
"""

import numpy as np
from agent_ppo.conf.conf import Config

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 200.0
MAX_BUFF_DURATION = 50.0
MAX_PROBE_STEPS = 4
STUCK_DISTANCE = 0.35
POSITION_HISTORY = 6
ENABLE_RADICAL_FLASH = True
EARLY_FLASH_STEP = 300
EARLY_AXIS_THREAT_DIST = 4.2
EARLY_DIAGONAL_THREAT_DIST = 3.0
LATE_AXIS_THREAT_DIST = 5.4
LATE_DIAGONAL_THREAT_DIST = 6.2
BUFF_ACTIVE_FLASH_DISCOUNT = 0.8
BUFF_TREASURE_OVERRIDE_RATIO = 1.8
ENDGAME_RATIO = 0.75
MIDGAME_LOOP_START = 200
MIDGAME_LOOP_END = 300
LOCAL_LOOP_THRESHOLD = 4
VISIT_HEAT_SATURATION = 8.0
BLOCK_MEMORY_TTL = 24
CURRENT_BLOCK_PENALTY = 2.5
RECENT_BLOCK_PENALTY = 4.0
FLASH_MIN_CLEARANCE = 0.5
FLASH_WALL_BUMP_PENALTY = 0.25
FLASH_PINCH_FAIL_PENALTY = 0.15
FLASH_PINCH_SUCCESS_BONUS = 0.15
FLASH_AXIS_DISTANCE = 10.0
FLASH_DIAG_DISTANCE = 8.0
FLASH_DEADEND_BAD_PENALTY = 0.35
FLASH_DEADEND_STUCK_CLEARANCE = 0.25
BUFF_MEMORY_ENABLE = True
BUFF_MEMORY_RESPAWN_COOLDOWN = 200
BUFF_MEMORY_MAX_ENTRIES = 8
BUFF_MEMORY_TREASURE_OVERRIDE_RATIO = 2.2
BUFF_MEMORY_ARRIVAL_DIST = 1.5

REWARD_COMPONENT_KEYS = (
    "rw_phase_base",
    "rw_scene",
    "rw_treasure",
    "rw_buff_gain",
    "rw_buff_seek",
    "rw_buff_active",
    "rw_discovery",
    "rw_revisit_pen",
    "rw_dead_end",
    "rw_motion",
    "rw_escape",
    "rw_flash",
)

VISITED_NORM = 400.0
DISCOVERED_NORM = 1200.0
BOUNDARY_NORM = 400.0
DEAD_END_NORM = 240.0

LOCAL_DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
LOCAL_DIRS_8 = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]
SPATIAL_SIZE = Config.SPATIAL_MAP_SIZE
SPATIAL_RADIUS = Config.SPATIAL_MAP_RADIUS
SPATIAL_CHANNELS = Config.SPATIAL_MAP_CHANNELS


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _distance(x1, z1, x2, z2):
    dx = x1 - x2
    dz = z1 - z2
    return float(np.sqrt(dx * dx + dz * dz))


def _reverse_dir(action):
    return (int(action) + 4) % 8


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5
        self.last_treasure_count = 0
        self.last_collected_buff = 0
        self.last_buff_remain_norm = 0.0
        self.last_buff_dist_norm = 1.0
        self.last_in_dead_end = False
        self.last_escape_routes = 0.0
        self.last_wall_bump = 0.0
        self.monster2_spawned = False
        self.visited_cells = set()
        self.visit_counts = {}
        self.discovered_walkable = set()
        self.discovered_blocked = set()
        self.discovered_boundaries = set()
        self.discovered_dead_ends = set()
        self.position_history = []
        self.action_history = []
        self.bump_cells = {}
        self.dir_block_history = {}
        self.buff_cell_memory = {}
        self.buff_cell_cooldown = {}
        self.memory_buff_target_steps = 0
        self.memory_buff_arrivals = 0
        self.dead_end_steps = 0
        self.dead_end_escape_cnt = 0
        self.dead_end_exit_move_cnt = 0
        self.dead_end_escape_routes_sum = 0.0
        self.flash_escape_cnt = 0
        self.last_reward_breakdown = {key: 0.0 for key in REWARD_COMPONENT_KEYS}

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation.get("map_info")
        legal_act_raw = observation.get("legal_action") or observation.get("legal_act") or []

        self.step_no = int(observation.get("step_no", 0))
        self.max_step = int(env_info.get("max_step", 1000))
        self._decay_memory_trackers()

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = float(hero_pos["x"])
        hero_z = float(hero_pos["z"])
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0.0), MAX_FLASH_CD)
        buff_remaining_time = float(hero.get("buff_remaining_time", 0.0))
        buff_remain_norm = _norm(buff_remaining_time, MAX_BUFF_DURATION)
        hero_feat = np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                flash_cd_norm,
                buff_remain_norm,
            ],
            dtype=np.float32,
        )

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_dist_norm = 1.0
        max_monster_speed = 1.0
        for i in range(2):
            if i < len(monsters):
                monster = monsters[i]
                in_view = float(monster.get("is_in_view", 0))
                speed = float(monster.get("speed", 1.0))
                max_monster_speed = max(max_monster_speed, speed)
                if in_view > 0:
                    monster_pos = monster["pos"]
                    mx = float(monster_pos["x"])
                    mz = float(monster_pos["z"])
                    raw_dist = _distance(hero_x, hero_z, mx, mz)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    cur_min_dist_norm = min(cur_min_dist_norm, dist_norm)
                    monster_feats.append(
                        np.array(
                            [
                                in_view,
                                _norm(mx, MAP_SIZE),
                                _norm(mz, MAP_SIZE),
                                _norm(speed, MAX_MONSTER_SPEED),
                                dist_norm,
                            ],
                            dtype=np.float32,
                        )
                    )
                else:
                    monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        base_legal_action = self._extract_legal_action(legal_act_raw)
        progress_feat = np.array(
            [
                _norm(self.step_no, max(1.0, float(self.max_step))),
                1.0 - _norm(self.step_no, max(1.0, float(self.max_step))),
            ],
            dtype=np.float32,
        )

        map_feat, memory_feat, terrain_feat, clearance_feat, discovery_info = self._build_map_memory(
            map_info, hero_x, hero_z
        )
        spatial_map_feat = self._build_spatial_map(frame_state, map_info, hero_x, hero_z)
        motion_feat, motion_info = self._build_motion_context(hero_x, hero_z, last_action)
        self._update_block_memory(hero_x, hero_z, last_action, motion_info)
        base_masked_action = self._apply_obstacle_mask(base_legal_action, clearance_feat)
        if ENABLE_RADICAL_FLASH:
            legal_action = self._apply_radical_flash_strategy(
                base_masked_action,
                clearance_feat,
                memory_feat,
                monsters,
                hero_x,
                hero_z,
                map_info,
                self.step_no,
                buff_remaining_time > 0.5,
                hero.get("flash_cooldown", 0.0) <= 0.0,
            )
        else:
            legal_action = base_masked_action

        phase = self._get_phase(
            monsters,
            env_info.get("monster_interval", 200.0),
            env_info.get("monster_speedup", 300.0),
            max_monster_speed,
        )
        phase_feat = self._phase_one_hot(phase)

        collectible_dir, collectible_dist, collectible_info = self._extract_collectible_feature(
            frame_state=frame_state,
            clearance_feat=clearance_feat,
            hero_x=hero_x,
            hero_z=hero_z,
            buff_active=buff_remaining_time > 0.5,
            phase=phase,
        )
        danger_dir = self._extract_danger_dir(monsters, hero_x, hero_z)

        if len(monsters) >= 2:
            self.monster2_spawned = True
        monster_interval = float(env_info.get("monster_interval", 200.0))
        if self.monster2_spawned:
            monster2_countdown_norm = 0.0
        else:
            countdown = max(0.0, monster_interval - float(self.step_no))
            monster2_countdown_norm = _norm(countdown, max(1.0, monster_interval))
        monster2_countdown_feat = np.array([monster2_countdown_norm], dtype=np.float32)

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                collectible_dir,
                collectible_dist,
                danger_dir,
                np.array([flash_cd_norm], dtype=np.float32),
                monster2_countdown_feat,
                phase_feat,
                memory_feat,
                terrain_feat,
                clearance_feat,
                motion_feat,
                spatial_map_feat,
            ]
        ).astype(np.float32)

        reward = [
            self._shape_reward(
                phase=phase,
                hero=hero,
                cur_min_dist_norm=cur_min_dist_norm,
                terrain_feat=terrain_feat,
                memory_feat=memory_feat,
                clearance_feat=clearance_feat,
                discovery_info=discovery_info,
                motion_info=motion_info,
                last_action=last_action,
                hero_cell=(int(round(hero_x)), int(round(hero_z))),
                env_info=env_info,
                buff_remain_norm=buff_remain_norm,
                collectible_info=collectible_info,
                monsters_in_view_count=sum(
                    float(mon.get("is_in_view", 0.0)) > 0.0 for mon in monsters
                ),
            )
        ]
        return feature, legal_action, reward

    def _extract_legal_action(self, legal_act_raw):
        legal_action = [1] * 16
        if legal_act_raw is None:
            return legal_action
        if isinstance(legal_act_raw, np.ndarray):
            legal_act_raw = legal_act_raw.tolist()
        elif isinstance(legal_act_raw, tuple):
            legal_act_raw = list(legal_act_raw)
        if not isinstance(legal_act_raw, list) or not legal_act_raw:
            return legal_action

        flat_values = []
        queue = list(legal_act_raw)
        while queue:
            item = queue.pop(0)
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                queue = list(item) + queue
                continue
            flat_values.append(item)

        if not flat_values:
            return legal_action
        if all(isinstance(v, (bool, np.bool_)) for v in flat_values):
            legal_action = [int(flat_values[i]) if i < len(flat_values) else 0 for i in range(16)]
        elif len(flat_values) >= 16 and all(float(v) in (0.0, 1.0) for v in flat_values[:16]):
            legal_action = [int(float(flat_values[i])) for i in range(16)]
        else:
            valid_set = set()
            for value in flat_values:
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < 16:
                    valid_set.add(idx)
            if valid_set:
                legal_action = [1 if i in valid_set else 0 for i in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16
        return legal_action

    def _apply_obstacle_mask(self, legal_action, clearance_feat):
        masked = list(legal_action)
        for idx in range(8):
            if masked[idx] > 0 and clearance_feat[idx] <= 0.0:
                masked[idx] = 0
        if sum(masked[:8]) == 0:
            masked[:8] = legal_action[:8]
        if sum(masked) == 0:
            return legal_action if sum(legal_action) > 0 else [1] * 16
        return masked

    def _apply_radical_flash_strategy(
        self,
        legal_action,
        clearance_feat,
        memory_feat,
        monsters,
        hero_x,
        hero_z,
        map_info,
        step_no,
        buff_active,
        flash_ready,
    ):
        masked = list(legal_action)
        hero_cell = (int(round(hero_x)), int(round(hero_z)))
        if not flash_ready:
            for idx in range(8, 16):
                masked[idx] = 0
            return masked

        early_phase = step_no < EARLY_FLASH_STEP
        current_dead_end = float(memory_feat[7]) > 0.5
        exit_dir = int(np.argmax(clearance_feat))

        allow_flash = False
        allowed_flash_dirs = set()

        candidate_dirs = [idx for idx in range(8) if legal_action[8 + idx] > 0]

        if not candidate_dirs:
            for idx in range(8, 16):
                masked[idx] = 0
            return masked

        if current_dead_end:
            allow_flash = True

        threatening = []

        for monster in monsters:
            if float(monster.get("is_in_view", 0)) <= 0:
                continue
            mx = float(monster["pos"]["x"])
            mz = float(monster["pos"]["z"])
            dx = mx - hero_x
            dz = mz - hero_z
            dist = _distance(hero_x, hero_z, mx, mz)
            axis_aligned = abs(dx) < 0.5 or abs(dz) < 0.5
            if early_phase:
                trigger_dist = (
                    EARLY_AXIS_THREAT_DIST if axis_aligned else EARLY_DIAGONAL_THREAT_DIST
                )
            else:
                trigger_dist = (
                    LATE_AXIS_THREAT_DIST if axis_aligned else LATE_DIAGONAL_THREAT_DIST
                )
            if buff_active and not current_dead_end:
                trigger_dist = max(2.5, trigger_dist - BUFF_ACTIVE_FLASH_DISCOUNT)
            if dist > trigger_dist:
                continue

            threatening.append({"mx": mx, "mz": mz, "dist": dist, "axis_aligned": axis_aligned})

        if current_dead_end:
            viable_dirs = []
            for dir_idx in candidate_dirs:
                landing_info = self._evaluate_flash_landing(map_info, hero_x, hero_z, dir_idx)
                if not landing_info["walkable"]:
                    continue
                viable_dirs.append((dir_idx, landing_info))

            if threatening:
                allow_flash = True
                scored_dirs = []
                for dir_idx, landing_info in viable_dirs:
                    landing_x = landing_info["x"]
                    landing_z = landing_info["z"]
                    gains = [
                        _distance(landing_x, landing_z, threat["mx"], threat["mz"]) - threat["dist"]
                        for threat in threatening
                    ]
                    min_gain = min(gains)
                    avg_gain = float(sum(gains) / len(gains))
                    exit_bonus = 0.8 if dir_idx == exit_dir else 0.25
                    wall_cross_bonus = 0.45 if landing_info["crosses_wall"] else 0.0
                    open_bonus = 0.18 * landing_info["escape_routes"]
                    dead_end_penalty = 0.50 if landing_info["dead_end"] else 0.0
                    block_penalty = (
                        0.35
                        if self.dir_block_history.get((hero_cell, dir_idx), 0) > 0
                        and not landing_info["crosses_wall"]
                        else 0.0
                    )
                    score = (
                        min_gain * 2.5
                        + avg_gain
                        + 0.20 * float(clearance_feat[dir_idx])
                        + exit_bonus
                        + wall_cross_bonus
                        + open_bonus
                        - dead_end_penalty
                        - block_penalty
                    )
                    scored_dirs.append((score, min_gain, dir_idx))
                scored_dirs.sort(reverse=True)
                filtered = [dir_idx for score, min_gain, dir_idx in scored_dirs if min_gain >= 0.0]
                if not filtered and scored_dirs:
                    filtered = [scored_dirs[0][2]]
                for dir_idx in filtered[:2]:
                    allowed_flash_dirs.add(dir_idx)
            elif self.last_wall_bump > 0.5 and max(clearance_feat) <= FLASH_DEADEND_STUCK_CLEARANCE:
                allow_flash = True
                viable_dir_set = {dir_idx for dir_idx, _ in viable_dirs}
                if exit_dir in viable_dir_set:
                    allowed_flash_dirs.add(exit_dir)
                else:
                    scored_dirs = []
                    for dir_idx, landing_info in viable_dirs:
                        score = (
                            float(clearance_feat[dir_idx])
                            + 0.35 * landing_info["escape_routes"]
                            + (0.60 if landing_info["crosses_wall"] else 0.0)
                            - (0.50 if landing_info["dead_end"] else 0.0)
                            - (
                                0.35
                                if self.dir_block_history.get((hero_cell, dir_idx), 0) > 0
                                and not landing_info["crosses_wall"]
                                else 0.0
                            )
                        )
                        scored_dirs.append((score, dir_idx))
                    scored_dirs.sort(reverse=True)
                    for _, dir_idx in scored_dirs[:1]:
                        allowed_flash_dirs.add(dir_idx)

        elif threatening:
            allow_flash = True
            scored_dirs = []
            for dir_idx in candidate_dirs:
                landing_info = self._evaluate_flash_landing(map_info, hero_x, hero_z, dir_idx)
                if not landing_info["walkable"]:
                    continue
                landing_x = landing_info["x"]
                landing_z = landing_info["z"]
                gains = [
                    _distance(landing_x, landing_z, threat["mx"], threat["mz"]) - threat["dist"]
                    for threat in threatening
                ]
                min_gain = min(gains)
                avg_gain = float(sum(gains) / len(gains))
                block_penalty = 0.35 if self.dir_block_history.get((hero_cell, dir_idx), 0) > 0 else 0.0
                exit_bonus = 0.4 if current_dead_end and dir_idx == exit_dir else 0.0
                wall_cross_bonus = 0.18 if landing_info["crosses_wall"] else 0.0
                open_bonus = 0.12 * landing_info["escape_routes"]
                dead_end_penalty = 0.35 if landing_info["dead_end"] else 0.0
                score = (
                    min_gain * 2.0
                    + avg_gain
                    + 0.16 * float(clearance_feat[dir_idx])
                    + exit_bonus
                    + wall_cross_bonus
                    + open_bonus
                    - dead_end_penalty
                    - block_penalty
                )
                scored_dirs.append((score, min_gain, dir_idx))

            scored_dirs.sort(reverse=True)
            if len(threatening) >= 2:
                filtered = [dir_idx for score, min_gain, dir_idx in scored_dirs if min_gain > 0.05]
                if not filtered and scored_dirs:
                    filtered = [scored_dirs[0][2]]
                for dir_idx in filtered[:2]:
                    allowed_flash_dirs.add(dir_idx)
            else:
                filtered = [dir_idx for score, min_gain, dir_idx in scored_dirs if score > 0.05]
                if not filtered and scored_dirs:
                    filtered = [scored_dirs[0][2]]
                for dir_idx in filtered[:2]:
                    allowed_flash_dirs.add(dir_idx)

        for idx in range(8, 16):
            if not allow_flash or (idx - 8) not in allowed_flash_dirs:
                masked[idx] = 0

        if sum(masked) == 0:
            movement_fallback = list(masked)
            movement_fallback[:8] = legal_action[:8]
            if sum(movement_fallback) > 0:
                return movement_fallback
            return legal_action if sum(legal_action) > 0 else [1] * 16
        return masked

    def _flash_landing(self, hero_x, hero_z, dir_idx):
        dx_dir, dz_dir = LOCAL_DIRS_8[dir_idx]
        flash_dist = FLASH_AXIS_DISTANCE if (dx_dir == 0 or dz_dir == 0) else FLASH_DIAG_DISTANCE
        return hero_x + dx_dir * flash_dist, hero_z + dz_dir * flash_dist

    def _evaluate_flash_landing(self, map_info, hero_x, hero_z, dir_idx):
        landing_x, landing_z = self._flash_landing(hero_x, hero_z, dir_idx)
        result = {
            "x": landing_x,
            "z": landing_z,
            "walkable": True,
            "escape_routes": 0.0,
            "dead_end": False,
            "crosses_wall": False,
        }
        if map_info is None or len(map_info) == 0:
            return result

        rows = len(map_info)
        cols = len(map_info[0])
        center_r = rows // 2
        center_c = cols // 2
        hero_gx = int(round(hero_x))
        hero_gz = int(round(hero_z))
        landing_gx = int(round(landing_x))
        landing_gz = int(round(landing_z))
        landing_r = center_r + (landing_gz - hero_gz)
        landing_c = center_c + (landing_gx - hero_gx)

        if landing_r < 0 or landing_r >= rows or landing_c < 0 or landing_c >= cols:
            result["walkable"] = False
            return result
        if map_info[landing_r][landing_c] == 0:
            result["walkable"] = False
            return result

        dx_dir, dz_dir = LOCAL_DIRS_8[dir_idx]
        max_path_steps = int(max(abs(landing_gx - hero_gx), abs(landing_gz - hero_gz)))
        for step in range(1, max_path_steps):
            rr = center_r + dz_dir * step
            cc = center_c + dx_dir * step
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                break
            if map_info[rr][cc] == 0:
                result["crosses_wall"] = True
                break

        escape_routes = 0
        for dx, dz in LOCAL_DIRS_8:
            for step in range(1, MAX_PROBE_STEPS + 1):
                rr = landing_r + dz * step
                cc = landing_c + dx * step
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break
                if map_info[rr][cc] == 0:
                    break
                if step >= 2:
                    escape_routes += 1
                    break
        result["escape_routes"] = escape_routes / 8.0
        result["dead_end"] = (landing_gx, landing_gz) in self.discovered_dead_ends
        return result

    def _extract_collectible_feature(self, frame_state, clearance_feat, hero_x, hero_z, buff_active, phase):
        collectible_dir = np.zeros(8, dtype=np.float32)
        collectible_dist = np.array([1.0], dtype=np.float32)
        max_dist = MAP_SIZE * 1.41

        hero_cell = (int(round(hero_x)), int(round(hero_z)))
        endgame = (float(self.step_no) / max(1.0, float(self.max_step))) > ENDGAME_RATIO
        late_buff_mode = BUFF_MEMORY_ENABLE and not buff_active and (phase == "late" or endgame)
        live_buff_cells = set()
        buff_from_memory = False

        best = {
            "treasure": (float("inf"), float("inf"), -1),
            "buff": (float("inf"), float("inf"), -1),
        }

        def score_candidate(cell, dist, angle, from_memory=False):
            revisit_count = self.visit_counts.get(cell, 0)
            revisit_scale = 0.4 if from_memory else 1.2
            revisit_penalty = revisit_scale * min(max(0, revisit_count - 1), 3)
            cleanup_bonus = 2.0 * branch_bonus(cell)
            if endgame:
                cleanup_bonus += 1.0
            current_block_penalty = CURRENT_BLOCK_PENALTY if clearance_feat[angle] <= 0.0 else 0.0
            recent_block_penalty = (
                RECENT_BLOCK_PENALTY if self.dir_block_history.get((hero_cell, angle), 0) > 0 else 0.0
            )
            memory_bonus = 1.2 if from_memory and late_buff_mode else 0.0
            return dist + revisit_penalty - cleanup_bonus + current_block_penalty + recent_block_penalty - memory_bonus

        def branch_bonus(cell):
            cx, cz = cell
            if cell in self.discovered_boundaries and cell not in self.visited_cells:
                return 1.0
            for dx, dz in LOCAL_DIRS_4:
                neighbor = (cx + dx, cz + dz)
                if neighbor in self.discovered_boundaries and neighbor not in self.visited_cells:
                    return 1.0
            return 0.0

        for organ in frame_state.get("organs", []) or []:
            sub_type = int(organ.get("sub_type", -1))
            if sub_type not in (1, 2):
                continue
            if int(organ.get("status", 1)) != 1:
                continue
            organ_pos = organ["pos"]
            ox = float(organ_pos["x"])
            oz = float(organ_pos["z"])
            dx = ox - hero_x
            dz = oz - hero_z
            dist = _distance(hero_x, hero_z, ox, oz)
            angle = int((np.arctan2(dz, dx) + np.pi) / (np.pi / 4)) % 8
            organ_cell = (int(round(ox)), int(round(oz)))

            if sub_type == 2 and BUFF_MEMORY_ENABLE:
                self.buff_cell_memory[organ_cell] = self.step_no
                self.buff_cell_cooldown.pop(organ_cell, None)
                if len(self.buff_cell_memory) > BUFF_MEMORY_MAX_ENTRIES:
                    oldest = min(self.buff_cell_memory, key=self.buff_cell_memory.get)
                    self.buff_cell_memory.pop(oldest, None)
                    self.buff_cell_cooldown.pop(oldest, None)
                live_buff_cells.add(organ_cell)

            score = score_candidate(organ_cell, dist, angle, from_memory=False)

            key = "treasure" if sub_type == 1 else "buff"
            if score < best[key][0]:
                best[key] = (score, dist, angle)

        if late_buff_mode and best["buff"][2] < 0:
            for cell, seen_step in list(self.buff_cell_memory.items()):
                if self.buff_cell_cooldown.get(cell, 0) > self.step_no:
                    continue
                cx, cz = cell
                dist = _distance(hero_x, hero_z, float(cx), float(cz))
                if dist <= BUFF_MEMORY_ARRIVAL_DIST:
                    continue
                dx = float(cx) - hero_x
                dz = float(cz) - hero_z
                angle = int((np.arctan2(dz, dx) + np.pi) / (np.pi / 4)) % 8
                score = score_candidate(cell, dist, angle, from_memory=True)
                if score < best["buff"][0]:
                    best["buff"] = (score, dist, angle)
                    buff_from_memory = True

        if late_buff_mode:
            for cell in list(self.buff_cell_memory.keys()):
                if _distance(hero_x, hero_z, float(cell[0]), float(cell[1])) <= BUFF_MEMORY_ARRIVAL_DIST:
                    if cell in live_buff_cells:
                        self.memory_buff_arrivals += 1
                        self.buff_cell_cooldown.pop(cell, None)
                    else:
                        self.buff_cell_cooldown[cell] = self.step_no + BUFF_MEMORY_RESPAWN_COOLDOWN

        _, buff_dist, buff_angle = best["buff"]
        _, treasure_dist, treasure_angle = best["treasure"]

        target_type = "none"
        target_dist = float("inf")
        target_angle = -1
        if not buff_active and buff_angle >= 0:
            buff_treasure_ratio = (
                BUFF_MEMORY_TREASURE_OVERRIDE_RATIO if buff_from_memory else BUFF_TREASURE_OVERRIDE_RATIO
            )
            if treasure_angle >= 0 and buff_dist > treasure_dist * buff_treasure_ratio:
                target_type = "treasure"
                target_dist = treasure_dist
                target_angle = treasure_angle
            else:
                target_type = "buff"
                target_dist = buff_dist
                target_angle = buff_angle
        elif treasure_angle >= 0:
            target_type = "treasure"
            target_dist = treasure_dist
            target_angle = treasure_angle
        elif buff_angle >= 0:
            target_type = "buff"
            target_dist = buff_dist
            target_angle = buff_angle

        if target_angle >= 0 and clearance_feat[target_angle] > 0.0:
            collectible_dir[target_angle] = 1.0
            collectible_dist[0] = _norm(target_dist, max_dist)
            if target_type == "buff" and buff_from_memory:
                self.memory_buff_target_steps += 1

        collectible_info = {
            "target_type": target_type,
            "target_angle": target_angle,
            "has_buff": buff_angle >= 0,
            "buff_dist_norm": _norm(buff_dist, max_dist) if buff_angle >= 0 else 1.0,
            "treasure_dist_norm": _norm(treasure_dist, max_dist) if treasure_angle >= 0 else 1.0,
            "target_source": "memory" if target_type == "buff" and buff_from_memory else "visible",
        }
        return collectible_dir, collectible_dist, collectible_info

    def _extract_danger_dir(self, monsters, hero_x, hero_z):
        danger_dir = np.zeros(8, dtype=np.float32)
        base_threshold = 15.0
        base_speed = 1.0
        for monster in monsters:
            if float(monster.get("is_in_view", 0)) <= 0:
                continue
            monster_pos = monster["pos"]
            dx = float(monster_pos["x"]) - hero_x
            dz = float(monster_pos["z"]) - hero_z
            dist = _distance(hero_x, hero_z, float(monster_pos["x"]), float(monster_pos["z"]))
            speed = float(monster.get("speed", base_speed))
            dynamic_threshold = base_threshold * (speed / base_speed)
            if dist < dynamic_threshold:
                angle = int((np.arctan2(dz, dx) + np.pi) / (np.pi / 4)) % 8
                danger_dir[angle] = 1.0
        return danger_dir

    def _get_phase(self, monsters, monster_interval, monster_speedup, max_monster_speed):
        if max_monster_speed > 1.0:
            return "late"
        if float(monster_speedup) > 0 and self.step_no >= float(monster_speedup):
            return "late"
        if len(monsters) >= 2:
            return "mid"
        if float(monster_interval) > 0 and self.step_no >= float(monster_interval):
            return "mid"
        return "early"

    def _phase_one_hot(self, phase):
        if phase == "early":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if phase == "mid":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _build_map_memory(self, map_info, hero_x, hero_z):
        map_feat = np.zeros(16, dtype=np.float32)
        memory_feat = np.zeros(8, dtype=np.float32)
        terrain_feat = np.zeros(4, dtype=np.float32)
        clearance_feat = np.zeros(8, dtype=np.float32)
        discovery_info = {
            "new_visit": 0.0,
            "new_walkable": 0.0,
            "new_boundary": 0.0,
            "new_dead_end": 0.0,
        }

        hero_cell = (int(round(hero_x)), int(round(hero_z)))
        revisit = 1.0 if hero_cell in self.visited_cells else 0.0
        self.visited_cells.add(hero_cell)
        self.visit_counts[hero_cell] = self.visit_counts.get(hero_cell, 0) + 1
        if revisit < 0.5:
            discovery_info["new_visit"] = 1.0

        if map_info is None or len(map_info) == 0:
            memory_feat = np.array(
                [
                    _norm(len(self.visited_cells), VISITED_NORM),
                    _norm(len(self.discovered_walkable), DISCOVERED_NORM),
                    _norm(len(self.discovered_blocked), DISCOVERED_NORM),
                    _norm(len(self.discovered_boundaries), BOUNDARY_NORM),
                    _norm(len(self.discovered_dead_ends), DEAD_END_NORM),
                    revisit,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            return map_feat, memory_feat, terrain_feat, clearance_feat, discovery_info

        rows = len(map_info)
        cols = len(map_info[0])
        center_r = rows // 2
        center_c = cols // 2
        local_cells = {}
        observed_cells = {}
        new_walkable_count = 0.0

        for row in range(rows):
            for col in range(cols):
                gx = int(round(hero_x)) + (col - center_c)
                gz = int(round(hero_z)) + (row - center_r)
                walkable = int(map_info[row][col] != 0)
                observed_cells[(gx, gz)] = walkable
                if walkable > 0:
                    if (gx, gz) not in self.discovered_walkable:
                        new_walkable_count += 1.0
                    self.discovered_walkable.add((gx, gz))
                else:
                    self.discovered_blocked.add((gx, gz))

        flat_idx = 0
        for row in range(center_r - 2, center_r + 2):
            for col in range(center_c - 2, center_c + 2):
                walkable = 0.0
                if 0 <= row < rows and 0 <= col < cols:
                    walkable = float(map_info[row][col] != 0)
                    gx = int(round(hero_x)) + (col - center_c)
                    gz = int(round(hero_z)) + (row - center_r)
                    local_cells[(gx, gz)] = int(walkable)
                map_feat[flat_idx] = walkable
                flat_idx += 1

        discovery_info["new_walkable"] = min(new_walkable_count, 4.0)

        for idx, (dx, dz) in enumerate(LOCAL_DIRS_8):
            clear_steps = 0
            for step in range(1, MAX_PROBE_STEPS + 1):
                rr = center_r + dz * step
                cc = center_c + dx * step
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break
                if map_info[rr][cc] == 0:
                    break
                clear_steps += 1
            clearance_feat[idx] = clear_steps / float(MAX_PROBE_STEPS)

        local_walkable = [cell for cell, walkable in observed_cells.items() if walkable > 0]
        local_obstacles = sum(1 for walkable in local_cells.values() if walkable == 0)
        local_boundary_cells = set()
        local_dead_end_cells = self._detect_dead_end_corridors(observed_cells)

        for cell in local_walkable:
            blocked_neighbors = 0
            walkable_neighbors = 0
            boundary = False
            cx, cz = cell
            for dx, dz in LOCAL_DIRS_4:
                neighbor = (cx + dx, cz + dz)
                if neighbor not in observed_cells:
                    continue
                if observed_cells[neighbor] > 0:
                    walkable_neighbors += 1
                else:
                    blocked_neighbors += 1
                    boundary = True
            if boundary:
                local_boundary_cells.add(cell)
                if cell not in self.discovered_boundaries:
                    discovery_info["new_boundary"] += 1.0
                self.discovered_boundaries.add(cell)

        observed_walkable = set(local_walkable)
        self.discovered_dead_ends.difference_update(observed_walkable - local_dead_end_cells)
        for cell in local_dead_end_cells:
            if cell not in self.discovered_dead_ends:
                discovery_info["new_dead_end"] += 1.0
            self.discovered_dead_ends.add(cell)

        immediate_escape_routes = sum(1 for value in clearance_feat if value > 0.0)
        current_boundary = 1.0 if hero_cell in self.discovered_boundaries else 0.0
        current_dead_end = 1.0 if hero_cell in self.discovered_dead_ends else 0.0

        memory_feat = np.array(
            [
                _norm(len(self.visited_cells), VISITED_NORM),
                _norm(len(self.discovered_walkable), DISCOVERED_NORM),
                _norm(len(self.discovered_blocked), DISCOVERED_NORM),
                _norm(len(self.discovered_boundaries), BOUNDARY_NORM),
                _norm(len(self.discovered_dead_ends), DEAD_END_NORM),
                revisit,
                current_boundary,
                current_dead_end,
            ],
            dtype=np.float32,
        )

        walkable_count = max(1, len(local_walkable))
        terrain_feat = np.array(
            [
                local_obstacles / 16.0,
                len(local_boundary_cells) / float(walkable_count),
                len(local_dead_end_cells) / float(walkable_count),
                immediate_escape_routes / 8.0,
            ],
            dtype=np.float32,
        )
        return map_feat, memory_feat, terrain_feat, clearance_feat, discovery_info

    def _detect_dead_end_corridors(self, observed_cells):
        walkable_cells = {cell for cell, walkable in observed_cells.items() if walkable > 0}
        if not walkable_cells:
            return set()

        xs = [cell[0] for cell in observed_cells]
        zs = [cell[1] for cell in observed_cells]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        def is_visible_edge(cell):
            x, z = cell
            return x in (min_x, max_x) or z in (min_z, max_z)

        neighbors = {}
        degree = {}
        for cell in walkable_cells:
            cx, cz = cell
            cell_neighbors = []
            for dx, dz in LOCAL_DIRS_4:
                neighbor = (cx + dx, cz + dz)
                if neighbor in walkable_cells:
                    cell_neighbors.append(neighbor)
            neighbors[cell] = cell_neighbors
            degree[cell] = len(cell_neighbors)

        leaves = [cell for cell, deg in degree.items() if deg <= 1 and not is_visible_edge(cell)]
        dead_end_cells = set()
        for leaf in leaves:
            prev_cell = None
            cur_cell = leaf
            corridor_cells = []
            reaches_visible_edge = False
            while True:
                if cur_cell in dead_end_cells:
                    break
                cur_degree = degree.get(cur_cell, 0)
                if cur_degree > 2:
                    break
                if is_visible_edge(cur_cell):
                    reaches_visible_edge = True
                    break
                corridor_cells.append(cur_cell)

                next_cells = [cell for cell in neighbors.get(cur_cell, []) if cell != prev_cell]
                if len(next_cells) != 1:
                    break
                next_cell = next_cells[0]
                if is_visible_edge(next_cell):
                    reaches_visible_edge = True
                    break
                if degree.get(next_cell, 0) >= 3:
                    break
                prev_cell, cur_cell = cur_cell, next_cell

            if not reaches_visible_edge:
                dead_end_cells.update(corridor_cells)

        return dead_end_cells

    def _build_spatial_map(self, frame_state, map_info, hero_x, hero_z):
        spatial = np.zeros((SPATIAL_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
        center = SPATIAL_RADIUS

        # Channel 0: local walkability / obstacles
        if map_info is not None and len(map_info) > 0:
            rows = len(map_info)
            cols = len(map_info[0])
            center_r = rows // 2
            center_c = cols // 2
            for local_r in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
                for local_c in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
                    rr = center_r + local_r
                    cc = center_c + local_c
                    sr = local_r + center
                    sc = local_c + center
                    if 0 <= rr < rows and 0 <= cc < cols:
                        spatial[0, sr, sc] = 1.0 if map_info[rr][cc] != 0 else 0.0

        # Channel 1: monsters
        for monster in frame_state.get("monsters", []) or []:
            if float(monster.get("is_in_view", 0.0)) <= 0.0:
                continue
            pos = monster.get("pos", {})
            self._mark_relative_entity(
                spatial[1],
                float(pos.get("x", hero_x)),
                float(pos.get("z", hero_z)),
                hero_x,
                hero_z,
                value=1.0,
            )

        # Channel 2: visible collectibles (treasure + buff)
        for organ in frame_state.get("organs", []) or []:
            if int(organ.get("sub_type", -1)) not in (1, 2):
                continue
            if int(organ.get("status", 1)) != 1:
                continue
            pos = organ.get("pos", {})
            self._mark_relative_entity(
                spatial[2],
                float(pos.get("x", hero_x)),
                float(pos.get("z", hero_z)),
                hero_x,
                hero_z,
                value=1.0,
            )

        # Channel 3: hero center + local visit heatmap + short movement trail
        hero_gx = int(round(hero_x))
        hero_gz = int(round(hero_z))
        for local_r in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
            for local_c in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
                gx = hero_gx + local_c
                gz = hero_gz + local_r
                visit_count = self.visit_counts.get((gx, gz), 0)
                if visit_count <= 0:
                    continue
                sr = local_r + center
                sc = local_c + center
                spatial[3, sr, sc] = min(
                    0.85,
                    float(np.log1p(visit_count) / np.log1p(VISIT_HEAT_SATURATION)),
                )
        if self.position_history:
            trail_len = len(self.position_history)
            for idx, (px, pz) in enumerate(self.position_history):
                weight = 0.35 + 0.45 * ((idx + 1) / max(1, trail_len))
                self._mark_relative_entity(spatial[3], px, pz, hero_x, hero_z, value=weight)
        spatial[3, center, center] = 1.0

        return spatial.reshape(-1).astype(np.float32)

    def _mark_relative_entity(self, channel, obj_x, obj_z, hero_x, hero_z, value):
        dx = int(round(obj_x)) - int(round(hero_x))
        dz = int(round(obj_z)) - int(round(hero_z))
        if abs(dx) > SPATIAL_RADIUS or abs(dz) > SPATIAL_RADIUS:
            return
        row = dz + SPATIAL_RADIUS
        col = dx + SPATIAL_RADIUS
        if 0 <= row < SPATIAL_SIZE and 0 <= col < SPATIAL_SIZE:
            channel[row, col] = max(channel[row, col], float(value))

    def _build_motion_context(self, hero_x, hero_z, last_action):
        move_dist = 0.0
        if self.position_history:
            prev_x, prev_z = self.position_history[-1]
            move_dist = _distance(hero_x, hero_z, prev_x, prev_z)

        recent_positions = self.position_history[-(POSITION_HISTORY - 1):] + [(hero_x, hero_z)]
        recent_moves = []
        for idx in range(1, len(recent_positions)):
            x1, z1 = recent_positions[idx - 1]
            x2, z2 = recent_positions[idx]
            recent_moves.append(_distance(x1, z1, x2, z2))

        avg_move = float(np.mean(recent_moves[-4:])) if recent_moves else 0.0
        still_tail = 0
        for dist in reversed(recent_moves):
            if dist < STUCK_DISTANCE:
                still_tail += 1
            else:
                break
        stuck_ratio = min(1.0, still_tail / 4.0)

        reverse_move = 0.0
        if (
            last_action is not None
            and last_action >= 0
            and last_action < 8
            and self.action_history
            and self.action_history[-1] < 8
            and int(last_action) == _reverse_dir(self.action_history[-1])
        ):
            reverse_move = 1.0

        oscillation = 0.0
        if len(self.position_history) >= 2:
            x_prev2, z_prev2 = self.position_history[-2]
            if _distance(hero_x, hero_z, x_prev2, z_prev2) < STUCK_DISTANCE and reverse_move > 0.5:
                oscillation = 1.0

        wall_bump = 0.0
        if last_action is not None and 0 <= last_action < 8 and move_dist < 0.10:
            wall_bump = 1.0

        motion_feat = np.array(
            [
                _norm(move_dist, 4.0),
                _norm(avg_move, 4.0),
                stuck_ratio,
                oscillation,
                reverse_move,
                wall_bump,
            ],
            dtype=np.float32,
        )
        motion_info = {
            "move_dist": move_dist,
            "avg_move": avg_move,
            "stuck_ratio": stuck_ratio,
            "oscillation": oscillation,
            "reverse_move": reverse_move,
            "wall_bump": wall_bump,
        }

        self.position_history.append((hero_x, hero_z))
        if len(self.position_history) > POSITION_HISTORY:
            self.position_history.pop(0)
        if last_action is not None and last_action >= 0:
            self.action_history.append(int(last_action))
            if len(self.action_history) > POSITION_HISTORY:
                self.action_history.pop(0)
        return motion_feat, motion_info

    def _shape_reward(
        self,
        phase,
        hero,
        cur_min_dist_norm,
        terrain_feat,
        memory_feat,
        clearance_feat,
        discovery_info,
        motion_info,
        last_action,
        hero_cell,
        env_info,
        buff_remain_norm,
        collectible_info,
        monsters_in_view_count,
    ):
        parts = {key: 0.0 for key in REWARD_COMPONENT_KEYS}

        def add(name, value):
            value = float(value)
            parts[name] += value
            return value

        prev_min_dist = self.last_min_monster_dist_norm
        dist_delta = cur_min_dist_norm - prev_min_dist
        treasure_count = int(hero.get("treasure_collected_count", 0))
        treasure_gain = max(0, treasure_count - self.last_treasure_count)

        collected_buff = int(env_info.get("collected_buff", self.last_collected_buff))
        buff_gain = max(0, collected_buff - self.last_collected_buff)
        buff_remain_delta = buff_remain_norm - self.last_buff_remain_norm
        if buff_gain <= 0 and buff_remain_delta > 0.65 and buff_remain_norm > 0.2:
            buff_gain = 1

        buff_active = buff_remain_norm > 0.05
        has_buff = bool(collectible_info.get("has_buff", False))
        buff_dist_norm = float(collectible_info.get("buff_dist_norm", 1.0))
        buff_dist_delta = self.last_buff_dist_norm - buff_dist_norm
        target_type = collectible_info.get("target_type", "none")
        target_angle = int(collectible_info.get("target_angle", -1))

        obstacle_density = float(terrain_feat[0])
        boundary_density = float(terrain_feat[1])
        dead_end_density = float(terrain_feat[2])
        escape_routes = float(terrain_feat[3])
        revisit_flag = float(memory_feat[5])
        visit_count = int(self.visit_counts.get(hero_cell, 1))
        current_boundary = float(memory_feat[6])
        current_dead_end = float(memory_feat[7])
        clearance_mean = float(np.mean(clearance_feat))
        exit_dir = int(np.argmax(clearance_feat))
        was_in_dead_end = self.last_in_dead_end
        escape_route_delta = escape_routes - self.last_escape_routes

        if was_in_dead_end and current_dead_end < 0.5:
            self.dead_end_escape_cnt += 1
            if last_action is not None and last_action >= 8:
                self.flash_escape_cnt += 1

        if current_dead_end > 0.5:
            self.dead_end_steps += 1
            self.dead_end_escape_routes_sum += escape_routes
            if last_action is not None and 0 <= int(last_action) < 8 and int(last_action) == exit_dir:
                self.dead_end_exit_move_cnt += 1

        reward = 0.0
        endgame = (float(self.step_no) / max(1.0, float(self.max_step))) > ENDGAME_RATIO
        midgame_pressure = MIDGAME_LOOP_START <= int(self.step_no) < MIDGAME_LOOP_END
        if phase == "early":
            reward += add("rw_phase_base", 0.015)
            reward += add("rw_scene", 0.08 * dist_delta)
            reward += add("rw_treasure", 1.2 * treasure_gain)
            reward += add("rw_discovery", 0.03 * discovery_info["new_visit"])
            reward += add("rw_discovery", 0.006 * discovery_info["new_walkable"])
            reward += add("rw_discovery", 0.010 * discovery_info["new_boundary"])
            reward += add("rw_discovery", 0.012 * discovery_info["new_dead_end"])
            if discovery_info["new_visit"] > 0 and cur_min_dist_norm > 0.12:
                reward += add("rw_discovery", 0.02)
            reward += add("rw_scene", 0.015 * escape_routes)
        elif phase == "mid":
            reward += add("rw_phase_base", 0.018)
            reward += add("rw_scene", 0.14 * dist_delta)
            reward += add("rw_treasure", 0.28 * treasure_gain)
            reward += add("rw_scene", 0.03 * obstacle_density)
            reward += add("rw_scene", 0.05 * boundary_density)
            reward += add("rw_scene", 0.06 * escape_routes)
            if (not buff_active or cur_min_dist_norm > 0.22):
                reward += add("rw_discovery", 0.010 * discovery_info["new_walkable"])
                reward += add("rw_discovery", 0.012 * discovery_info["new_boundary"])
                reward += add("rw_discovery", 0.014 * discovery_info["new_dead_end"])
            if 0.08 <= cur_min_dist_norm <= 0.22 and boundary_density > 0.15 and escape_routes > 0.25:
                reward += add("rw_scene", 0.04)
            if dead_end_density > 0.30 and escape_routes < 0.25 and cur_min_dist_norm < 0.20:
                reward += add("rw_dead_end", -0.10)
            if current_boundary > 0 and current_dead_end < 0.5:
                reward += add("rw_scene", 0.02)
        else:
            reward += add("rw_phase_base", 0.028)
            reward += add("rw_scene", 0.22 * dist_delta)
            reward += add("rw_scene", 0.08 * escape_routes)
            reward += add("rw_scene", 0.04 * boundary_density)
            reward += add("rw_scene", 0.02 * clearance_mean)
            if (not buff_active or cur_min_dist_norm > 0.22):
                reward += add("rw_discovery", 0.010 * discovery_info["new_walkable"])
                reward += add("rw_discovery", 0.012 * discovery_info["new_boundary"])
                reward += add("rw_discovery", 0.014 * discovery_info["new_dead_end"])
            if current_boundary > 0 and current_dead_end < 0.5:
                reward += add("rw_scene", 0.025)
            if current_dead_end > 0.5:
                reward += add("rw_dead_end", -0.14)
            if treasure_gain > 0 and cur_min_dist_norm > 0.32:
                reward += add("rw_treasure", 0.10 * treasure_gain)

        if revisit_flag > 0.5:
            reward += add("rw_revisit_pen", -0.004)
            if motion_info["stuck_ratio"] < 0.25 and cur_min_dist_norm > 0.22:
                reward += add("rw_revisit_pen", -0.004)
        if visit_count > LOCAL_LOOP_THRESHOLD and cur_min_dist_norm > 0.18:
            reward += add(
                "rw_revisit_pen",
                -0.003 * min(visit_count - LOCAL_LOOP_THRESHOLD, 6),
            )
        if midgame_pressure and revisit_flag > 0.5:
            reward += add("rw_revisit_pen", -0.012)
            if (
                motion_info["oscillation"] > 0.5
                or motion_info["reverse_move"] > 0.5
                or motion_info["wall_bump"] > 0.5
            ):
                reward += add("rw_revisit_pen", -0.018)
            if visit_count > LOCAL_LOOP_THRESHOLD:
                reward += add("rw_revisit_pen", -0.010)

        if buff_gain > 0:
            if phase == "early":
                reward += add("rw_buff_gain", 1.45 * buff_gain)
            elif phase == "mid":
                reward += add("rw_buff_gain", 1.75 * buff_gain)
            else:
                reward += add("rw_buff_gain", 2.00 * buff_gain)
            if cur_min_dist_norm < 0.20:
                reward += add("rw_buff_gain", 0.35 * buff_gain)
        elif not buff_active and has_buff:
            if phase == "early":
                reward += add("rw_buff_seek", 0.26 * buff_dist_delta)
            elif phase == "mid":
                reward += add("rw_buff_seek", 0.32 * buff_dist_delta)
            else:
                reward += add("rw_buff_seek", 0.24 * buff_dist_delta)

            if (
                target_type == "buff"
                and last_action is not None
                and 0 <= int(last_action) < 8
                and int(last_action) == target_angle
                and motion_info["move_dist"] > 0.20
            ):
                reward += add("rw_buff_seek", 0.028)

            if buff_dist_norm < 0.14 and buff_dist_delta < -0.02 and cur_min_dist_norm > 0.18:
                reward += add("rw_buff_seek", -0.05)
        elif buff_active:
            reward += add("rw_buff_active", 0.012)
            if cur_min_dist_norm < 0.20 and dist_delta > 0.0:
                reward += add("rw_buff_active", 0.10 * dist_delta)

        if endgame and discovery_info["new_visit"] > 0 and cur_min_dist_norm > 0.22:
            reward += add("rw_discovery", 0.025)
        if endgame and treasure_gain > 0:
            reward += add("rw_treasure", 0.20 * treasure_gain)
        if midgame_pressure and discovery_info["new_visit"] > 0 and cur_min_dist_norm > 0.18:
            reward += add("rw_discovery", 0.018)
        if midgame_pressure and current_boundary > 0.5 and escape_routes > 0.25 and cur_min_dist_norm > 0.18:
            reward += add("rw_escape", 0.028)

        if (
            target_type == "treasure"
            and target_angle >= 0
            and last_action is not None
            and 0 <= int(last_action) < 8
            and int(last_action) == target_angle
            and motion_info["move_dist"] > 0.18
            and cur_min_dist_norm > 0.18
        ):
            reward += add("rw_treasure", 0.028 if phase == "mid" else 0.020)

        if current_dead_end > 0.5 and last_action is not None and 0 <= int(last_action) < 8:
            if int(last_action) == exit_dir:
                reward += add("rw_dead_end", 0.04)

        if (
            last_action is not None
            and 0 <= int(last_action) < 8
            and cur_min_dist_norm < 0.20
            and escape_routes < 0.30
            and float(clearance_feat[int(last_action)]) <= 0.25
        ):
            reward += add("rw_dead_end", -0.04)

        if motion_info["move_dist"] > 0.25 and motion_info["wall_bump"] < 0.5:
            reward += add("rw_motion", 0.012)
        reward += add("rw_motion", -0.06 * motion_info["stuck_ratio"])
        reward += add("rw_motion", -0.05 * motion_info["oscillation"])
        reward += add("rw_motion", -0.04 * motion_info["reverse_move"])
        reward += add("rw_motion", -0.12 * motion_info["wall_bump"])

        escape_reward = 0.0
        if current_dead_end > 0.5:
            if escape_route_delta > 0.0:
                escape_reward += 0.18 * min(0.35, escape_route_delta)
            if motion_info["move_dist"] > 0.18 and motion_info["wall_bump"] < 0.5:
                escape_reward += 0.02
            if current_boundary > 0.5 and escape_routes > 0.25:
                escape_reward += 0.02
        elif was_in_dead_end:
            escape_reward += 0.22 if phase == "early" else 0.28
            if motion_info["move_dist"] > 0.18:
                escape_reward += 0.04
        reward += add("rw_escape", escape_reward)

        flash_reward = 0.0
        if last_action is not None and last_action >= 8:
            escaped_dead_end = was_in_dead_end and current_dead_end < 0.5
            opened_space = escape_route_delta > 0.12 and escape_routes > 0.25
            from_bump_to_open = self.last_wall_bump > 0.5 and escape_routes > 0.25
            if escaped_dead_end:
                flash_reward = 1.05 if phase != "late" else 1.20
            elif opened_space and from_bump_to_open:
                flash_reward = 0.65
            elif opened_space:
                flash_reward = 0.40
            elif prev_min_dist < 0.12 and dist_delta > 0.03:
                flash_reward = 0.9 if phase != "late" else 1.1
            elif prev_min_dist < 0.20 and dist_delta > 0.0:
                flash_reward = 0.35
            elif phase == "early" and prev_min_dist > 0.24:
                flash_reward = -0.15
            elif phase == "late" and dist_delta <= 0.0:
                flash_reward = -0.2
            else:
                flash_reward = -0.03
            if was_in_dead_end and escape_route_delta > 0.18:
                flash_reward += 0.08
            if motion_info["wall_bump"] > 0.5 or motion_info["move_dist"] < 0.18:
                flash_reward = min(flash_reward, -FLASH_WALL_BUMP_PENALTY)
            if monsters_in_view_count >= 2:
                if dist_delta > 0.05:
                    flash_reward += FLASH_PINCH_SUCCESS_BONUS
                elif dist_delta <= 0.0:
                    flash_reward = min(flash_reward, -FLASH_PINCH_FAIL_PENALTY)
            if current_dead_end > 0.5 and monsters_in_view_count == 0 and not escaped_dead_end and not opened_space:
                flash_reward = min(flash_reward, -FLASH_DEADEND_BAD_PENALTY)
        reward += add("rw_flash", flash_reward)

        self.last_reward_breakdown = {key: round(float(value), 6) for key, value in parts.items()}

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_treasure_count = treasure_count
        self.last_collected_buff = collected_buff
        self.last_buff_remain_norm = buff_remain_norm
        self.last_buff_dist_norm = buff_dist_norm
        self.last_in_dead_end = current_dead_end > 0.5
        self.last_escape_routes = escape_routes
        self.last_wall_bump = motion_info["wall_bump"]
        return float(reward)

    def get_reward_breakdown_snapshot(self):
        return dict(self.last_reward_breakdown)

    def _decay_memory_trackers(self):
        for store in (self.bump_cells, self.dir_block_history):
            expired = []
            for key in list(store.keys()):
                store[key] -= 1
                if store[key] <= 0:
                    expired.append(key)
            for key in expired:
                store.pop(key, None)
        expired_buffs = [cell for cell, cooldown in self.buff_cell_cooldown.items() if cooldown <= self.step_no]
        for cell in expired_buffs:
            self.buff_cell_cooldown.pop(cell, None)

    def _update_block_memory(self, hero_x, hero_z, last_action, motion_info):
        if last_action is None or not (0 <= int(last_action) < 8):
            return
        if float(motion_info.get("wall_bump", 0.0)) <= 0.5:
            return
        hero_cell = (int(round(hero_x)), int(round(hero_z)))
        action = int(last_action)
        self.bump_cells[hero_cell] = BLOCK_MEMORY_TTL
        self.dir_block_history[(hero_cell, action)] = BLOCK_MEMORY_TTL

    def get_debug_snapshot(self):
        dead_end_steps = max(0, int(self.dead_end_steps))
        step_base = max(1, int(self.step_no))
        return {
            "dead_end_steps": dead_end_steps,
            "dead_end_step_ratio": round(dead_end_steps / float(step_base), 4),
            "dead_end_escape_cnt": int(self.dead_end_escape_cnt),
            "dead_end_exit_dir_rate": round(
                float(self.dead_end_exit_move_cnt) / float(max(1, dead_end_steps)), 4
            ),
            "dead_end_avg_escape_routes": round(
                float(self.dead_end_escape_routes_sum) / float(max(1, dead_end_steps)), 4
            ),
            "flash_escape_cnt": int(self.flash_escape_cnt),
            "max_visit_count": max(self.visit_counts.values()) if self.visit_counts else 0,
            "blocked_dir_memory": len(self.dir_block_history),
            "remembered_buff_cells": len(self.buff_cell_memory),
            "memory_buff_target_steps": int(self.memory_buff_target_steps),
            "memory_buff_arrivals": int(self.memory_buff_arrivals),
            "memory_buff_cooldowns": len(self.buff_cell_cooldown),
        }
