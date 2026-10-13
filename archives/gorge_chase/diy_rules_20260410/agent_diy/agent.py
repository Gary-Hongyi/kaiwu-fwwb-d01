#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
DIY rule-based agent for Gorge Chase.
"""


import os

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.conf.prompt import MONSTER_CHASE_SYSTEM_PROMPT, build_monster_chase_prompt
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.model.model import Model
from agent_diy.rule_policy import RulePolicy
from agent_ppo.feature.preprocessor import Preprocessor

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device or "cpu"
        self.model = Model(Config.FEATURE_VECTOR_SHAPE).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.START_LR)
        self.algorithm = Algorithm(self.model, self.optimizer, device=self.device, logger=logger, monitor=monitor)
        self.preprocessor = Preprocessor()
        self.rule_policy = RulePolicy()
        self.logger = logger
        self.monitor = monitor
        self.last_action = -1
        self.last_positions = []
        self.strategy_prompt = MONSTER_CHASE_SYSTEM_PROMPT
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.last_action = -1
        self.last_positions = []
        self.preprocessor.reset()

    def build_decision_prompt(self, state_summary=None):
        return build_monster_chase_prompt(state_summary)

    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        summary = getattr(obs_data, "state_summary", None) or {}
        decision = self.rule_policy.decide(summary, list(obs_data.legal_action), prefer_greedy=True)

        return [
            ActData(
                action=[decision.action],
                d_action=[decision.greedy_action],
                prob=decision.prob,
                value=[0.0],
                prompt=self.build_decision_prompt(summary),
            )
        ]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, f"model.ckpt-{str(id)}.pkl")
        torch.save(self.model.state_dict(), model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if not path:
            return
        model_file_path = os.path.join(path, f"model.ckpt-{str(id)}.pkl")
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.info(f"model file {model_file_path} not found, skip loading")
            return
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, env_obs):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)

        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]

        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        hero_x = float(hero_pos.get("x", 0.0))
        hero_z = float(hero_pos.get("z", 0.0))
        flash_cd = float(hero.get("flash_cooldown", 0.0))

        self._update_positions(hero_x, hero_z)
        monsters = self._normalize_entities(frame_state.get("monsters", []))
        treasures = self._normalize_entities(frame_state.get("treasures", frame_state.get("treasure", [])))
        buffs = self._normalize_entities(frame_state.get("buffs", frame_state.get("buff", [])))
        nearest_monsters = self._nearest_entities(hero_x, hero_z, monsters, limit=2)
        while len(nearest_monsters) < 2:
            nearest_monsters.append(self._empty_entity())
        nearest_treasure = self._nearest_entities(hero_x, hero_z, treasures, limit=1)[0] if treasures else self._empty_entity()
        nearest_buff = self._nearest_entities(hero_x, hero_z, buffs, limit=1)[0] if buffs else self._empty_entity()

        total_score = float(env_info.get("total_score", 0.0))
        progress = float(observation.get("step_no", 0.0)) / max(1.0, float(env_info.get("max_step", 1000.0)))

        state_summary = {
            "score": total_score,
            "progress": progress,
            "flash_ready": float(flash_cd <= 0.0) > 0,
            "hero": {
                "x": hero_x,
                "z": hero_z,
                "escape_bias_dx": self._escape_bias(nearest_monsters)[0],
                "escape_bias_dz": self._escape_bias(nearest_monsters)[1],
            },
            "monsters": nearest_monsters,
            "treasure": nearest_treasure,
            "buff": nearest_buff,
            "clearance": feature[58:66],
            "trap": feature[66:70],
            "stuck": feature[70:74],
            "safe_treasure": feature[74:78],
            "legal_actions": [i for i, flag in enumerate(legal_action) if flag],
        }

        obs_data = ObsData(feature=list(feature), legal_action=legal_action, state_summary=state_summary)
        remain_info = {
            "reward": reward,
            "nearest_monster_dist": float(nearest_monsters[0]["dist_norm"]),
            "nearest_treasure_dist": float(nearest_treasure["dist_norm"]),
            "score": total_score,
            "step_no": float(observation.get("step_no", 0.0)),
        }
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _update_positions(self, hero_x, hero_z):
        self.last_positions.append((hero_x, hero_z))
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)

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
                    "dx_norm": np.clip(dx / Config.MAP_SIZE, -1.0, 1.0),
                    "dz_norm": np.clip(dz / Config.MAP_SIZE, -1.0, 1.0),
                    "speed_norm": self._norm(entity["speed"], Config.MAX_MONSTER_SPEED),
                    "dist_norm": self._norm(dist, Config.MAP_SIZE * 1.42),
                }
            )
        scored.sort(key=lambda item: item["dist_norm"])
        return scored[:limit]

    def _empty_entity(self):
        return {"visible": 0.0, "dx_norm": 0.0, "dz_norm": 0.0, "speed_norm": 0.0, "dist_norm": 1.0}

    def _escape_bias(self, monsters):
        dx = 0.0
        dz = 0.0
        for idx, monster in enumerate(monsters[:2]):
            if monster["visible"] <= 0:
                continue
            weight = 1.0 / max(0.05, float(monster["dist_norm"]))
            if idx == 1:
                weight *= 0.7
            dx -= monster["dx_norm"] * weight
            dz -= monster["dz_norm"] * weight
        return dx, dz

    def _norm(self, value, max_value, min_value=0.0):
        value = float(np.clip(value, min_value, max_value))
        return (value - min_value) / (max_value - min_value) if max_value - min_value > 1e-6 else 0.0
