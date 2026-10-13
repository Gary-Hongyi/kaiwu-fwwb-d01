#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Gorge Chase PPO.
"""

import copy
from collections import defaultdict
import os
import random
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

MODEL_SAVE_INTERVAL_SECONDS = 300
ENV_REWARD_WEIGHT = 0.25
TERMINATED_PENALTY = -4.0
TRUNCATED_BONUS = 4.0
HARD_MAP_WARMUP_EPISODES = 200
SHORT_DEATH_STEPS = 120
LONG_GAME_STEPS = 800
MAP_LOG_INTERVAL_EPISODES = 50


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= MODEL_SAVE_INTERVAL_SECONDS:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.map_pool = self._extract_map_pool(usr_conf)
        self.map_stats = {map_id: self._new_map_stat() for map_id in self.map_pool}

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            episode_conf, selected_map = self._build_episode_conf()
            env_obs = self.env.reset(episode_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            actual_map = self._extract_map_id(env_obs, selected_map)
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            reward_component_sums = defaultdict(float)

            self.logger.info(
                f"Episode {self.episode_cnt} start map:{actual_map} "
                f"sample_weights:{self._format_map_weights()}"
            )

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                shaped_reward = float(_remain_info.get("reward", [0.0])[0])
                reward_breakdown = self.agent.preprocessor.get_reward_breakdown_snapshot()
                for key, value in reward_breakdown.items():
                    reward_component_sums[key] += float(value)
                reward = np.array(
                    [shaped_reward + ENV_REWARD_WEIGHT * self._to_scalar_reward(env_reward)],
                    dtype=np.float32,
                )
                total_reward += float(reward[0])

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    collected_buff = int(env_info.get("collected_buff", 0))
                    total_buff = int(env_info.get("total_buff", 0))
                    buff_pick_rate = collected_buff / max(1, total_buff)

                    if terminated:
                        final_reward[0] = TERMINATED_PENALTY
                        result_str = "FAIL"
                    else:
                        final_reward[0] = TRUNCATED_BONUS
                        result_str = "WIN"

                    debug_info = self.agent.preprocessor.get_debug_snapshot()
                    self._record_map_result(actual_map, step, total_score, result_str)

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"map:{actual_map} result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f} "
                        f"buff:{collected_buff}/{total_buff} rate:{buff_pick_rate:.2f} "
                        f"dead_end_steps:{debug_info['dead_end_steps']} "
                        f"dead_end_ratio:{debug_info['dead_end_step_ratio']:.2f} "
                        f"dead_end_escape:{debug_info['dead_end_escape_cnt']} "
                        f"exit_dir_rate:{debug_info['dead_end_exit_dir_rate']:.2f} "
                        f"avg_escape_routes:{debug_info['dead_end_avg_escape_routes']:.2f} "
                        f"flash_escape:{debug_info['flash_escape_cnt']} "
                        f"buff_memory:{debug_info['remembered_buff_cells']} "
                        f"buff_memory_target:{debug_info['memory_buff_target_steps']} "
                        f"buff_memory_arrival:{debug_info['memory_buff_arrivals']} "
                        f"buff_memory_cd:{debug_info['memory_buff_cooldowns']} "
                        f"reward_decomp:{self._format_reward_components(reward_component_sums)}"
                    )
                    if self.episode_cnt % MAP_LOG_INTERVAL_EPISODES == 0:
                        self.logger.info(self._format_hard_map_snapshot())

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "map_id": actual_map,
                            "collected_buff": collected_buff,
                            "buff_pick_rate": round(buff_pick_rate, 4),
                            "dead_end_steps": debug_info["dead_end_steps"],
                            "dead_end_step_ratio": debug_info["dead_end_step_ratio"],
                            "dead_end_escape_cnt": debug_info["dead_end_escape_cnt"],
                            "dead_end_exit_dir_rate": debug_info["dead_end_exit_dir_rate"],
                            "dead_end_avg_escape_routes": debug_info["dead_end_avg_escape_routes"],
                            "flash_escape_cnt": debug_info["flash_escape_cnt"],
                            "remembered_buff_cells": debug_info["remembered_buff_cells"],
                            "memory_buff_target_steps": debug_info["memory_buff_target_steps"],
                            "memory_buff_arrivals": debug_info["memory_buff_arrivals"],
                            "memory_buff_cooldowns": debug_info["memory_buff_cooldowns"],
                        }
                        for key, value in reward_component_sums.items():
                            monitor_data[f"{key}_sum"] = round(float(value), 4)
                        map_stat = self.map_stats.get(actual_map)
                        if map_stat:
                            episodes = max(1, map_stat["episodes"])
                            monitor_data["map_short_death_rate"] = round(
                                map_stat["short_deaths"] / episodes, 4
                            )
                            monitor_data["map_avg_steps"] = round(map_stat["avg_steps"], 2)
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info

    def _to_scalar_reward(self, env_reward):
        if env_reward is None:
            return 0.0
        if isinstance(env_reward, np.ndarray):
            return float(env_reward.reshape(-1)[0]) if env_reward.size > 0 else 0.0
        if isinstance(env_reward, (list, tuple)):
            return float(env_reward[0]) if env_reward else 0.0
        try:
            return float(env_reward)
        except (TypeError, ValueError):
            return 0.0

    def _extract_map_pool(self, usr_conf):
        env_conf = usr_conf.get("env_conf", {}) if isinstance(usr_conf, dict) else {}
        map_pool = env_conf.get("map", [1])
        if isinstance(map_pool, int):
            return [int(map_pool)]
        if isinstance(map_pool, (list, tuple)):
            cleaned = []
            for map_id in map_pool:
                try:
                    cleaned.append(int(map_id))
                except (TypeError, ValueError):
                    continue
            return cleaned or [1]
        return [1]

    def _new_map_stat(self):
        return {
            "episodes": 0,
            "short_deaths": 0,
            "long_games": 0,
            "wins": 0,
            "avg_steps": 0.0,
            "avg_score": 0.0,
        }

    def _format_reward_components(self, reward_component_sums):
        if not reward_component_sums:
            return "none"
        ranked = sorted(
            reward_component_sums.items(), key=lambda item: abs(float(item[1])), reverse=True
        )
        top_items = ranked[:4]
        return ",".join(f"{key}:{float(value):.3f}" for key, value in top_items)

    def _build_episode_conf(self):
        episode_conf = copy.deepcopy(self.usr_conf)
        selected_map = self._select_map()
        env_conf = episode_conf.setdefault("env_conf", {})
        env_conf["map"] = [selected_map]
        env_conf["map_random"] = False
        return episode_conf, selected_map

    def _select_map(self):
        if len(self.map_pool) == 1:
            return self.map_pool[0]
        if self.episode_cnt < HARD_MAP_WARMUP_EPISODES:
            return random.choice(self.map_pool)

        weighted_maps = []
        weights = []
        for map_id in self.map_pool:
            stat = self.map_stats.setdefault(map_id, self._new_map_stat())
            episodes = stat["episodes"]
            short_death_rate = stat["short_deaths"] / max(1, episodes)
            long_game_rate = stat["long_games"] / max(1, episodes)
            win_rate = stat["wins"] / max(1, episodes)
            avg_steps = stat["avg_steps"]

            exploration_boost = 1.25 if episodes < 30 else 1.0
            low_step_pressure = max(0.0, (350.0 - avg_steps) / 350.0)
            weight = (
                1.0
                + 3.0 * short_death_rate
                + 1.5 * low_step_pressure
                + 0.75 * (1.0 - long_game_rate)
                + 0.5 * (1.0 - win_rate)
            ) * exploration_boost
            weighted_maps.append(map_id)
            weights.append(max(0.1, weight))

        return random.choices(weighted_maps, weights=weights, k=1)[0]

    def _extract_map_id(self, env_obs, fallback_map):
        env_info = (
            env_obs.get("observation", {}).get("env_info", {})
            if isinstance(env_obs, dict)
            else {}
        )
        for key in ("map_id", "map_idx", "map_index", "map_no", "map"):
            value = env_info.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, (list, tuple)) and value:
                try:
                    return int(value[0])
                except (TypeError, ValueError):
                    continue
        return int(fallback_map)

    def _record_map_result(self, map_id, step, total_score, result_str):
        stat = self.map_stats.setdefault(map_id, self._new_map_stat())
        stat["episodes"] += 1
        episodes = stat["episodes"]
        stat["avg_steps"] += (float(step) - stat["avg_steps"]) / episodes
        stat["avg_score"] += (float(total_score) - stat["avg_score"]) / episodes
        if result_str == "WIN":
            stat["wins"] += 1
        if step < SHORT_DEATH_STEPS:
            stat["short_deaths"] += 1
        if step >= LONG_GAME_STEPS:
            stat["long_games"] += 1

    def _format_map_weights(self):
        if len(self.map_pool) <= 1:
            return "fixed"
        if self.episode_cnt < HARD_MAP_WARMUP_EPISODES:
            return "warmup_uniform"

        parts = []
        for map_id in self.map_pool:
            stat = self.map_stats.get(map_id, self._new_map_stat())
            episodes = max(1, stat["episodes"])
            short_rate = stat["short_deaths"] / episodes
            avg_steps = stat["avg_steps"]
            parts.append(f"{map_id}:sd={short_rate:.2f},avg={avg_steps:.0f}")
        return " | ".join(parts)

    def _format_hard_map_snapshot(self):
        ranked = []
        for map_id, stat in self.map_stats.items():
            if stat["episodes"] == 0:
                continue
            episodes = stat["episodes"]
            short_rate = stat["short_deaths"] / episodes
            ranked.append(
                (
                    short_rate,
                    -stat["avg_steps"],
                    map_id,
                    episodes,
                    stat["avg_steps"],
                    stat["avg_score"],
                    stat["wins"],
                )
            )
        if not ranked:
            return "[MAP_STATS] no map stats yet"

        ranked.sort(reverse=True)
        top_items = []
        for short_rate, _, map_id, episodes, avg_steps, avg_score, wins in ranked[:3]:
            win_rate = wins / max(1, episodes)
            top_items.append(
                f"map:{map_id} ep:{episodes} short:{short_rate:.2f} "
                f"avg_steps:{avg_steps:.1f} avg_score:{avg_score:.1f} win:{win_rate:.2f}"
            )
        return "[MAP_STATS] hardest -> " + " ; ".join(top_items)
