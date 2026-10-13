#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Gorge Chase DIY agent.
"""


import os
import time

from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    last_save_model_time = time.time()
    episode_cnt = 0
    last_monitor_time = 0
    last_metrics_time = 0

    while True:
        if time.time() - last_metrics_time >= 60:
            training_metrics = get_training_metrics()
            if training_metrics is not None and logger:
                logger.info(f"training_metrics is {training_metrics}")
            last_metrics_time = time.time()

        env_obs = env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, logger):
            continue

        agent.reset(env_obs)
        obs_data, remain_info = agent.observation_process(env_obs)

        episode_cnt += 1
        step = 0
        total_reward = 0.0
        done = False

        while not done:
            act_data = agent.predict([obs_data])[0]
            act = agent.action_process(act_data)
            env_reward, next_env_obs = env.step(act)

            if handle_disaster_recovery(next_env_obs, logger):
                break

            terminated = next_env_obs["terminated"]
            truncated = next_env_obs["truncated"]
            done = terminated or truncated
            step += 1

            next_obs_data, next_remain_info = agent.observation_process(next_env_obs)
            score = float(env_reward.get("reward", 0.0)) if isinstance(env_reward, dict) else 0.0
            total_reward = score

            env_obs = next_env_obs
            obs_data = next_obs_data
            remain_info = next_remain_info

            if done:
                env_info = next_env_obs["observation"]["env_info"]
                total_score = float(env_info.get("total_score", 0.0))
                result_str = "FAIL" if terminated else "WIN"

                now = time.time()
                if monitor and now - last_monitor_time >= 60:
                    monitor.put_data(
                        {
                            os.getpid(): {
                                "reward": round(total_reward, 4),
                                "episode_steps": step,
                                "episode_cnt": episode_cnt,
                            }
                        }
                    )
                    last_monitor_time = now

                if logger:
                    logger.info(
                        f"[DIY GAMEOVER] episode:{episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} total_reward:{total_reward:.3f}"
                    )

        now = time.time()
        if now - last_save_model_time >= 1800:
            agent.save_model()
            last_save_model_time = now
