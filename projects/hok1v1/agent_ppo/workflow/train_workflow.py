#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
import os
import time
import random
import numpy as np
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FedExingFrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import Config, GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from agent_ppo.utils.frame_dump import FedExingFrameDumper


# Official baseline model ids (baseline5 strongest -> baseline3).
# Referenced directly via usr_conf opponent_agent; the platform serves them
# server-side during training. Do NOT put them in kaiwu.json model_pool — that
# is only for YOUR own uploaded models, and listing not-uploaded ids there makes
# the platform fail at startup.
# 官方 baseline 模型 id（baseline5 最强 -> baseline3）；通过 usr_conf 的
# opponent_agent 直接引用、平台服务端加载。不要写进 kaiwu.json 的 model_pool ——
# 那个只放你自己上传的模型，填没上传的 id 平台启动就会失败。
BASELINE_OPPONENT_TYPES = ["286236", "274015", "260247"]
# Weight of each baseline inside the training opponent mix (tilted toward stronger).
# 每个 baseline 在训练对手池中的权重（偏向更强的）。
BASELINE_OPPONENT_WEIGHTS = [0.25, 0.15, 0.10]
SELFPLAY_OPPONENT_WEIGHT = 0.40
COMMON_AI_OPPONENT_WEIGHT = 0.10
# NOTE: official baselines are NOT usable as TRAINING opponents. During training
# the platform only auto-drives common_ai; passing a baseline id opens a match but
# leaves the opponent hero idle (logs show kill/death/hurt_to_hero/hurt_by_hero
# all 0), which poisons training (the agent just pushes an empty lane). So the
# training pool is selfplay + common_ai only; use baselines for EVAL/ladder.
# Re-enable here ONLY once baseline weights can actually be driven locally
# (do_predict=True + a real load_opponent_agent loading the baseline weights).
# 注意：官方 baseline 不能当训练对手 —— 训练时平台只驱动 common_ai，填 baseline id
# 只会开一局但对手站着不动（日志 kill/death/伤害全 0），会污染训练。所以训练池只留
# selfplay + common_ai；baseline 用于评估/天梯。要在训练里用 baseline，必须能本地
# 真正加载并驱动其权重（do_predict=True + 实现 load_opponent_agent）。
TRAIN_OPPONENT_TYPES = ["selfplay", "common_ai"]
TRAIN_OPPONENT_WEIGHTS = [0.85, 0.15]
# Eval opponents: common_ai (sanity) + 289614 (compatible model, to track progress
# vs a fixed strong model). Needs 289614 in kaiwu.json model_pool + a working
# load_opponent_agent. 评估对手：common_ai + 289614（兼容模型，看相对强度进步）。
EVAL_OPPONENT_TYPES = ["common_ai", "289614"]
HISTORY_SELFPLAY_START_EPISODE = 50
HISTORY_SELFPLAY_RANDOM_PROB = 0.30


class FedExingEpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        # Cumulative count of how many episodes each opponent was used for, so the
        # training log can confirm baselines are actually being engaged.
        # 累计每种对手被使用的局数，方便在日志里确认 baseline 真的被打到了。
        self.opponent_usage = {}
        self.frame_dumper = FedExingFrameDumper(logger)

    def _fedexing_accumulate_reward_stats(self, stats, agent_idx, reward):
        stats["reward"][agent_idx] += reward.get("reward_sum", 0.0)
        for head_name, reward_names in GameConfig.REWARD_HEAD_GROUPS.items():
            stats[head_name][agent_idx] += sum(
                reward.get(f"{reward_name}_weight", 0.0) for reward_name in reward_names
            )
        for group_name, head_names in GameConfig.REWARD_MONITOR_GROUPS.items():
            stats[group_name][agent_idx] = sum(stats[head_name][agent_idx] for head_name in head_names)

    def _fedexing_build_reward_monitor_data(self, stats, monitor_side):
        metric_names = ("reward", *GameConfig.REWARD_MONITOR_GROUPS.keys())
        return {metric_name: round(stats[metric_name][monitor_side], 2) for metric_name in metric_names}

    def _fedexing_call_init_config(self, usr_conf):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def _fedexing_init_episode_reward_stats(self):
        stats = {"reward": [0.0] * self.agent_num}
        for head_name in GameConfig.REWARD_HEAD_GROUPS:
            stats[head_name] = [0.0] * self.agent_num
        for group_name in GameConfig.REWARD_MONITOR_GROUPS:
            stats[group_name] = [0.0] * self.agent_num
        return stats

    def _fedexing_init_legal_action_stats(self):
        return {
            "frame_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "density_sum": 0.0,
            "button_count_sum": 0.0,
            "target_count_sum": 0.0,
            "normal_attack_legal_sum": 0.0,
            "tower_target_legal_sum": 0.0,
        }

    def _fedexing_accumulate_legal_action_stats(self, stats, observation):
        stats["frame_count"] += 1
        raw_legal_action = np.asarray(observation.get("legal_action", []), dtype=np.float32).reshape(-1)
        expected_len = sum(Config.LABEL_SIZE_LIST[:-1]) + Config.LABEL_SIZE_LIST[0] * Config.LABEL_SIZE_LIST[-1]
        if raw_legal_action.size != expected_len:
            stats["invalid_count"] += 1
            return

        stats["valid_count"] += 1
        legal_mask = (raw_legal_action > 0.5).astype(np.float32)
        stats["density_sum"] += float(np.mean(legal_mask))

        button_size = Config.LABEL_SIZE_LIST[0]
        target_size = Config.LABEL_SIZE_LIST[-1]
        target_offset = sum(Config.LABEL_SIZE_LIST[:-1])
        button_mask = legal_mask[:button_size]
        target_mask = legal_mask[target_offset:].reshape(button_size, target_size)

        stats["button_count_sum"] += float(np.sum(button_mask))
        stats["target_count_sum"] += float(np.sum(target_mask))
        if button_size > 3:
            stats["normal_attack_legal_sum"] += float(button_mask[3] > 0)
        if button_size > 3 and target_size > 7:
            stats["tower_target_legal_sum"] += float(button_mask[3] > 0 and target_mask[3][7] > 0)

    def _fedexing_build_legal_action_monitor_data(self, stats):
        frame_count = max(stats["frame_count"], 1)
        valid_count = max(stats["valid_count"], 1)
        return {
            "raw_legal_action_valid_ratio": round(stats["valid_count"] / frame_count, 4),
            "raw_legal_action_density": round(stats["density_sum"] / valid_count, 4),
            "raw_legal_button_count": round(stats["button_count_sum"] / valid_count, 3),
            "raw_legal_target_count": round(stats["target_count_sum"] / valid_count, 3),
            "raw_normal_attack_legal_ratio": round(stats["normal_attack_legal_sum"] / valid_count, 4),
            "raw_tower_target_legal_ratio": round(stats["tower_target_legal_sum"] / valid_count, 4),
        }

    @staticmethod
    def _fedexing_is_builtin_opponent(opponent_agent):
        return str(opponent_agent) in ("common_ai", "selfplay")

    def _fedexing_valid_model_pool(self):
        try:
            return get_valid_model_pool(self.logger)
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"get_valid_model_pool failed, fallback to empty model pool: {exc}")
            return []

    def _fedexing_sanitize_opponent_agent(self, opponent_agent):
        opponent_agent = str(opponent_agent)
        if self._fedexing_is_builtin_opponent(opponent_agent):
            return opponent_agent
        if opponent_agent in BASELINE_OPPONENT_TYPES:
            # Platform-served official baseline; it is not in the local self-play
            # model pool, so skip the pool check and pass it through.
            # 平台服务端的官方 baseline，不在本地自对弈模型池，跳过校验直接放行。
            return opponent_agent
        try:
            opponent_model_id = int(opponent_agent)
        except (TypeError, ValueError):
            if self.logger:
                self.logger.warning(f"invalid opponent_agent={opponent_agent}, fallback to common_ai")
            return "common_ai"

        valid_model_pool = self._fedexing_valid_model_pool()
        if opponent_model_id in valid_model_pool:
            return opponent_agent

        if self.logger:
            self.logger.warning(
                f"opponent_agent model_id {opponent_agent} not in {valid_model_pool}, fallback to common_ai"
            )
        return "common_ai"

    def _fedexing_choose_eval_opponent(self, usr_conf):
        opponent = self._fedexing_choose_weighted_opponent(EVAL_OPPONENT_TYPES)
        return opponent or "common_ai"

    def _fedexing_choose_train_opponent(self, usr_conf):
        opponent = self._fedexing_choose_weighted_opponent(TRAIN_OPPONENT_TYPES, TRAIN_OPPONENT_WEIGHTS)
        return opponent or self._fedexing_sanitize_opponent_agent(usr_conf["episode"].get("opponent_agent", "selfplay"))

    def _fedexing_choose_weighted_opponent(self, opponents, weights=None):
        if not isinstance(opponents, list) or not opponents:
            return None
        candidates = []
        candidate_weights = []
        for index, opponent in enumerate(opponents):
            opponent = self._fedexing_sanitize_opponent_agent(opponent)
            if not opponent:
                continue
            weight = 1.0
            if isinstance(weights, list) and index < len(weights):
                try:
                    weight = float(weights[index])
                except (TypeError, ValueError):
                    weight = 0.0
            if weight <= 0:
                continue
            candidates.append(opponent)
            candidate_weights.append(weight)
        if not candidates:
            return None
        return random.choices(candidates, weights=candidate_weights, k=1)[0]

    def _fedexing_choose_selfplay_model_id(self):
        if self.episode_cnt < HISTORY_SELFPLAY_START_EPISODE:
            return "latest"
        if random.random() < HISTORY_SELFPLAY_RANDOM_PROB:
            return "random"
        return "latest"

    def _fedexing_load_selfplay_opponent(self, agent):
        model_id = self._fedexing_choose_selfplay_model_id()
        try:
            agent.load_model(id=model_id)
            if self.logger:
                self.logger.info(f"selfplay opponent loaded model_id={model_id}")
        except Exception as exc:
            if model_id == "latest":
                raise
            if self.logger:
                self.logger.warning(f"load random selfplay opponent failed: {exc}, fallback to latest")
            agent.load_model(id="latest")

    def reset_agents(self, observation, opponent_agent=None):
        if opponent_agent is None:
            opponent_agent = self.env_conf_manager.get_opponent_agent()
        opponent_agent = self._fedexing_sanitize_opponent_agent(opponent_agent)
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        for i, agent in enumerate(self.agents):
            if i == monitor_side:
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    self._fedexing_load_selfplay_opponent(agent)
                elif opponent_agent in BASELINE_OPPONENT_TYPES:
                    # Platform serves the baseline opponent (set via usr_conf
                    # opponent_agent), same as common_ai: this side does not
                    # predict or produce training samples locally.
                    # baseline 由平台服务端加载（通过 usr_conf 的 opponent_agent），
                    # 与 common_ai 一样：本地这侧不预测、不采样。
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                    if self.logger:
                        self.logger.info(
                            f"[opponent] baseline model_id={opponent_agent} engaged as opponent "
                            f"(platform-served), agent_idx={i}, monitor_side={monitor_side}"
                        )
                else:
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    eval_candidate_model = self._fedexing_valid_model_pool()
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            agent.reset(observation[str(i)])

    def run_episodes(self):
        while True:
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            if is_eval:
                opponent_agent = self._fedexing_choose_eval_opponent(usr_conf)
                usr_conf["episode"]["eval_opponent_type"] = opponent_agent
                usr_conf["episode"]["opponent_agent"] = opponent_agent
                self.logger.info(f"eval opponent selected: {opponent_agent}")
            else:
                opponent_agent = self._fedexing_choose_train_opponent(usr_conf)
                usr_conf["episode"]["opponent_agent"] = opponent_agent
                self.opponent_usage[opponent_agent] = self.opponent_usage.get(opponent_agent, 0) + 1
                self.logger.info(
                    f"train opponent selected: {opponent_agent} "
                    f"(is_baseline={opponent_agent in BASELINE_OPPONENT_TYPES}); "
                    f"opponent_usage so far: {self.opponent_usage}"
                )

            self._fedexing_call_init_config(usr_conf)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break
            self.frame_dumper.dump("reset", self.episode_cnt + 1, env_obs=env_obs)

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            self.reset_agents(observation, opponent_agent)

            frame_collector = FedExingFrameCollector(self.agent_num)

            self.episode_cnt += 1
            frame_no = 0
            reward_stats = self._fedexing_init_episode_reward_stats()
            legal_action_stats = self._fedexing_init_legal_action_stats()
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    self._fedexing_accumulate_reward_stats(reward_stats, i, reward)

            while True:
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if index == monitor_side:
                            self._fedexing_accumulate_legal_action_stats(
                                legal_action_stats,
                                observation[str(index)],
                            )
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break
                self.frame_dumper.dump(
                    "step",
                    self.episode_cnt,
                    env_obs=env_obs,
                    actions=actions,
                    env_reward=env_reward,
                )

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        self._fedexing_accumulate_reward_stats(reward_stats, i, reward)

                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000) or self.frame_dumper.should_stop()
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_stats['reward'][monitor_side]}"
                    )
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            monitor_data.update(self._fedexing_build_legal_action_monitor_data(legal_action_stats))
                            if is_eval:
                                monitor_data.update(self._fedexing_build_reward_monitor_data(reward_stats, monitor_side))
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    do_learns = [True, True]
    last_save_model_time = time.time()

    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133])

    episode_runner = FedExingEpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now
