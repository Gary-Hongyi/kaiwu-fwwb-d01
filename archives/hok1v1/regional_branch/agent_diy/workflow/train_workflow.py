#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import time
from agent_diy.feature.definition import sample_process, build_frame, FrameCollector, NONE_ACTION
from agent_diy.conf.conf import GameConfig
from agent_diy.workflow.lineup_heros import lineup_iterator_roundrobin_camp_heroes
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


BUTTON_NAMES = {
    1: "noop",
    2: "move",
    3: "attack",
    4: "skill",
    5: "skill",
    6: "skill",
    10: "skill",
}

PUSH_AXIS_HALF_RANGE = 62430.0
REWARD_MONITOR_KEYS = [
    "hp_point",
    "tower_hp_point",
    "money",
    "exp",
    "ep_rate",
    "death",
    "kill",
    "last_hit",
    "forward",
]


def _camp_to_zero(camp):
    if isinstance(camp, str):
        if camp.endswith("_1") or camp == "1":
            return 0
        if camp.endswith("_2") or camp == "2":
            return 1
        return -1
    if isinstance(camp, int):
        return camp - 1 if camp in (1, 2) else camp
    return -1


def _actor(obj):
    return obj.get("actor_state", obj)


def _main_hero_from_observation(obs):
    frame_state = obs.get("frame_state", {})
    player_id = obs.get("player_id")
    player_camp = _camp_to_zero(obs.get("player_camp", obs.get("camp", -1)))
    fallback = None
    for hero in frame_state.get("hero_states", []):
        hero_actor = _actor(hero)
        runtime_id = hero_actor.get("runtime_id")
        if hero.get("player_id") == player_id or runtime_id == player_id:
            return hero_actor, _camp_to_zero(hero_actor.get("camp", hero.get("camp", player_camp)))
        if _camp_to_zero(hero_actor.get("camp", hero.get("camp"))) == player_camp:
            fallback = hero_actor
    return fallback, player_camp


def _push_progress(obs):
    hero, camp = _main_hero_from_observation(obs)
    if not hero:
        return None
    location = hero.get("location") or {}
    x, z = location.get("x"), location.get("z")
    if x is None or z is None or x == 100000 or z == 100000:
        return None
    axis = float(x) + float(z)
    if camp == 0:
        progress = (axis + PUSH_AXIS_HALF_RANGE) / (2.0 * PUSH_AXIS_HALF_RANGE)
    else:
        progress = (-axis + PUSH_AXIS_HALF_RANGE) / (2.0 * PUSH_AXIS_HALF_RANGE)
    return max(0.0, min(1.0, progress))


class EpisodeMonitorStats:
    def __init__(self, monitor_side):
        self.monitor_side = monitor_side
        self.action_total = 0
        self.action_counts = {name: 0 for name in ["noop", "move", "attack", "skill", "other"]}
        self.reward_sums = {}
        self.progress_first = None
        self.progress_last = None
        self.progress_max = None
        self.force_move_mask_count_start = 0
        self.force_move_mask_count_end = 0

    def observe(self, obs):
        progress = _push_progress(obs)
        if progress is None:
            return
        if self.progress_first is None:
            self.progress_first = progress
            self.progress_max = progress
        self.progress_last = progress
        self.progress_max = max(self.progress_max, progress)

    def record_action(self, action):
        self.action_total += 1
        try:
            button = int(action[0])
        except (TypeError, IndexError, ValueError):
            button = -1
        name = BUTTON_NAMES.get(button, "other")
        self.action_counts[name] += 1

    def record_reward(self, reward):
        for key in REWARD_MONITOR_KEYS:
            weight_key = key + "_weight"
            origin_key = key + "_origin"
            if weight_key in reward:
                self.reward_sums[weight_key] = self.reward_sums.get(weight_key, 0.0) + reward[weight_key]
            if origin_key in reward:
                self.reward_sums[origin_key] = self.reward_sums.get(origin_key, 0.0) + reward[origin_key]

    def record_force_move_count(self, agent, is_start=False):
        count = getattr(agent, "rule_state", {}).get("force_move_mask_count", 0)
        if is_start:
            self.force_move_mask_count_start = count
        self.force_move_mask_count_end = count

    def summary(self, episode_cnt, frame_no, reward_sum, is_eval):
        total = max(self.action_total, 1)
        first_progress = self.progress_first if self.progress_first is not None else 0.0
        last_progress = self.progress_last if self.progress_last is not None else first_progress
        max_progress = self.progress_max if self.progress_max is not None else last_progress
        data = {
            "episode_cnt": episode_cnt,
            "reward": round(reward_sum, 4),
            "episode_frame": frame_no,
            "is_eval": int(is_eval),
            "progress": round(last_progress, 4),
            "progress_delta": round(last_progress - first_progress, 4),
            "max_progress": round(max_progress, 4),
            "noop_rate": round(self.action_counts["noop"] / total, 4),
            "move_rate": round(self.action_counts["move"] / total, 4),
            "attack_rate": round(self.action_counts["attack"] / total, 4),
            "skill_rate": round(self.action_counts["skill"] / total, 4),
            "other_action_rate": round(self.action_counts["other"] / total, 4),
            "force_move_mask_count": max(0, self.force_move_mask_count_end - self.force_move_mask_count_start),
        }
        for key, value in self.reward_sums.items():
            data[key] = round(value, 4)
        return data


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    do_learns = [True, True]
    last_save_model_time = time.time()
    env_conf_manager = EnvConfManager(config_path="agent_diy/conf/train_env_conf.toml", logger=logger)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes(GameConfig.CAMP_HEROES)
    episode_runner = EpisodeRunner(env=envs[0], agents=agents, logger=logger, monitor=monitor, env_conf_manager=env_conf_manager, lineup_iterator=lineup_iterator)
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


class EpisodeRunner:
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

    def _call_init_config(self, usr_conf):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                my_hero_ids, opponent_hero_ids, camp_key = blue_hero_ids, red_hero_ids, "blue_camp"
            else:
                my_hero_ids, opponent_hero_ids, camp_key = red_hero_ids, blue_hero_ids, "red_camp"
            config_data = {"my_camp": camp_key, "my_heroes": my_hero_ids, "opponent_heroes": opponent_hero_ids}
            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            if self.logger:
                self.logger.info(f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}")

    def run_episodes(self):
        while True:
            training_metrics = get_training_metrics()
            if training_metrics and self.logger:
                for key, value in training_metrics.items():
                    self.logger.info(f"training_metrics {key} is {value}")

            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(next(self.lineup_iterator))
            self._call_init_config(usr_conf)
            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break
            observation = env_obs["observation"]
            self.reset_agents(observation)
            frame_collector = FrameCollector(self.agent_num)
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            episode_monitor = EpisodeMonitorStats(monitor_side)
            episode_monitor.observe(observation[str(monitor_side)])
            episode_monitor.record_force_move_count(self.agents[monitor_side], is_start=True)
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            if self.logger:
                self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)].get("frame_state", {}))
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    if i == monitor_side:
                        episode_monitor.record_reward(reward)

            while True:
                actions = [NONE_ACTION] * self.agent_num
                for index, (do_predict, do_sample, agent) in enumerate(zip(self.do_predicts, self.do_samples, self.agents)):
                    if do_predict:
                        actions[index] = agent.exploit(observation[str(index)]) if is_eval else agent.predict(observation[str(index)])
                        if index == monitor_side:
                            episode_monitor.record_action(actions[index])
                            episode_monitor.record_force_move_count(agent)
                        if not is_eval and do_sample:
                            frame_collector.save_frame(build_frame(agent, observation[str(index)]), agent_id=index)

                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break
                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)].get("frame_state", {}))
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        if i == monitor_side:
                            episode_monitor.record_reward(reward)
                episode_monitor.observe(observation[str(monitor_side)])

                is_gameover = terminated or truncated or (is_train_test and frame_no >= GameConfig.debug_total_frames)
                if is_gameover:
                    if self.logger:
                        self.logger.info(f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}")
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(agent_id=i, reward=observation[str(i)].get("reward", {}).get("reward_sum", 0))
                    now = time.time()
                    if self.monitor and now - self.last_report_monitor_time >= 60:
                        monitor_data = episode_monitor.summary(
                            episode_cnt=self.episode_cnt,
                            frame_no=frame_no,
                            reward_sum=reward_sum_list[monitor_side],
                            is_eval=is_eval,
                        )
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now
                    if len(frame_collector) > 0 and not is_eval:
                        yield sample_process(frame_collector)
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
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
                    agent.load_model(id="latest")
                else:
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    if is_train_test:
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=opponent_agent)
                    self.do_samples[i] = False
            agent.reset(observation[str(i)])
