#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
try:
    from kaiwudrl.interface.agent import BaseAgent
except ModuleNotFoundError:
    class BaseAgent:
        def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
            self.agent_type = agent_type
            self.device = device
            self.logger = logger
            self.monitor = monitor

        def send_sample_data(self, sample_data):
            return self.learn(sample_data)
from agent_diy.model.model import Model
from agent_diy.feature.definition import *
from agent_diy.conf.conf import Config
from agent_diy.feature.reward_manager import GameRewardManager
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.unpack_state_dict import Info
from agent_diy.feature.obs_builder import ObsBuilder

FLASH_SKILL_ID = 80115

BUTTON_NOOP = 1
BUTTON_MOVE = 2
OPENING_FORCE_MOVE_FRAMES = 1200
OPENING_PROGRESS_DONE = 0.22
NOOP_STUCK_THRESHOLD = 5
PUSH_AXIS_HALF_RANGE = 62430.0
FALLBACK_ENEMY_TOWER_POS = np.array([31310.0, 31120.0], dtype=np.float32)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.info = Info()
        self.obs_builder = ObsBuilder()
        self.rule_state = {"consecutive_noop": 0, "last_button": None, "force_move_mask_count": 0}
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)

    def init_config(self, config_data):
        return {hero_id: FLASH_SKILL_ID for hero_id in config_data.get("my_heroes", [])}

    def reset(self, observation):
        self.hero_camp = observation.get("player_camp", observation.get("camp", 1))
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.info.reset()
        self.obs_builder.reset()
        self.rule_state = {"consecutive_noop": 0, "last_button": None, "force_move_mask_count": 0}

    def _model_inference(self, list_obs_data):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]
        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            torch_inputs[i] = data.reshape(-1).float()
        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)
        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model([feature_vec, lstm_hidden_state, lstm_cell_state], inference=True)
        np_output = [output.cpu().numpy() for output in output_list]
        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]
        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)
        list_act_data = []
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(ActData(action=action, d_action=d_action, prob=prob, d_prob=d_prob, value=value, lstm_cell=_lstm_cell[i], lstm_hidden=_lstm_hidden[i]))
        return list_act_data

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def observation_process(self, observation):
        self.info.update(observation)
        feature = self.obs_builder.build_observation(self.info)
        legal_action = self._apply_action_masks(observation)
        return ObsData(feature=feature, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden)

    def action_process(self, observation, act_data, is_stochastic):
        return act_data.action if is_stochastic else act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden
        self._update_rule_state(act_data.action)

    def _update_rule_state(self, action):
        try:
            button = int(action[0])
        except (TypeError, ValueError, IndexError):
            return
        self.rule_state["last_button"] = button
        if button == BUTTON_NOOP:
            self.rule_state["consecutive_noop"] = self.rule_state.get("consecutive_noop", 0) + 1
        else:
            self.rule_state["consecutive_noop"] = 0

    def _apply_action_masks(self, observation):
        legal_action = observation.get("legal_action", [])
        legal = np.asarray(legal_action, dtype=np.float32).copy()
        if legal.size != int(np.sum(self.legal_action_size)):
            return legal_action

        force_opening_move = self._should_force_opening_move(observation)
        stuck_in_noop = self.rule_state.get("consecutive_noop", 0) >= NOOP_STUCK_THRESHOLD

        if force_opening_move and self._force_move_mask(legal):
            self.rule_state["force_move_mask_count"] = self.rule_state.get("force_move_mask_count", 0) + 1
            return legal.tolist()
        if stuck_in_noop and BUTTON_NOOP < self.label_size_list[0]:
            legal[BUTTON_NOOP] = 0.0

        if np.sum(legal[: self.label_size_list[0]]) <= 0:
            return legal_action
        return legal.tolist()

    def _should_force_opening_move(self, observation):
        frame_state = observation.get("frame_state", {})
        frame_no = frame_state.get("frame_no", frame_state.get("frameNo", 0))
        if frame_no > OPENING_FORCE_MOVE_FRAMES:
            return False
        return self._current_push_progress() < OPENING_PROGRESS_DONE

    def _current_push_progress(self):
        try:
            pos = self.info.hero_our.info.position
        except AttributeError:
            return 1.0
        if not self._valid_position(pos):
            return 1.0
        axis = float(pos[0]) + float(pos[1])
        progress = (axis + PUSH_AXIS_HALF_RANGE) / (2.0 * PUSH_AXIS_HALF_RANGE)
        return float(np.clip(progress, 0.0, 1.0))

    def _force_move_mask(self, legal):
        if BUTTON_MOVE >= self.label_size_list[0] or legal[BUTTON_MOVE] <= 0:
            return False

        legal[: self.label_size_list[0]] = 0.0
        legal[BUTTON_MOVE] = 1.0

        move_x, move_z = self._opening_move_indices()
        move_x_offset = self.label_size_list[0]
        move_z_offset = move_x_offset + self.label_size_list[1]
        self._force_head_index(legal, move_x_offset, self.label_size_list[1], move_x)
        self._force_head_index(legal, move_z_offset, self.label_size_list[2], move_z)
        return True

    def _opening_move_indices(self):
        try:
            center = np.asarray(self.info.hero_our.info.position, dtype=np.float32)
            enemy_tower_pos = np.asarray(self.info.organ_enemy.sub_tower.position, dtype=np.float32)
        except AttributeError:
            center = np.zeros(2, dtype=np.float32)
            enemy_tower_pos = FALLBACK_ENEMY_TOWER_POS
        target = enemy_tower_pos if self._valid_position(enemy_tower_pos) else FALLBACK_ENEMY_TOWER_POS
        return self._delta_action_16x16(center, target)

    @staticmethod
    def _valid_position(pos):
        if pos is None or len(pos) < 2:
            return False
        return (
            np.isfinite(float(pos[0]))
            and np.isfinite(float(pos[1]))
            and float(pos[0]) != Info.UNSEEN_PADDING
            and float(pos[1]) != Info.UNSEEN_PADDING
        )

    @staticmethod
    def _delta_action_16x16(center, target):
        delta = np.asarray(target, dtype=np.float32) - np.asarray(center, dtype=np.float32)
        max_abs = float(np.max(np.abs(delta)))
        if max_abs < 1e-6:
            return 8, 8
        action = np.ceil(delta / max_abs * 7.0).astype(np.int32) + 8
        action = np.clip(action, 1, 15)
        return int(action[0]), int(action[1])

    @staticmethod
    def _force_head_index(legal, offset, size, desired_index):
        head = legal[offset : offset + size]
        legal_indices = np.flatnonzero(head > 0)
        if legal_indices.size == 0:
            return
        nearest = int(legal_indices[np.argmin(np.abs(legal_indices - int(desired_index)))])
        head[:] = 0.0
        head[nearest] = 1.0

    def _sample_masked_action(self, logits, legal_action):
        prob_list, d_prob_list, action_list, d_action_list = [], [], [], []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))
        target_legal_action_o = np.reshape(legal_actions[-1], [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]])
        one_hot_actions = np.reshape(np.eye(self.label_size_list[0])[action_list[0]], [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))
        one_hot_actions = np.reshape(np.eye(self.label_size_list[0])[d_action_list[0]], [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))
        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 0.00001
        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        return tmp / np.sum(tmp, keepdims=True)

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if use_max:
            return np.argmax(probs)
        return np.argmax(np.random.multinomial(1, probs, size=1))
