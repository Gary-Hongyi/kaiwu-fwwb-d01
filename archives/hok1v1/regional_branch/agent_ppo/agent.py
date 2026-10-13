"""PPO 智能体封装：连接观测构建、动作采样、合法性修复和模型生命周期。"""

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import os
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

# Decorator that lets the framework actually load the opponent model configured in
# kaiwu.json model_pool (for eval/PVP). Try both framework packages; fall back to a
# no-op decorator so a missing wrapper can never break startup.
# 让框架真正加载 kaiwu.json model_pool 里对手模型的装饰器；两个框架包都试，
# 都没有就退化成空装饰器，保证缺它也不会导致训练起不来。
try:
    from kaiwudrl.interface.agent import load_opponent_agent_wrapper
except (ImportError, AttributeError):
    try:
        from kaiwu_agent.agent.base_agent import load_opponent_agent_wrapper
    except (ImportError, AttributeError):
        def load_opponent_agent_wrapper(func):
            return func

from agent_ppo.conf.conf import Config
from agent_ppo.feature.reward_manager import FedExingGameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.obs_builder import FedExingObsBuilder
from agent_ppo.feature.unpack_state_dict import Info


SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())
DEFAULT_SUMMONER_SKILL = 80110
BUTTON_CHOSEN_SUMMONER_SKILL = 8
DEFAULT_DISCRETE_ACTION = [0, 15, 15, 15, 15, 0]
BUTTON_MOVE = 2
BUTTON_NORMAL_ATTACK = 3
TARGET_ENEMY_HERO = 1
TARGET_ENEMY_TOWER = 7
FALLBACK_BUTTON_ORDER = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
TARGET_FALLBACK_ORDER = [0, 1, 3, 4, 5, 6, 7, 8, 2]
COMBAT_TARGET_FALLBACK_ORDER = [1, 3, 4, 5, 6, 7, 8, 2, 0]
DEAD_BEHAVIORS = {"State_Dead", "State_Revive"}
MAP_COORDINATE_LIMIT = 400000
DEFAULT_TOWER_RANGE = 6000
ALLY_TOWER_COVER_BONUS = 1800
ALLY_HERO_COVER_RADIUS = 5500
SAFE_PUSH_TOWER_BONUS = 2800
ENEMY_THREAT_RADIUS = 6000
LOW_ENEMY_HP_RATIO = 0.35
SAFE_HERO_HP_RATIO = 0.45
CHASE_HERO_HP_RATIO = 0.55
LEGAL_EPSILON = 1e-6
MASKED_LOGIT_BARRIER = 1e20
SOFTMAX_EPSILON = 1e-6


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self._setup_model(device)
        self._setup_action_space()
        self._setup_episode_state()
        self._setup_optimizer()
        self._setup_runtime_services(logger, monitor)
        super().__init__(agent_type, device, logger, monitor)

    def _setup_model(self, device):
        """初始化网络权重容器，并切换为 channels_last 内存布局。"""

        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

    def _setup_action_space(self):
        """缓存动作空间和 LSTM 维度配置，减少热路径上的 Config 查表。"""

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

    def _setup_episode_state(self):
        """初始化跨帧状态，reset 时会按新对局再次刷新。"""

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None
        self.reward_manager = None
        self.info = Info()
        self._fedexing_hero_dead = False
        self._fedexing_was_dead = False
        self._fedexing_last_dead_cnt = 0
        self._fedexing_last_frame_no = 0

    def _setup_optimizer(self):
        """装配 Adam 优化器与线性学习率调度器。"""

        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

    def _setup_runtime_services(self, logger, monitor):
        """连接观测构建器、奖励管理器入口和 PPO 学习器。"""

        self.obs_builder = FedExingObsBuilder(logger)
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            act_data.action = self._fedexing_repair_action(observation, act_data.action)
            return act_data.action
        else:
            act_data.d_action = self._fedexing_repair_action(observation, act_data.d_action)
            return act_data.d_action

    def _fedexing_raw_legal_action_len(self):
        return sum(self.label_size_list[:-1]) + self.label_size_list[0] * self.label_size_list[-1]

    def _fedexing_default_raw_legal_action(self):
        legal_action = np.zeros(self._fedexing_raw_legal_action_len(), dtype=np.float32)
        offset = 0
        for index, size in enumerate(self.label_size_list[:-1]):
            fallback = max(0, min(DEFAULT_DISCRETE_ACTION[index], size - 1))
            legal_action[offset + fallback] = 1.0
            offset += size
        target_offset = offset + DEFAULT_DISCRETE_ACTION[0] * self.label_size_list[-1]
        legal_action[target_offset + DEFAULT_DISCRETE_ACTION[-1]] = 1.0
        return legal_action

    def _fedexing_split_raw_legal_action(self, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float32).reshape(-1)
        if legal_action.size != self._fedexing_raw_legal_action_len():
            return None
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = list(np.split(legal_action, label_split_size[:-1]))
        legal_actions[-1] = legal_actions[-1].reshape(self.label_size_list[0], self.label_size_list[-1])
        return legal_actions

    def _fedexing_pick_legal_index(self, legal_mask, preferred, fallback_order):
        legal_mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
        if legal_mask.size == 0:
            return int(preferred)
        preferred = int(max(0, min(int(preferred), legal_mask.size - 1)))
        if legal_mask[preferred] > 0:
            return preferred
        for candidate in fallback_order:
            candidate = int(candidate)
            if 0 <= candidate < legal_mask.size and legal_mask[candidate] > 0:
                return candidate
        legal_indices = np.flatnonzero(legal_mask > 0)
        if legal_indices.size > 0:
            return int(legal_indices[0])
        return preferred

    def _fedexing_mask_empty_target_if_possible(self, target_mask, button):
        target_mask = np.asarray(target_mask, dtype=np.float32).reshape(-1)
        if button >= 3 and target_mask.size > 1 and target_mask[0] > 0 and np.sum(target_mask[1:]) > 0:
            target_mask = target_mask.copy()
            target_mask[0] = 0.0
        return target_mask

    def _fedexing_pick_target_index(self, target_mask, preferred, button):
        target_mask = self._fedexing_mask_empty_target_if_possible(target_mask, button)
        fallback_order = COMBAT_TARGET_FALLBACK_ORDER if button >= 3 else TARGET_FALLBACK_ORDER
        return self._fedexing_pick_legal_index(target_mask, preferred, fallback_order)

    @staticmethod
    def _fedexing_actor_point(actor):
        """从对象或协议字典中提取可用于距离计算的二维坐标。"""

        pos = Agent._fedexing_raw_position(actor)
        return Agent._fedexing_position_tuple(pos)

    @staticmethod
    def _fedexing_raw_position(actor):
        """兼容 Info 对象、ActorInfo 对象和环境原始 dict 三种位置结构。"""

        if actor is None:
            return None
        if hasattr(actor, "position"):
            return actor.position
        if hasattr(actor, "info") and hasattr(actor.info, "position"):
            return actor.info.position
        if isinstance(actor, dict):
            return actor.get("location")
        return None

    @staticmethod
    def _fedexing_position_tuple(pos):
        """过滤缺失、维度不足或明显越界的坐标。"""

        if isinstance(pos, dict):
            return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=np.float32).reshape(-1)
        if pos.size < 2 or np.any(np.abs(pos[:2]) > MAP_COORDINATE_LIMIT):
            return None
        return float(pos[0]), float(pos[1])

    @staticmethod
    def _fedexing_dist(p1, p2):
        if p1 is None or p2 is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))

    @staticmethod
    def _fedexing_actor_alive(actor):
        if actor is None:
            return False
        hp = getattr(actor, "hp", None)
        if hp is None and hasattr(actor, "info"):
            hp = getattr(actor.info, "hp", 0)
        return float(hp or 0) > 0

    @staticmethod
    def _fedexing_actor_hp_ratio(actor):
        if actor is None:
            return 0.0
        hp = getattr(actor, "hp", None)
        hp_max = getattr(actor, "hp_max", None)
        if hp is None and hasattr(actor, "info"):
            hp = getattr(actor.info, "hp", 0)
            hp_max = getattr(actor.info, "hp_max", 1)
        return max(0.0, min(1.0, float(hp or 0) / max(float(hp_max or 1), 1.0)))

    def _fedexing_has_ally_minion_near(self, point, radius):
        soldiers = getattr(getattr(self.info, "soldiers_our", None), "merge", [])
        for soldier in soldiers:
            if self._fedexing_actor_alive(soldier):
                if self._fedexing_dist(self._fedexing_actor_point(soldier), point) <= radius:
                    return True
        return False

    def _fedexing_safe_push_info_context(self):
        """评估当前是否处在可安全点塔的窗口，并返回判定上下文。"""

        hero = getattr(getattr(self.info, "hero_our", None), "info", None)
        enemy = getattr(getattr(self.info, "hero_enemy", None), "info", None)
        tower = getattr(getattr(self.info, "organ_enemy", None), "sub_tower", None)
        context = self._fedexing_base_safe_push_context(hero, enemy, tower)
        if not self._fedexing_actor_alive(hero) or not self._fedexing_actor_alive(tower):
            return False, context

        context.update(self._fedexing_push_geometry(hero, tower))
        hero_hp_ratio = self._fedexing_actor_hp_ratio(hero)
        context["ally_cover"] = self._fedexing_has_push_cover(context)
        context["enemy_threat"] = self._fedexing_enemy_push_threat(enemy, context["hero_pos"], hero_hp_ratio)
        safe = self._fedexing_is_safe_push_context(context, hero_hp_ratio)
        return safe, context

    @staticmethod
    def _fedexing_base_safe_push_context(hero, enemy, tower):
        """创建安全推塔判定的默认上下文，所有缺省值都按保守策略处理。"""

        return {
            "hero": hero,
            "enemy": enemy,
            "tower": tower,
            "hero_pos": None,
            "tower_pos": None,
            "tower_dist": float("inf"),
            "tower_range": DEFAULT_TOWER_RANGE,
            "enemy_threat": False,
            "ally_cover": False,
        }

    def _fedexing_push_geometry(self, hero, tower):
        """计算英雄与敌方外塔之间的距离、塔攻击范围和坐标。"""

        hero_pos = self._fedexing_actor_point(hero)
        tower_pos = self._fedexing_actor_point(tower)
        tower_range = int(getattr(tower, "attack_range", 0) or DEFAULT_TOWER_RANGE)
        return {
            "hero_pos": hero_pos,
            "tower_pos": tower_pos,
            "tower_dist": self._fedexing_dist(hero_pos, tower_pos),
            "tower_range": tower_range,
        }

    def _fedexing_has_push_cover(self, context):
        """判断小兵是否已经提供抗塔或站位保护。"""

        tower_cover = self._fedexing_has_ally_minion_near(
            context["tower_pos"],
            context["tower_range"] + ALLY_TOWER_COVER_BONUS,
        )
        hero_cover = self._fedexing_has_ally_minion_near(context["hero_pos"], ALLY_HERO_COVER_RADIUS)
        return tower_cover or hero_cover

    def _fedexing_enemy_push_threat(self, enemy, hero_pos, hero_hp_ratio):
        """敌方英雄可见且不处于被追击残血状态时，近距离视为推塔威胁。"""

        if not self._fedexing_actor_alive(enemy):
            return False
        enemy_dist = self._fedexing_dist(hero_pos, self._fedexing_actor_point(enemy))
        enemy_hp_ratio = self._fedexing_actor_hp_ratio(enemy)
        is_chase_window = enemy_hp_ratio < LOW_ENEMY_HP_RATIO and hero_hp_ratio > CHASE_HERO_HP_RATIO
        return enemy_dist <= ENEMY_THREAT_RADIUS and not is_chase_window

    @staticmethod
    def _fedexing_is_safe_push_context(context, hero_hp_ratio):
        """把血量、兵线、塔距和敌方威胁合成为最终推塔许可。"""

        return (
            hero_hp_ratio >= SAFE_HERO_HP_RATIO
            and context["ally_cover"]
            and context["tower_dist"] <= context["tower_range"] + SAFE_PUSH_TOWER_BONUS
            and not context["enemy_threat"]
        )

    def _fedexing_apply_safe_push_tower_rule(self, observation, legal_actions, action):
        """在安全推塔窗口内，把非击杀窗口的动作修正为普攻防御塔。"""

        attack_sub_mask = self._fedexing_tower_attack_mask(observation, legal_actions)
        if attack_sub_mask is None or self._fedexing_keep_current_combat_target(action):
            return action
        return self._fedexing_force_tower_attack(legal_actions, action, attack_sub_mask)

    def _fedexing_tower_attack_mask(self, observation, legal_actions):
        """校验安全推塔、普攻按钮、目标头和子动作掩码是否同时可用。"""

        safe_push, _ = self._fedexing_safe_push_info_context()
        if not safe_push:
            return None
        if BUTTON_NORMAL_ATTACK >= legal_actions[0].size or legal_actions[0][BUTTON_NORMAL_ATTACK] <= 0:
            return None
        if TARGET_ENEMY_TOWER >= self.label_size_list[-1]:
            return None

        attack_sub_mask = self._fedexing_normalize_sub_action_mask(
            observation.get("sub_action_mask", {}),
            BUTTON_NORMAL_ATTACK,
        )
        if attack_sub_mask[-1] <= 0:
            return None
        if legal_actions[-1][BUTTON_NORMAL_ATTACK][TARGET_ENEMY_TOWER] <= 0:
            return None
        return attack_sub_mask

    @staticmethod
    def _fedexing_keep_current_combat_target(action):
        """当当前动作已经指向敌方英雄时，保留击杀窗口优先级。"""

        current_button = int(action[0])
        current_target = int(action[-1])
        return current_button >= BUTTON_NORMAL_ATTACK and current_target == TARGET_ENEMY_HERO

    def _fedexing_force_tower_attack(self, legal_actions, action, attack_sub_mask):
        """按合法动作掩码重建一个普攻防御塔动作。"""

        forced = np.asarray(action, dtype=np.int32).copy()
        forced[0] = BUTTON_NORMAL_ATTACK
        for index in range(1, len(self.label_size_list) - 1):
            if attack_sub_mask[index] <= 0:
                forced[index] = DEFAULT_DISCRETE_ACTION[index]
            else:
                forced[index] = self._fedexing_pick_legal_index(
                    legal_actions[index],
                    DEFAULT_DISCRETE_ACTION[index],
                    [DEFAULT_DISCRETE_ACTION[index]],
                )
        forced[-1] = TARGET_ENEMY_TOWER
        return forced

    def _fedexing_clip_action(self, action):
        fixed = np.asarray(DEFAULT_DISCRETE_ACTION, dtype=np.int32).copy()
        if action is None:
            return fixed
        action = np.asarray(action, dtype=np.int32).reshape(-1)
        if action.size != len(self.label_size_list):
            return fixed
        for index, size in enumerate(self.label_size_list):
            fixed[index] = int(max(0, min(int(action[index]), size - 1)))
        return fixed

    def _fedexing_normalize_sub_action_mask(self, sub_action_mask, button):
        if isinstance(sub_action_mask, dict):
            mask = sub_action_mask.get(str(button), sub_action_mask.get(button))
        elif isinstance(sub_action_mask, list):
            if len(sub_action_mask) == self.label_size_list[0] * len(self.label_size_list):
                start = button * len(self.label_size_list)
                mask = sub_action_mask[start : start + len(self.label_size_list)]
            elif 0 <= button < len(sub_action_mask):
                mask = sub_action_mask[button]
            else:
                mask = None
        else:
            mask = None

        if mask is None:
            return np.ones(len(self.label_size_list), dtype=np.int32)
        mask = np.asarray(mask, dtype=np.int32).reshape(-1)
        if mask.size != len(self.label_size_list):
            return np.ones(len(self.label_size_list), dtype=np.int32)
        return mask

    def _fedexing_repair_action(self, observation, action):
        """按环境合法动作和子动作掩码修复模型输出，保证下发动作可执行。"""

        action = self._fedexing_clip_action(action)
        legal_actions = self._fedexing_split_raw_legal_action(observation.get("legal_action", []))
        if legal_actions is None:
            return DEFAULT_DISCRETE_ACTION.copy()

        button = int(action[0])
        action[0] = self._fedexing_pick_legal_index(legal_actions[0], button, FALLBACK_BUTTON_ORDER)
        button = int(action[0])
        sub_action_mask = self._fedexing_normalize_sub_action_mask(observation.get("sub_action_mask", {}), button)
        self._fedexing_repair_argument_heads(action, legal_actions, sub_action_mask)
        self._fedexing_repair_target_head(action, legal_actions, sub_action_mask, button)
        action = self._fedexing_apply_safe_push_tower_rule(observation, legal_actions, action)
        return action.astype(np.int32).tolist()

    def _fedexing_repair_argument_heads(self, action, legal_actions, sub_action_mask):
        """修复方向、技能槽等非目标动作头。"""

        for index in range(1, len(self.label_size_list) - 1):
            if sub_action_mask[index] <= 0:
                action[index] = DEFAULT_DISCRETE_ACTION[index]
            else:
                action[index] = self._fedexing_pick_legal_index(
                    legal_actions[index],
                    action[index],
                    [DEFAULT_DISCRETE_ACTION[index]],
                )

    def _fedexing_repair_target_head(self, action, legal_actions, sub_action_mask, button):
        """根据最终按钮修复目标动作头。"""

        if sub_action_mask[-1] <= 0:
            action[-1] = DEFAULT_DISCRETE_ACTION[-1]
        else:
            action[-1] = self._fedexing_pick_target_index(legal_actions[-1][button], action[-1], button)

    def _fedexing_build_default_prob(self, legal_action, button):
        legal_actions = self._fedexing_split_raw_legal_action(legal_action)
        if legal_actions is None:
            legal_actions = self._fedexing_split_raw_legal_action(self._fedexing_default_raw_legal_action())

        prob_list = []
        for index in range(len(self.label_size_list) - 1):
            mask = np.asarray(legal_actions[index], dtype=np.float64)
            if np.sum(mask) <= 0:
                mask = np.zeros(self.label_size_list[index], dtype=np.float64)
                mask[DEFAULT_DISCRETE_ACTION[index]] = 1.0
            prob_list += list(mask / np.sum(mask))

        button = int(max(0, min(int(button), self.label_size_list[0] - 1)))
        target_mask = np.asarray(legal_actions[-1][button], dtype=np.float64)
        if np.sum(target_mask) <= 0:
            target_mask = np.zeros(self.label_size_list[-1], dtype=np.float64)
            target_mask[DEFAULT_DISCRETE_ACTION[-1]] = 1.0
        prob_list += list(target_mask / np.sum(target_mask))
        return [prob_list]

    def _fedexing_default_act_data(self, observation, obs_data):
        action = self._fedexing_repair_action(observation, DEFAULT_DISCRETE_ACTION)
        prob = self._fedexing_build_default_prob(observation.get("legal_action", []), action[0])
        value = np.zeros((1, Config.VALUE_HEAD_NUM), dtype=np.float32)
        return ActData(
            action=action,
            d_action=action.copy(),
            prob=prob,
            d_prob=prob,
            value=value,
            lstm_cell=np.asarray(obs_data.lstm_cell, dtype=np.float32).copy(),
            lstm_hidden=np.asarray(obs_data.lstm_hidden, dtype=np.float32).copy(),
        )

    def _fedexing_should_use_default_action(self, obs_data):
        legal_action = np.asarray(obs_data.legal_action, dtype=np.float32).reshape(-1)
        return self._fedexing_hero_dead or legal_action.size != self._fedexing_raw_legal_action_len()

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        if self._fedexing_should_use_default_action(obs_data):
            act_data = self._fedexing_default_act_data(observation, obs_data)
        else:
            act_data = self._fedexing_model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    def _fedexing_apply_sub_action_mask(self, observation, action):
        return self._fedexing_repair_action(observation, action)

    def _fedexing_get_sub_action_mask(self, sub_action_mask, button):
        return self._fedexing_normalize_sub_action_mask(sub_action_mask, button)

    def _fedexing_is_legal_action(self, legal_action, action, sub_action_mask=None):
        """校验动作在当前合法动作掩码和子动作约束下是否可执行。"""

        legal_actions = self._fedexing_split_raw_legal_action(legal_action)
        if legal_actions is None:
            return True
        sub_action_mask = self._fedexing_legality_sub_mask(sub_action_mask)

        button = int(action[0])
        if not self._fedexing_is_legal_button(legal_actions, button):
            return False
        if not self._fedexing_are_legal_arguments(legal_actions, action, sub_action_mask):
            return False
        return self._fedexing_is_legal_target(legal_actions, action, sub_action_mask, button)

    def _fedexing_legality_sub_mask(self, sub_action_mask):
        """把外部子动作掩码转换为固定长度数组，缺省时全部放开。"""

        if sub_action_mask is None or len(sub_action_mask) != len(self.label_size_list):
            return np.ones(len(self.label_size_list), dtype=np.int32)
        return np.asarray(sub_action_mask, dtype=np.int32)

    def _fedexing_is_legal_button(self, legal_actions, button):
        """校验主按钮动作头是否越界且被环境放开。"""

        if button < 0 or button >= self.label_size_list[0]:
            return False
        return bool(legal_actions[0][button] > LEGAL_EPSILON)

    def _fedexing_are_legal_arguments(self, legal_actions, action, sub_action_mask):
        """校验主按钮之外、目标头之前的参数动作头。"""

        for index in range(1, len(self.label_size_list) - 1):
            if sub_action_mask[index] <= 0:
                continue
            value = int(action[index])
            if value < 0 or value >= self.label_size_list[index]:
                return False
            if legal_actions[index][value] <= LEGAL_EPSILON:
                return False
        return True

    def _fedexing_is_legal_target(self, legal_actions, action, sub_action_mask, button):
        """校验目标头；当子动作禁用目标选择时直接视为合法。"""

        if sub_action_mask[-1] <= 0:
            return True
        target = int(action[-1])
        if target < 0 or target >= self.label_size_list[-1]:
            return False
        return bool(legal_actions[-1][button][target] > LEGAL_EPSILON)

    def _fedexing_legal_sample(self, probs, legal_action=None, use_max=False):
        if use_max:
            return np.argmax(probs)

        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / np.sum(probs)
        return min(int(np.searchsorted(np.cumsum(probs), np.random.random(), side="right")), len(probs) - 1)

    def _fedexing_legal_soft_max(self, input_hidden, legal_action, fallback_index=0):
        """在非法动作 logits 上施加屏蔽后计算稳定 softmax。"""

        legal_action = np.asarray(legal_action, dtype=np.float64)
        if legal_action.size == 0:
            return legal_action
        legal_action = self._fedexing_non_empty_legal_mask(legal_action, fallback_index)
        return self._fedexing_masked_softmax(input_hidden, legal_action)

    @staticmethod
    def _fedexing_non_empty_legal_mask(legal_action, fallback_index):
        """当环境未放开任何动作时，仅保留兜底动作以维持采样稳定。"""

        if np.sum(legal_action) <= 1e-8:
            legal_action = np.zeros_like(legal_action, dtype=np.float64)
            fallback_index = max(0, min(int(fallback_index), legal_action.size - 1))
            legal_action[fallback_index] = 1.0
        return legal_action

    @staticmethod
    def _fedexing_masked_softmax(input_hidden, legal_action):
        """用大负偏置屏蔽非法动作，并进行两次归一化抑制数值误差。"""

        input_hidden = np.asarray(input_hidden, dtype=np.float64)
        tmp = input_hidden - MASKED_LOGIT_BARRIER * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -MASKED_LOGIT_BARRIER, 1)
        tmp = (np.exp(tmp) + SOFTMAX_EPSILON) * legal_action

        tmp_sum = np.sum(tmp, keepdims=True)
        tmp_sum = np.maximum(tmp_sum, SOFTMAX_EPSILON)
        probs = tmp / tmp_sum
        return probs / np.maximum(np.sum(probs, keepdims=True), SOFTMAX_EPSILON)

    def _fedexing_model_inference(self, list_obs_data):
        """执行模型推理，并把网络输出转换为框架动作数据对象。"""
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        format_inputs = self._fedexing_inference_inputs(list_obs_data)
        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)
        logits, value, lstm_cell, lstm_hidden = self._fedexing_numpy_outputs(output_list)
        return self._fedexing_build_act_data(logits, value, lstm_cell, lstm_hidden, legal_action)

    def _fedexing_inference_inputs(self, list_obs_data):
        """把 ObsData 列表转换为模型推理所需的特征与 LSTM 状态张量。"""

        input_list = [
            np.array([obs_data.feature for obs_data in list_obs_data]),
            np.array([obs_data.lstm_cell for obs_data in list_obs_data]),
            np.array([obs_data.lstm_hidden for obs_data in list_obs_data]),
        ]
        feature, lstm_cell, lstm_hidden = self._fedexing_to_model_tensors(input_list)
        return [
            feature.reshape(-1, self.seri_vec_split_shape[0][0]),
            lstm_hidden.reshape(-1, self.lstm_unit_size),
            lstm_cell.reshape(-1, self.lstm_unit_size),
        ]

    def _fedexing_to_model_tensors(self, input_list):
        """把 numpy 输入移动到模型所在设备，并压平 batch 外维度。"""

        model_device = next(self.model.parameters()).device
        return [
            torch.as_tensor(nparr, dtype=torch.float32, device=model_device).reshape(-1).float()
            for nparr in input_list
        ]

    @staticmethod
    def _fedexing_numpy_outputs(output_list):
        """把网络输出从 torch.Tensor 转为 numpy，便于后续 numpy 采样。"""

        logits, value, lstm_cell, lstm_hidden = [output.detach().cpu().numpy() for output in output_list[:4]]
        return logits, value, lstm_cell.squeeze(axis=0), lstm_hidden.squeeze(axis=0)

    def _fedexing_build_act_data(self, logits, value, lstm_cell, lstm_hidden, legal_action):
        """根据 logits 和合法动作掩码生成随机动作与贪心动作。"""

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._fedexing_sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=lstm_cell[i],
                    lstm_hidden=lstm_hidden[i],
                )
            )
        return list_act_data

    def _fedexing_repair_target_mask(self, target_mask, button_target_masks):
        target_mask = np.asarray(target_mask, dtype=np.float32)
        if np.sum(target_mask) > 0:
            return target_mask

        merged_mask = np.asarray(button_target_masks, dtype=np.float32).max(axis=0)
        if np.sum(merged_mask) > 0:
            return merged_mask

        target_mask = np.zeros(self.label_size_list[-1], dtype=np.float32)
        target_mask[0] = 1.0
        return target_mask

    def _fedexing_sample_masked_action(self, logits, legal_action):
        """按合法动作掩码同时采样训练动作和贪心展示动作。"""

        legal_actions, logits_split = self._fedexing_split_action_heads(logits, legal_action)
        prob_list, d_prob_list, action_list, d_action_list = self._fedexing_sample_argument_heads(
            legal_actions,
            logits_split,
        )
        target_masks = self._fedexing_target_masks_for_buttons(legal_actions, action_list[0], d_action_list[0])
        self._fedexing_append_target_sample(prob_list, action_list, logits_split[-1], target_masks[0], False)
        self._fedexing_append_target_sample(d_prob_list, d_action_list, logits_split[-1], target_masks[1], True)
        return [prob_list], [d_prob_list], action_list, d_action_list

    def _fedexing_split_action_heads(self, logits, legal_action):
        """按动作头维度同步拆分 logits 和合法动作掩码。"""

        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(np.asarray(legal_action, dtype=np.float32), label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        return legal_actions, logits_split

    def _fedexing_sample_argument_heads(self, legal_actions, logits_split):
        """采样除目标头以外的动作头，目标头依赖主按钮后续单独处理。"""

        prob_list, d_prob_list, action_list, d_action_list = [], [], [], []
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._fedexing_head_probs(logits_split[index], legal_actions[index], index)
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._fedexing_legal_sample(probs, use_max=False))
            d_action_list.append(self._fedexing_legal_sample(probs, use_max=True))
        return prob_list, d_prob_list, action_list, d_action_list

    def _fedexing_head_probs(self, logits, legal_mask, head_index):
        """计算单个动作头在合法掩码下的概率分布。"""

        return self._fedexing_legal_soft_max(
            logits,
            legal_mask,
            fallback_index=DEFAULT_DISCRETE_ACTION[head_index],
        )

    def _fedexing_target_masks_for_buttons(self, legal_actions, sample_button, greedy_button):
        """分别为随机按钮和贪心按钮构造目标头合法掩码。"""

        expanded_target_mask = self._fedexing_expanded_target_mask(legal_actions[-1])
        return (
            self._fedexing_target_mask_for_button(expanded_target_mask, sample_button),
            self._fedexing_target_mask_for_button(expanded_target_mask, greedy_button),
        )

    def _fedexing_expanded_target_mask(self, target_mask):
        """恢复按钮展开式目标掩码矩阵。"""

        return np.reshape(
            target_mask,
            [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]],
        )

    def _fedexing_target_mask_for_button(self, expanded_target_mask, button):
        """根据主按钮选择对应目标掩码，并在空掩码时合并兜底。"""

        one_hot_actions = np.eye(self.label_size_list[0])[button].reshape([self.label_size_list[0], 1])
        target_mask = np.sum(expanded_target_mask * one_hot_actions, axis=0)
        target_mask = self._fedexing_repair_target_mask(target_mask, expanded_target_mask)
        return self._fedexing_mask_empty_target_if_possible(target_mask, button)

    def _fedexing_append_target_sample(self, prob_list, action_list, logits, target_mask, use_max):
        """把目标头概率和采样结果追加到动作序列末尾。"""

        probs = self._fedexing_legal_soft_max(logits, target_mask, fallback_index=DEFAULT_DISCRETE_ACTION[-1])
        prob_list += list(probs)
        action_list.append(self._fedexing_legal_sample(probs, use_max=use_max))

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        opponent_heroes = config_data.get("opponent_heroes", [])
        opponent_hero_id = opponent_heroes[0] if opponent_heroes else None
        select_skills = {}
        for hero_id in my_heroes:
            select_skills[hero_id] = _fedexing_choose_summoner_skill(hero_id, opponent_hero_id)
        return select_skills

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            load_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(load_dict)
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        # Framework loads the opponent model (from kaiwu.json model_pool) via this
        # decorator; the body stays empty, per the official template.
        # 框架通过该装饰器加载对手模型（来自 kaiwu.json model_pool），函数体留空即可。
        pass

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def _fedexing_reset_lstm_state(self):
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

    def _fedexing_update_life_status(self):
        hero = self.info.hero_our
        hp = int(getattr(hero.info, "hp", 0) or 0)
        behave = getattr(hero.info, "behave", "")
        revive_time = int(getattr(hero, "revive_time", 0) or 0)
        dead_cnt = int(getattr(hero, "deadCnt", 0) or 0)
        frame_no = int(getattr(self.info, "n_frame", 0) or 0)

        is_dead = hp <= 0 or revive_time > 0 or behave == "State_Dead"
        just_revived = self._fedexing_hero_dead and not is_dead
        new_death = dead_cnt > self._fedexing_last_dead_cnt
        revive_transition = behave in DEAD_BEHAVIORS

        if just_revived or new_death or revive_transition:
            self._fedexing_reset_lstm_state()

        self._fedexing_hero_dead = is_dead
        self._fedexing_was_dead = is_dead
        self._fedexing_last_dead_cnt = max(self._fedexing_last_dead_cnt, dead_cnt)
        self._fedexing_last_frame_no = frame_no

    def observation_process(self, observation):
        self.info.update(observation)
        self._fedexing_update_life_status()
        feature = self.obs_builder.build_observation(self.info)
        feature_vec, legal_action = (
            feature,
            observation.get("legal_action", []),
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        if self._fedexing_should_use_default_action(obs_data):
            act_data = self._fedexing_default_act_data(observation, obs_data)
        else:
            act_data = self._fedexing_model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    def reset(self, observation):
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self._fedexing_reset_lstm_state()
        self.reward_manager = FedExingGameRewardManager(self.player_id)
        self._fedexing_hero_dead = False
        self._fedexing_was_dead = False
        self._fedexing_last_dead_cnt = 0
        self._fedexing_last_frame_no = 0
        self.info.reset()
        self.obs_builder.reset()

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        if self._fedexing_hero_dead:
            self._fedexing_reset_lstm_state()
        else:
            self.lstm_cell = act_data.lstm_cell
            self.lstm_hidden = act_data.lstm_hidden


def _fedexing_choose_summoner_skill(my_hero_id, opponent_hero_id):
    """当前英雄池统一携带狂暴，强化对拼与追击窗口的爆发收益。"""

    return DEFAULT_SUMMONER_SKILL
