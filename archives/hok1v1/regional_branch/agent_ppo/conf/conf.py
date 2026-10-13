#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
class GameConfig:
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 5.0,
        "money": 0.0005,
        "ep_rate": 0.75,
        "death": -1.2,
        "kill": 0.3,
        "exp": 0.0005,
        "last_hit": 0.8,
        "forward": 0.01,
        "hurt_to_hero": 0.00005,
        "lane_progress": 0.35,
        "minion_engage": 0.08,
        "base_idle": 0.18,
        "danger_zone": 0.35,
        "tower_damage": 1.0,
        "push_window": 0.12,
        "grass_idle": 0.12,
        "tower_edge_idle": 0.14,
        "win_loss": 2.0,
        "frenzy_use": 0.5,
        "damage_taken": -0.3,
    }

    REWARD_HEAD_GROUPS = {
        "farm": ("money", "exp", "last_hit", "minion_engage"),
        "kda": ("death", "kill", "frenzy_use"),
        "damage": ("hp_point", "hurt_to_hero", "damage_taken"),
        "push": ("tower_hp_point", "tower_damage", "push_window", "win_loss"),
        "state": ("ep_rate", "forward", "lane_progress", "base_idle", "danger_zone", "grass_idle", "tower_edge_idle"),
    }

    REWARD_HEAD_WEIGHTS = {
        "farm": 1.0,
        "kda": 1.0,
        "damage": 1.0,
        "push": 1.0,
        "state": 1.0,
    }

    VALUE_LOSS_HEAD_WEIGHTS = {
        "farm": 1.0,
        "kda": 1.0,
        "damage": 1.0,
        "push": 1.0,
        "state": 1.0,
    }

    REWARD_MONITOR_GROUPS = {
        "reward_health": ("damage", "kda"),
        "reward_tower": ("push",),
        "reward_economy": ("farm",),
        "reward_state": ("state",),
    }
    TIME_SCALE_ARG = 0
    REWARD_WITHOUT_TIME_SCALE = ("forward",)
    REMOVE_FORWARD_AFTER = None
    MODEL_SAVE_INTERVAL = 1800
    debug_total_frames = 20000


GameConfig.VALUE_HEAD_NAMES = tuple(GameConfig.REWARD_HEAD_GROUPS.keys())
GameConfig.VALUE_HEAD_NUM = len(GameConfig.VALUE_HEAD_NAMES)
GameConfig.REWARD_HEAD_WEIGHT_LIST = tuple(
    GameConfig.REWARD_HEAD_WEIGHTS[name] for name in GameConfig.VALUE_HEAD_NAMES
)
GameConfig.VALUE_LOSS_HEAD_WEIGHT_LIST = tuple(
    GameConfig.VALUE_LOSS_HEAD_WEIGHTS[name] for name in GameConfig.VALUE_HEAD_NAMES
)

assert set(GameConfig.REWARD_HEAD_WEIGHTS) == set(GameConfig.VALUE_HEAD_NAMES)
assert set(GameConfig.VALUE_LOSS_HEAD_WEIGHTS) == set(GameConfig.VALUE_HEAD_NAMES)
assert len(GameConfig.REWARD_HEAD_WEIGHT_LIST) == GameConfig.VALUE_HEAD_NUM
assert len(GameConfig.VALUE_LOSS_HEAD_WEIGHT_LIST) == GameConfig.VALUE_HEAD_NUM


class DimConfig:
    DIM_OF_FEATURE = [0]


class Args:
    RELATIVE_DISTANCE_UNIT_SIZE = 600
    RELATIVE_DISTANCE_MAX_SIZE = 24600
    WHOLE_DISTANCE_UNIT_SIZE = 3000
    WHOLE_DISTANCE_MAX_SIZE = 90000

    HP_UNIT_SIZE = 500
    HP_MAX_SIZE = 12000
    EP_UNIT_SIZE = 50
    EP_MAX_SIZE = 400
    CD_UNIT_SIZE = 10000
    CD_MAX_SIZE = 120000
    MONEY_UNIT_SIZE = 100
    MONEY_MAX_SIZE = 1500
    LEVEL_MAX = 15

    HERO_CONFIG_ID = [112, 133]
    HERO_ID_VALUE_DICT = {
        112: -1.0,  # Luban No.7 / 鲁班七号
        133: 1.0,  # Di Renjie / 狄仁杰
    }
    HERO_BEHAVE = [
        "State_Dead",
        "State_Idle",
        "Direction_Move",
        "Normal_Attack",
        "Attack_Move",
        "State_Revive",
        "UseSkill_1",
        "UseSkill_2",
        "UseSkill_3",
    ]
    SOLDIER_BEHAVE = ["Attack_Path", "State_Dead"]
    RIVER_CRAB_BEHAVE = ["State_Auto", "State_Dead", "State_Born"]
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]

    MARK_ID_LAYERS = {}
    BUFFS = [
        90015,
        10000,
        10010,
        11001,
        11002,
        11010,
        11111,
        112001,
        112015,
        112025,
        112035,
        112040,
        112043,
        112044,
        112045,
        112046,
        112047,
        112048,
        112201,
        112210,
        112300,
        112301,
        112320,
        112890,
        131956,
        133000,
        133001,
        133010,
        133011,
        133020,
        133090,
        133250,
        133260,
        133320,
        133330,
        133350,
        133950,
        133951,
        801090,
        801100,
        801220,
        90019,
        911260,
        911290,
        914110,
        914210,
        914211,
        914267,
    ]
    BULLET_SLOT = [
        "SLOT_SKILL_VALID",
        "SLOT_SKILL_0",
        "SLOT_SKILL_1",
        "SLOT_SKILL_2",
        "SLOT_SKILL_3",
        "SLOT_SKILL_4",
        "SLOT_SKILL_5",
        "SLOT_SKILL_6",
        "SLOT_SKILL_7",
        "SLOT_SKILL_UNKNOWN",
    ]
    BULLET_GSL_SLOT = BULLET_SLOT.copy()
    SUMMONER_SKILL_ID = [
        80102,
        80103,
        80104,
        80105,
        80107,
        80108,
        80109,
        80110,
        80115,
        80121,
    ]
    SOLDIER_MAX_NUM = 4
    BULLET_MAX_NUM = 10


Args.DIM_RPOS = int(
    2 * ((Args.RELATIVE_DISTANCE_MAX_SIZE / Args.RELATIVE_DISTANCE_UNIT_SIZE + 1) / 2) + 1
) * 2 + 1
Args.DIM_WPOS = int(Args.WHOLE_DISTANCE_MAX_SIZE / Args.WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
Args.DIM_DISTANCE = Args.DIM_RPOS + Args.DIM_WPOS
Args.DIM_HP = 1 + int(Args.HP_MAX_SIZE / Args.HP_UNIT_SIZE) + 2
Args.DIM_MARK = sum(layer + 1 for layer in Args.MARK_ID_LAYERS.values())
Args.DIM_UNIT = Args.DIM_HP + Args.DIM_MARK + Args.DIM_DISTANCE
Args.DIM_SKILL = 2 + int(Args.CD_MAX_SIZE / Args.CD_UNIT_SIZE) + 2
Args.DIM_EP = 1 + int(Args.EP_MAX_SIZE / Args.EP_UNIT_SIZE) + 1
Args.DIM_MONEY = 2 + int(Args.MONEY_MAX_SIZE / Args.MONEY_UNIT_SIZE) + 1
Args.DIM_BUFF = len(Args.BUFFS)
Args.DIM_SUMMONER_SKILL = len(Args.SUMMONER_SKILL_ID) + 1
Args.DIM_HERO = (
    1
    + len(Args.HERO_BEHAVE)
    + 1
    + Args.DIM_EP
    + Args.DIM_SUMMONER_SKILL
    + 5 * Args.DIM_SKILL
    + Args.LEVEL_MAX
    + Args.DIM_MONEY
    + 1
    + 2
    + Args.DIM_BUFF
    + Args.DIM_UNIT
)
Args.DIM_SOLDIER = (
    len(Args.SOLDIER_BEHAVE)
    + 1
    + len(Args.SOLDIER_CONFIG_ID)
    + 2
    + Args.DIM_UNIT
)
Args.DIM_SOLDIERS = Args.SOLDIER_MAX_NUM * Args.DIM_SOLDIER
Args.DIM_RIVER_CRAB = len(Args.RIVER_CRAB_BEHAVE) + 1 + Args.DIM_UNIT
Args.DIM_SUB_TOWER = 5 + Args.DIM_UNIT
Args.DIM_ALL_UNITS = (
    2 * Args.DIM_HERO
    + 2 * Args.DIM_SOLDIERS
    + Args.DIM_RIVER_CRAB
    + 2 * Args.DIM_SUB_TOWER
)
Args.DIM_BULLET = len(Args.BULLET_SLOT) + Args.DIM_DISTANCE
Args.DIM_BULLETS = Args.BULLET_MAX_NUM * Args.DIM_BULLET
Args.DIM_GSL_BULLET = len(Args.BULLET_GSL_SLOT) + Args.DIM_DISTANCE
Args.DIM_ALL = Args.DIM_ALL_UNITS + Args.DIM_BULLETS
DimConfig.DIM_OF_FEATURE = [Args.DIM_ALL]
DimConfig.DIM_OF_HERO_FRD = [Args.DIM_HERO]
DimConfig.DIM_OF_HERO_EMY = [Args.DIM_HERO]
DimConfig.DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
DimConfig.DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
DimConfig.DIM_OF_RIVER_CRAB = [Args.DIM_RIVER_CRAB]
DimConfig.DIM_OF_ORGAN_1 = [Args.DIM_SUB_TOWER]
DimConfig.DIM_OF_ORGAN_2 = [Args.DIM_SUB_TOWER]
DimConfig.DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * (Args.BULLET_MAX_NUM - 1)
DimConfig.DIM_OF_BULLET_10 = [Args.DIM_BULLET]


class Config:
    NETWORK_NAME = "network"

    LSTM_TIME_STEPS = 16
    VALUE_HEAD_NAMES = GameConfig.VALUE_HEAD_NAMES
    VALUE_HEAD_NUM = GameConfig.VALUE_HEAD_NUM
    REWARD_HEAD_WEIGHTS = GameConfig.REWARD_HEAD_WEIGHT_LIST
    VALUE_LOSS_HEAD_WEIGHTS = GameConfig.VALUE_LOSS_HEAD_WEIGHT_LIST
    LSTM_UNIT_SIZE = 512
    LSTM_DROPOUT = 0
    DIM_PUBLIC = 256
    FEATURE_DIM = Args.DIM_ALL
    LEGAL_ACTION_DIM = 12 + 16 + 16 + 16 + 16 + 9
    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        VALUE_HEAD_NUM,
        VALUE_HEAD_NUM,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]
    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 5e-5
    TARGET_STEP = 5000
    BETA_START = 0.005
    # Entropy coefficient (beta) annealing. var_beta is linearly decayed from
    # BETA_START to BETA_END over BETA_ANNEAL_STEPS steps (starting at
    # BETA_ANNEAL_START_STEP), then held at BETA_END. Lowering the entropy
    # bonus late in training removes exploration noise so the policy can sharpen.
    # Keyed on the GLOBAL step (Adam optimizer step, restored on resume) so it
    # does NOT reset across restarts. Current run is ~277k global steps; this
    # anneals to 0 by ~317k (~27h @ 25 steps/min). Verify via the monitored
    # "beta_step" (~277k+ = working) and "var_beta" (should tick down).
    # 改为按全局步（Adam 优化器 step，续训会恢复）计，重启不重置；
    # 当前约 27.7 万步，到约 31.7 万步退到 0；用监控 beta_step / var_beta 确认。
    BETA_END = 0.0
    BETA_ANNEAL_START_STEP = 277000
    BETA_ANNEAL_STEPS = 12000
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [VALUE_HEAD_NUM * LSTM_TIME_STEPS],
        [VALUE_HEAD_NUM * LSTM_TIME_STEPS],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
