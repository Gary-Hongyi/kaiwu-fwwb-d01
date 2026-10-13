"""Current-protocol observation adapter exposing a structured object API."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from agent_ppo.conf.conf import Config
except Exception:  # pragma: no cover - lets this module run as a loose script.
    class Config:
        LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]


UNSEEN_PADDING = 100000

# Module-level flag controlling cvt_position behavior.
inverse_position = False

_ACTOR_TYPE_NAMES = {
    0: "ACTOR_HERO",
    1: "ACTOR_MONSTER",
    2: "ACTOR_ORGAN",
    3: "ACTOR_BULLET",
    4: "ACTOR_SHENFU",
}
_ACTOR_TYPE_ALIASES = {
    "ACTOR_TYPE_HERO": "ACTOR_HERO",
    "ACTOR_TYPE_MONSTER": "ACTOR_MONSTER",
    "ACTOR_TYPE_ORGAN": "ACTOR_ORGAN",
    "ACTOR_TYPE_BULLET": "ACTOR_BULLET",
    "ACTOR_TYPE_SHENFU": "ACTOR_SHENFU",
}
_ACTOR_SUB_TYPE_NAMES = {
    0: "ACTOR_SUB_NONE",
    11: "ACTOR_SUB_SOLDIER",
    21: "ACTOR_SUB_TOWER",
    23: "ACTOR_SUB_TOWER_SPRING",
    24: "ACTOR_SUB_CRYSTAL",
}
_ACTOR_SUB_TYPE_ALIASES = {
    "ACTOR_SUB_TOWER_SPRING": "ACTOR_SUB_TOWER_SPRING",
    "ACTOR_SUB_TOWER": "ACTOR_SUB_TOWER",
    "ACTOR_SUB_CRYSTAL": "ACTOR_SUB_CRYSTAL",
    "ACTOR_SUB_SOLDIER": "ACTOR_SUB_SOLDIER",
}
_BEHAVIOR_NAMES = {
    0: "State_Idle",
    1: "State_Dead",
    2: "Direction_Move",
    3: "State_GameOver",
    4: "Normal_Attack",
    5: "Attack_Move",
    6: "Attack_Path",
    7: "State_Auto",
    8: "State_Born",
    9: "UseSkill_1",
    10: "UseSkill_2",
    11: "UseSkill_3",
    12: "UseSkill_4",
    14: "State_Revive",
}
_SOLDIER_CLOSE_CONFIG_IDS = {6801, 6804}
_SOLDIER_REMOTE_CONFIG_IDS = {6800, 6803}
_SOLDIER_CAR_CONFIG_IDS = {6802, 6805}
_RIVER_CRAB_CONFIG_ID = 6827
_SHOW_SKIP_KEYS = {
    "state_dict",
    "legal_action",
    "_raw_legal_action",
    "reward",
    "frame_state",
    "actor_state",
    "hero_state",
    "skill_state",
    "slot_state",
    "equip_state",
    "buff_state",
    "values",
    "logger",
}


class Struct:
    """Small recursive object wrapper kept for backwards compatibility."""

    def __init__(self, data: Optional[dict] = None):
        for key, value in (data or {}).items():
            setattr(self, key, _fedexing_wrap(value))

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Struct({items})"


def _fedexing_wrap(value):
    if isinstance(value, dict):
        return Struct(value)
    if isinstance(value, list):
        return [_fedexing_wrap(item) if isinstance(item, dict) else item for item in value]
    return value


class Info:
    """Wraps one side of env observation using the structured field layout."""

    UNSEEN_PADDING = UNSEEN_PADDING

    def __init__(self, observation: Optional[dict] = None, reverse_camp1: bool = True):
        self.reverse_camp1 = reverse_camp1
        self.id2type: Dict[int, str] = {}
        self._fedexing_set_empty_state()
        if observation is not None:
            self.update(observation)

    def _fedexing_set_empty_state(self):
        self.state_dict = {}
        self.player_id = 0
        self.player_camp = -1
        self.game_id = ""
        self.env_id = ""
        self.legal_action = np.asarray([], dtype=np.float32)
        self._raw_legal_action = self.legal_action
        self.legal_actions = []
        self.sub_action_mask = {}
        self.frame_state = {}
        self.n_frame = 0
        self.map_state = False
        self.hero_our = HeroInfo(None, 0)
        self.hero_enemy = HeroInfo(None, 1)
        self.organ_our = OrganInfo(None, 0)
        self.organ_enemy = OrganInfo(None, 1)
        self.soldiers_our = SoldierInfo(None, 0)
        self.soldiers_enemy = SoldierInfo(None, 1)
        self.river_crab = None
        self.cake_our = None
        self.cake_enemy = None
        self.bullets_our = BulletsInfo(None, 0, self.id2type)
        self.bullets_enemy = BulletsInfo(None, 1, self.id2type)
        self.deads = DeadsInfo(None)

    def reset(self):
        self.id2type = {}
        self._fedexing_set_empty_state()

    def update(self, observation: dict):
        s = _fedexing_select_single_observation(observation)
        self.state_dict = s

        self.player_id = int(_fedexing_get(s, "player_id", default=0) or 0)
        self.player_camp = _fedexing_camp_index(_fedexing_get(s, "camp", "player_camp", default=0))
        self.game_id = _fedexing_get(s, "env_id", "game_id", default="")
        self.env_id = self.game_id

        global inverse_position
        inverse_position = self.player_camp == 1 and self.reverse_camp1

        self.legal_action = np.asarray(_fedexing_get(s, "legal_action", default=[]), dtype=np.float32)
        self._raw_legal_action = self.legal_action
        self.legal_actions = _fedexing_split_legal_actions(self.legal_action)
        self.sub_action_mask = _fedexing_normalize_sub_action_mask(_fedexing_get(s, "sub_action_mask", default={}))

        self.frame_state = frame = _fedexing_get(s, "frame_state", default={}) or {}
        self.n_frame = int(_fedexing_get(frame, "frame_no", "frameNo", default=0) or 0)
        self.map_state = bool(_fedexing_get(frame, "map_state", default=False))

        hero_states = _fedexing_get(frame, "hero_states", default=[]) or []
        enemy_camp = 1 - self.player_camp if self.player_camp in (0, 1) else -1
        self.hero_our = HeroInfo(hero_states, self.player_camp, self.player_id)
        self.hero_enemy = HeroInfo(hero_states, enemy_camp)
        for hero in (self.hero_our, self.hero_enemy):
            if hero.info.id:
                self.id2type[hero.info.id] = "hero"

        npc_states = _fedexing_get(frame, "npc_states", default=[]) or []
        self.organ_our = OrganInfo(npc_states, self.player_camp)
        self.organ_enemy = OrganInfo(npc_states, enemy_camp)
        for organ in (self.organ_our, self.organ_enemy):
            for actor in (organ.sub_tower, organ.crystal, organ.spring):
                if actor.id:
                    self.id2type[actor.id] = "organ"

        self.soldiers_our = SoldierInfo(npc_states, self.player_camp)
        self.soldiers_enemy = SoldierInfo(npc_states, enemy_camp)
        for soldiers in (self.soldiers_our, self.soldiers_enemy):
            for soldier in soldiers.merge:
                if soldier.id:
                    self.id2type[soldier.id] = "soldier"

        self.river_crab = None
        for npc in npc_states:
            if int(_fedexing_get_actor_value(npc, "config_id", default=0) or 0) == _RIVER_CRAB_CONFIG_ID:
                self.river_crab = ActorInfo(npc)
                if self.river_crab.id:
                    self.id2type[self.river_crab.id] = "river_crab"
                break

        cakes = _fedexing_get(frame, "cakes", default=None)
        self.cake_our = None
        self.cake_enemy = None
        if cakes is not None:
            self.cake_our = CakeInfo(cakes, self.player_camp)
            self.cake_enemy = CakeInfo(cakes, enemy_camp)
            if self.cake_our.position is None:
                self.cake_our = None
            if self.cake_enemy.position is None:
                self.cake_enemy = None

        bullets = _fedexing_get(frame, "bullets", default=[]) or []
        self.bullets_our = BulletsInfo(bullets, self.player_camp, self.id2type)
        self.bullets_enemy = BulletsInfo(bullets, enemy_camp, self.id2type)

        frame_action = _fedexing_get(frame, "frame_action", default={}) or {}
        self.deads = DeadsInfo(_fedexing_get(frame_action, "dead_action", default=[]) or [])
        return self


class HeroInfo:
    """One hero state, split into hero, actor, skill fields."""

    def __init__(
        self,
        hero_states: Optional[Iterable[dict]],
        camp: int,
        player_runtime_id: Optional[int] = None,
    ):
        self.hero_state = _fedexing_select_hero_state(hero_states, camp, player_runtime_id)
        s = self.hero_state or {}
        actor_state = _fedexing_get(s, "actor_state", default=s) or {}

        self.player_id = int(_fedexing_get(s, "player_id", default=0) or 0)
        self.info = ActorInfo(actor_state)
        self.skill = SkillInfo(_fedexing_get(s, "skill_state", default={}) or {})
        self.equip_state = _fedexing_get(s, "equip_state", default={}) or {}
        self.level = max(1, int(_fedexing_get(s, "level", default=1) or 1))
        self.exp = int(_fedexing_get(s, "exp", default=0) or 0)
        self.money = int(_fedexing_get(s, "money", default=0) or 0)
        self.money_total = int(_fedexing_get(s, "money_cnt", "moneyCnt", default=self.money) or 0)
        self.moneyCnt = self.money_total
        self.revive_time = int(_fedexing_get(s, "revive_time", default=0) or 0)

        self.killCnt = int(_fedexing_get(s, "kill_cnt", "killCnt", default=0) or 0)
        self.deadCnt = int(_fedexing_get(s, "dead_cnt", "deadCnt", default=0) or 0)
        self.assistCnt = int(_fedexing_get(s, "assist_cnt", "assistCnt", default=0) or 0)
        self.kill_cnt = self.killCnt
        self.dead_cnt = self.deadCnt
        self.assist_cnt = self.assistCnt
        self.kda = [self.killCnt, self.deadCnt, self.assistCnt]

        self.totalHurt = int(_fedexing_get(s, "total_hurt", "totalHurt", default=0) or 0)
        self.totalHurtToHero = int(_fedexing_get(s, "total_hurt_to_hero", "totalHurtToHero", default=0) or 0)
        self.totalBeHurtByHero = int(
            _fedexing_get(s, "total_be_hurt_by_hero", "totalBeHurtByHero", default=0) or 0
        )
        self.hurt_total = self.totalHurt
        self.hurt_hero_total = self.totalHurtToHero
        self.be_hurt_total = self.totalBeHurtByHero

        self.isInGrass = bool(_fedexing_get(s, "is_in_grass", "isInGrass", default=False))
        self.flag_in_grass = self.isInGrass
        self.canBuyEquip = bool(_fedexing_get(s, "canBuyEquip", default=False))
        self.flag_buy_equip = self.canBuyEquip
        self.passive_skill = _fedexing_get(s, "passive_skill", default=None)
        self.real_cmd = _fedexing_get(s, "real_cmd", default=[]) or []
        self.take_hurt_infos = _fedexing_get(s, "take_hurt_infos", default=[]) or []


class ActorInfo:
    """Actor-like state shared by heroes, soldiers, organs, and monsters."""

    def __init__(self, actor_state: Optional[dict]):
        s = _fedexing_actor_payload(actor_state)
        self.actor_state = s
        self.config_id = int(_fedexing_get(s, "config_id", "configId", default=0) or 0)
        self.id = int(_fedexing_get(s, "runtime_id", "runtimeId", default=0) or 0)
        self.runtime_id = self.id
        self.actor_type = _fedexing_get(s, "actor_type", default=-1)
        self.sub_type = _fedexing_get(s, "sub_type", default=-1)
        self.type = [_fedexing_actor_type_name(self.actor_type), _fedexing_actor_sub_type_name(self.sub_type)]
        self.camp = _fedexing_camp_index(_fedexing_get(s, "camp", default=0))
        self.behave_raw = _fedexing_get(s, "behav_mode", default=0)
        self.behave = _fedexing_behavior_name(self.behave_raw)
        self.position = cvt_position(_fedexing_get(s, "location", default=None))
        self.forward = cvt_position(_fedexing_get(s, "forward", default=None))
        self.hp = int(_fedexing_get(s, "hp", default=0) or 0)
        self.hp_max = max(1, int(_fedexing_get(s, "max_hp", "maxHp", default=1) or 1))

        values = _fedexing_get(s, "values", default={}) or {}
        self.values = values
        self.ep = int(_fedexing_get(s, "ep", default=_fedexing_get(values, "ep", default=0)) or 0)
        self.ep_max = max(1, int(_fedexing_get(s, "max_ep", default=_fedexing_get(values, "max_ep", default=1)) or 1))
        self.hp_recover = int(_fedexing_get(s, "hp_recover", default=_fedexing_get(values, "hp_recover", default=0)) or 0)
        self.ep_recover = int(_fedexing_get(s, "ep_recover", default=_fedexing_get(values, "ep_recover", default=0)) or 0)

        self.attack_range = int(_fedexing_get(s, "attack_range", default=0) or 0)
        self.attack_target = int(_fedexing_get(s, "attack_target", default=0) or 0)
        self.kill_bonus = int(_fedexing_get(s, "kill_income", "kill_bonus", default=0) or 0)
        self.hit_target_infos = [
            HitTargetInfo(item) for item in (_fedexing_get(s, "hit_target_info", default=[]) or [])
        ]
        self.sight_range = int(_fedexing_get(s, "sight_area", "sight_range", default=0) or 0)
        self.buff_state = _fedexing_get(s, "buff_state", default={}) or {}
        self.buff = BuffInfo(self.buff_state)
        self.abilities = _fedexing_get(s, "abilities", default=[]) or []
        self.camp_visible = _fedexing_get(s, "camp_visible", default=[]) or []
        self.hurt_hero_info = _fedexing_get(s, "hurt_hero_info", default=[]) or []


class SkillInfo:
    """Hero skill slots with structured attribute names."""

    def __init__(self, skill_state: Optional[dict]):
        self.skill_state = skill_state or {}
        raw_slots = list(_fedexing_get(self.skill_state, "slot_states", default=[]) or [])
        slots = [SlotInfo(raw_slots[i] if i < len(raw_slots) else None) for i in range(7)]
        self.slots = slots
        self.noraml_attack = slots[0]
        self.normal_attack = slots[0]
        self.first = slots[1]
        self.second = slots[2]
        self.thrid = slots[3]
        self.third = slots[3]
        self.recover = slots[4]
        self.flash = slots[5]
        self.back = slots[6]


class SlotInfo:
    """One entry in skill_state.slot_states."""

    def __init__(self, slot_state: Optional[dict]):
        s = slot_state or {}
        self.slot_state = s
        self.config_id = int(_fedexing_get(s, "configId", "config_id", default=0) or 0)
        self.configId = self.config_id
        self.slot_type_raw = _fedexing_get(s, "slot_type", default=0)
        self.slot_type = _fedexing_slot_type_name(self.slot_type_raw)
        self.level = int(_fedexing_get(s, "level", default=0) or 0)
        self.usable = bool(_fedexing_get(s, "usable", default=False))
        self.cd = int(_fedexing_get(s, "cooldown", "cd", default=0) or 0)
        self.cd_max = int(_fedexing_get(s, "cooldown_max", "cd_max", default=1 if slot_state is None else 0) or 0)
        self.cooldown = self.cd
        self.cooldown_max = self.cd_max
        self.count = int(_fedexing_get(s, "usedTimes", "used_times", default=0) or 0)
        self.hit_hero_count = int(_fedexing_get(s, "hitHeroTimes", "hit_hero_times", default=0) or 0)
        self.flag_used = int(_fedexing_get(s, "succUsedInFrame", "succ_used_in_frame", default=0) or 0)
        self.next_config_id = int(_fedexing_get(s, "nextConfigID", "next_config_id", default=0) or 0)
        self.combo_effect_time = int(_fedexing_get(s, "comboEffectTime", "combo_effect_time", default=0) or 0)


class OrganInfo:
    """Organ states for one camp."""

    def __init__(self, npc_states: Optional[Iterable[dict]], camp: int):
        self.sub_tower = ActorInfo(None)
        self.crystal = ActorInfo(None)
        self.spring = ActorInfo(None)
        for npc in npc_states or []:
            if _fedexing_camp_index(_fedexing_get_actor_value(npc, "camp", default=0)) != camp:
                continue
            sub_type = _fedexing_get_actor_value(npc, "sub_type", default=None)
            if _fedexing_sub_type_matches(sub_type, "ACTOR_SUB_TOWER", 21):
                self.sub_tower = ActorInfo(npc)
            elif _fedexing_sub_type_matches(sub_type, "ACTOR_SUB_CRYSTAL", 24):
                self.crystal = ActorInfo(npc)
            elif _fedexing_sub_type_matches(sub_type, "ACTOR_SUB_TOWER_SPRING", 23):
                self.spring = ActorInfo(npc)


class SoldierInfo:
    """Soldier states for one camp."""

    def __init__(self, npc_states: Optional[Iterable[dict]], camp: int):
        self.close: List[ActorInfo] = []
        self.remote: List[ActorInfo] = []
        self.car: List[ActorInfo] = []
        self.merge: List[ActorInfo] = []
        for npc in npc_states or []:
            if _fedexing_camp_index(_fedexing_get_actor_value(npc, "camp", default=0)) != camp:
                continue
            if not _fedexing_sub_type_matches(_fedexing_get_actor_value(npc, "sub_type", default=None), "ACTOR_SUB_SOLDIER", 11):
                continue
            info = ActorInfo(npc)
            if info.config_id in _SOLDIER_CLOSE_CONFIG_IDS:
                self.close.append(info)
            elif info.config_id in _SOLDIER_REMOTE_CONFIG_IDS:
                self.remote.append(info)
            elif info.config_id in _SOLDIER_CAR_CONFIG_IDS:
                self.car.append(info)
            else:
                self.remote.append(info)
        for group in (self.close, self.remote, self.car):
            self.merge.extend(group)


class CakeInfo:
    """Health-pack / shenfu state for one side."""

    def __init__(self, cakes: Iterable[dict], camp: int):
        self.cake_state = None
        self.config_id = 0
        self.position = None
        self.camp = None
        for cake in cakes or []:
            collider = _fedexing_get(cake, "collider", default={}) or {}
            location = _fedexing_get(collider, "location", default=None)
            x = _fedexing_position_component(location, "x", 0)
            cake_camp = int(float(x) > 0)
            if cake_camp != camp:
                continue
            self.cake_state = cake
            self.config_id = int(_fedexing_get(cake, "configId", "config_id", default=0) or 0)
            self.position = cvt_position(location)
            self.camp = cake_camp
            break


class BulletsInfo:
    """Bullets grouped by their source actor type."""

    def __init__(self, bullets: Optional[Iterable[dict]], camp: int, id2type: Dict[int, str]):
        self.id2type = id2type
        self.soldier: List[BulletInfo] = []
        self.hero: List[BulletInfo] = []
        self.organ: List[BulletInfo] = []
        self.merge: List[BulletInfo] = []
        for bullet in bullets or []:
            if _fedexing_camp_index(_fedexing_get(bullet, "camp", default=0)) != camp:
                continue
            source = int(_fedexing_get(bullet, "source_actor", default=0) or 0)
            source_type = self.id2type.get(source)
            if source_type is None:
                continue
            info = BulletInfo(bullet, source_type)
            if source_type == "soldier":
                self.soldier.append(info)
            elif source_type == "hero":
                self.hero.append(info)
            elif source_type == "organ":
                self.organ.append(info)
            self.merge.append(info)


class BulletInfo:
    """One bullet."""

    def __init__(self, bullet: dict, source_type: str):
        self.bullet_state = bullet
        self.id = int(_fedexing_get(bullet, "runtime_id", default=0) or 0)
        self.source_id = int(_fedexing_get(bullet, "source_actor", default=0) or 0)
        self.camp = _fedexing_camp_index(_fedexing_get(bullet, "camp", default=0))
        self.slot_type_raw = _fedexing_get(bullet, "slot_type", default=0)
        self.slot_type = _fedexing_slot_type_name(self.slot_type_raw)
        self.skill_id = int(_fedexing_get(bullet, "skill_id", default=0) or 0)
        self.position = cvt_position(_fedexing_get(bullet, "location", default=None))
        self.type = source_type


class DeadsInfo:
    """frame_action.dead_action grouped by dead actor type."""

    def __init__(self, deads: Optional[Iterable[dict]]):
        self.soldier: List[DeadInfo] = []
        self.organ: List[DeadInfo] = []
        self.hero: List[DeadInfo] = []
        self.river_crab: List[DeadInfo] = []
        self.merge: List[DeadInfo] = []
        for dead in deads or []:
            info = DeadInfo(dead)
            if info.death.type[1] == "ACTOR_SUB_SOLDIER":
                self.soldier.append(info)
            if info.death.type[0] == "ACTOR_ORGAN":
                self.organ.append(info)
            if info.death.type[0] == "ACTOR_HERO":
                self.hero.append(info)
            if info.death.config_id == _RIVER_CRAB_CONFIG_ID:
                self.river_crab.append(info)
        for group in (self.soldier, self.hero, self.organ, self.river_crab):
            self.merge.extend(group)


class DeadInfo:
    def __init__(self, dead: dict):
        self.dead_state = dead
        self.death = ActionActorInfo(_fedexing_get(dead, "death", default={}) or {})
        self.killer = ActionActorInfo(_fedexing_get(dead, "killer", default={}) or {})
        self.assist_set = [
            ActionActorInfo(item) for item in (_fedexing_get(dead, "assist_set", default=[]) or [])
        ]


class ActionActorInfo:
    """Actor subset used in frame actions."""

    def __init__(self, info: Optional[dict]):
        s = info or {}
        self.action_actor_state = s
        self.config_id = int(_fedexing_get(s, "config_id", "configId", default=0) or 0)
        self.id = int(_fedexing_get(s, "runtime_id", default=0) or 0)
        self.actor_type = _fedexing_get(s, "actor_type", default=-1)
        self.sub_type = _fedexing_get(s, "sub_type", default=-1)
        self.type = [_fedexing_actor_type_name(self.actor_type), _fedexing_actor_sub_type_name(self.sub_type)]
        self.camp = _fedexing_camp_index(_fedexing_get(s, "camp", default=0))


class BuffInfo:
    """actor_state.buff_state."""

    def __init__(self, buff_state: Optional[dict]):
        self.buff_state = buff_state or {}
        self.skills: List[BuffSkillInfo] = []
        self.skill_ids: List[int] = []
        self.marks: List[BuffMarkInfo] = []
        self.marks_ids: List[int] = []
        self.marks_layers: List[int] = []
        for item in _fedexing_get(self.buff_state, "buff_skills", default=[]) or []:
            info = BuffSkillInfo(item)
            self.skills.append(info)
            self.skill_ids.append(info.id)
        for item in _fedexing_get(self.buff_state, "buff_marks", default=[]) or []:
            info = BuffMarkInfo(item)
            self.marks.append(info)
            self.marks_ids.append(info.id)
            self.marks_layers.append(info.layer)


class BuffSkillInfo:
    def __init__(self, buff_skill: dict):
        self.buff_skill_state = buff_skill
        self.id = int(_fedexing_get(buff_skill, "configId", "config_id", default=0) or 0)
        self.start_time = _fedexing_get(buff_skill, "startTime", "start_time", default=0)
        self.times = int(_fedexing_get(buff_skill, "times", default=0) or 0)
        self.buff_skill_id = self.id


class BuffMarkInfo:
    def __init__(self, buff_mark: dict):
        self.buff_mark_state = buff_mark
        self.actor_id = int(_fedexing_get(buff_mark, "origin_actorId", "origin_actor_id", default=0) or 0)
        self.id = int(_fedexing_get(buff_mark, "configId", "config_id", default=0) or 0)
        self.layer = int(_fedexing_get(buff_mark, "layer", default=0) or 0)
        self.buff_mark_id = self.id


class HitTargetInfo:
    def __init__(self, info: dict):
        self.hit_target_info = info
        self.hit_target = int(_fedexing_get(info, "hit_target", default=0) or 0)
        self.skill_id = int(_fedexing_get(info, "skill_id", default=0) or 0)
        self.slot_type_raw = _fedexing_get(info, "slot_type", default=0)
        self.slot_type = _fedexing_slot_type_name(self.slot_type_raw)
        self.conti_hit_count = int(_fedexing_get(info, "conti_hit_count", default=0) or 0)


def cvt_position(position: Any):
    """Return [x, z] with red-side mirroring and unseen padding preserved."""

    flag = -1 if inverse_position else 1
    x = _fedexing_position_component(position, "x", 0)
    z = _fedexing_position_component(position, "z", 0)

    def convert(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(UNSEEN_PADDING)
        if value == UNSEEN_PADDING:
            return float(UNSEEN_PADDING)
        return float(flag * value)

    return np.asarray([convert(x), convert(z)], dtype=np.float32)


def cvt_camp_str2int(camp: Any):
    return _fedexing_camp_index(camp)


def info2dict(
    info: Any,
    skip_keys: bool = True,
    keys: Optional[Iterable[str]] = None,
    depth: int = 0,
    show: bool = False,
):
    """Convert an Info-like object into plain containers.

    The top-level return shape is (dict_value, printable_string).
    Nested recursive calls return only the nested value.
    """

    skip = set(keys or _SHOW_SKIP_KEYS)
    value = _fedexing_to_plain(info, skip_keys=skip_keys, skip=skip)
    if depth > 0:
        return value
    text = json.dumps(value, ensure_ascii=False, indent=2, default=_fedexing_json_default)
    if show:
        print(text)
    return value, text


def _fedexing_select_single_observation(observation: dict) -> dict:
    if not isinstance(observation, dict):
        return {}
    if "frame_state" in observation or "camp" in observation or "player_camp" in observation:
        return observation
    for value in observation.values():
        if isinstance(value, dict) and ("frame_state" in value or "camp" in value):
            return value
    return observation


def _fedexing_select_hero_state(
    hero_states: Optional[Iterable[dict]],
    camp: int,
    player_runtime_id: Optional[int],
):
    if not hero_states:
        return None
    by_camp = None
    for hero in hero_states:
        runtime_id = int(_fedexing_get_actor_value(hero, "runtime_id", default=0) or 0)
        if player_runtime_id and runtime_id == int(player_runtime_id):
            return hero
        if _fedexing_camp_index(_fedexing_get_actor_value(hero, "camp", default=0)) == camp:
            by_camp = hero
    return by_camp


def _fedexing_split_legal_actions(raw_la: np.ndarray):
    label_sizes = list(getattr(Config, "LABEL_SIZE_LIST", [12, 16, 16, 16, 16, 9]))
    expected = sum(label_sizes[:-1]) + label_sizes[0] * label_sizes[-1]
    if raw_la.size != expected:
        return []
    splits = []
    offset = 0
    for size in label_sizes[:-1]:
        splits.append(raw_la[offset : offset + size])
        offset += size
    splits.append(raw_la[offset:].reshape(label_sizes[0], label_sizes[-1]))
    return splits


def _fedexing_normalize_sub_action_mask(raw_mask: Any):
    if isinstance(raw_mask, dict):
        return {
            int(key): np.asarray(value, dtype=np.float32)
            for key, value in raw_mask.items()
        }
    if isinstance(raw_mask, list):
        return {
            idx: np.asarray(value, dtype=np.float32)
            for idx, value in enumerate(raw_mask)
        }
    return {}


def _fedexing_actor_payload(data: Optional[dict]):
    if not isinstance(data, dict):
        return {}
    nested = data.get("actor_state")
    if isinstance(nested, dict):
        return nested
    return data


def _fedexing_get_actor_value(data: Optional[dict], key: str, default=None):
    if not isinstance(data, dict):
        return default
    if key in data:
        return data[key]
    actor_state = data.get("actor_state")
    if isinstance(actor_state, dict) and key in actor_state:
        return actor_state[key]
    return default


def _fedexing_get(data: Any, *keys: str, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data:
            return data[key]
    return default


def _fedexing_camp_index(camp: Any):
    if camp is None:
        return -1
    if isinstance(camp, str):
        if camp == "PLAYERCAMP_MID":
            return -1
        if camp.startswith("PLAYERCAMP_"):
            suffix = camp.rsplit("_", 1)[-1]
            return int(suffix) - 1 if suffix.isdigit() else -1
        if camp.isdigit():
            return int(camp) - 1
        return -1
    try:
        camp = int(camp)
    except (TypeError, ValueError):
        return -1
    if camp in (1, 2):
        return camp - 1
    return -1


def _fedexing_position_component(position: Any, axis: str, default=0):
    if isinstance(position, dict):
        return position.get(axis, default)
    if isinstance(position, (list, tuple)):
        if axis == "x":
            return position[0] if len(position) >= 1 else default
        if axis == "z":
            if len(position) >= 3:
                return position[2]
            if len(position) >= 2:
                return position[1]
    return default


def _fedexing_actor_type_name(actor_type: Any):
    if isinstance(actor_type, str):
        return _ACTOR_TYPE_ALIASES.get(actor_type, actor_type)
    try:
        actor_type = int(actor_type)
    except (TypeError, ValueError):
        return "ACTOR_UNKNOWN"
    return _ACTOR_TYPE_NAMES.get(actor_type, f"ACTOR_UNKNOWN_{actor_type}")


def _fedexing_actor_sub_type_name(sub_type: Any):
    if isinstance(sub_type, str):
        return _ACTOR_SUB_TYPE_ALIASES.get(sub_type, sub_type)
    try:
        sub_type = int(sub_type)
    except (TypeError, ValueError):
        return "ACTOR_SUB_UNKNOWN"
    return _ACTOR_SUB_TYPE_NAMES.get(sub_type, f"ACTOR_SUB_UNKNOWN_{sub_type}")


def _fedexing_sub_type_matches(sub_type: Any, name: str, value: int):
    return _fedexing_actor_sub_type_name(sub_type) == name or sub_type == value


def _fedexing_behavior_name(behavior: Any):
    if isinstance(behavior, str):
        return behavior
    try:
        behavior = int(behavior)
    except (TypeError, ValueError):
        return ""
    return _BEHAVIOR_NAMES.get(behavior, f"State_{behavior}")


def _fedexing_slot_type_name(slot_type: Any):
    if isinstance(slot_type, str):
        return slot_type
    try:
        slot_type = int(slot_type)
    except (TypeError, ValueError):
        return "SLOT_SKILL_UNKNOWN"
    if slot_type == 16:
        return "SLOT_SKILL_VALID"
    return f"SLOT_SKILL_{slot_type}"


def _fedexing_to_plain(obj: Any, skip_keys: bool, skip: set):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {
            _fedexing_to_plain(key, skip_keys=False, skip=skip): _fedexing_to_plain(value, skip_keys, skip)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_fedexing_to_plain(item, skip_keys, skip) for item in obj]
    if hasattr(obj, "__dict__"):
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith("_"):
                continue
            if skip_keys and key in skip:
                continue
            result[key] = _fedexing_to_plain(value, skip_keys, skip)
        return result
    return obj


def _fedexing_json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
