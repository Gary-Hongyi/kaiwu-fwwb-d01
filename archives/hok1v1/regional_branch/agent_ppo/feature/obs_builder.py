"""Taichu-style structured observation builder for the current protocol.

This module is intentionally independent from the template 10-dim feature path.
Use it to validate the migrated feature design before switching Config and the
model input dimensions.
"""

import math
from typing import List

import numpy as np

from agent_ppo.conf.conf import Args
from agent_ppo.feature.unpack_state_dict import (
    ActorInfo,
    BulletInfo,
    CakeInfo,
    HeroInfo,
    Info,
    SlotInfo,
)


def _fedexing_clip(x, mn, mx):
    return max(min(x, mx), mn)


def _fedexing_fix(x):
    return math.floor(x) if x > 0 else math.ceil(x)


def _fedexing_safe_ratio(num, den):
    return 0.0 if den == 0 else float(num) / float(den)


def _fedexing_set_one_hot(vec, idx):
    if -len(vec) <= idx < len(vec):
        vec[idx] = 1.0


class FedExingObsBuilder:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def build_observation(self, info: Info, need_mask=False):
        self.info = info
        self.n_frame = info.n_frame
        self.pos = info.hero_our.info.position
        if self.last_money[0] is None:
            self.last_money = [info.hero_our.money_total, info.hero_enemy.money_total]

        x_hero_our = self.process_hero(info.hero_our, False, info.organ_enemy.sub_tower)
        x_hero_enemy = self.process_hero(info.hero_enemy, True, info.organ_our.sub_tower)
        x_soldier_our, mask_soldier_our = self.process_soldier(info.soldiers_our.merge, info.organ_enemy.sub_tower)
        x_soldier_enemy, mask_soldier_enemy = self.process_soldier(
            info.soldiers_enemy.merge, info.organ_our.sub_tower
        )
        mask_soldier = mask_soldier_our + mask_soldier_enemy
        x_river_crab, mask_river_crab = self.process_river_crab()
        x_sub_tower_our = self.process_sub_tower(info.organ_our.sub_tower, info.cake_our, False)
        x_sub_tower_enemy = self.process_sub_tower(info.organ_enemy.sub_tower, info.cake_enemy, True)

        x_all_units = (
            x_hero_our
            + x_hero_enemy
            + x_soldier_our
            + x_soldier_enemy
            + x_river_crab
            + x_sub_tower_our
            + x_sub_tower_enemy
        )
        assert len(x_all_units) == Args.DIM_ALL_UNITS, f"{len(x_all_units)=}, {Args.DIM_ALL_UNITS=}"

        x_bullet, mask_bullet = self.process_bullets()
        x = np.asarray(x_all_units + x_bullet, np.float32)
        assert x.shape == (Args.DIM_ALL,), f"{x.shape=}, {Args.DIM_ALL=}"
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if need_mask:
            return x, (mask_soldier, mask_bullet, mask_river_crab)
        return x

    def _fedexing_warn(self, message):
        if self.logger is not None:
            self.logger.warning(message)
        else:
            print(message)

    def process_bullet(self, bullet: BulletInfo):
        x_slot = [0.0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        slots = Args.BULLET_SLOT
        idx = slots.index(bullet.slot_type) if bullet.slot_type in slots else len(slots) - 1
        x_slot[idx] = 1.0
        return x_slot + self.process_position(bullet.position)

    def process_bullets(self):
        hero_bullets = sorted(self.info.bullets_enemy.hero, key=lambda b: math.dist(b.position, self.pos))
        hero_bullets = hero_bullets[: Args.BULLET_MAX_NUM - 1]
        x_bullets = []
        masks = [0.0] * Args.BULLET_MAX_NUM
        for i, bullet in enumerate(hero_bullets):
            x_bullets += self.process_bullet(bullet)
            masks[i] = 1.0
        x_bullets += [0.0] * ((Args.BULLET_MAX_NUM - 1) * Args.DIM_BULLET - len(x_bullets))

        organ_bullets = sorted(self.info.bullets_enemy.organ, key=lambda b: math.dist(b.position, self.pos))
        if organ_bullets:
            x_bullets += self.process_bullet(organ_bullets[0])
            masks[-1] = 1.0
        x_bullets += [0.0] * (Args.DIM_BULLETS - len(x_bullets))
        return x_bullets, masks

    def process_gsl_bullet(self, bullet: BulletInfo):
        x_gsl_bullet = [0.0] * (Args.DIM_GSL_BULLET - Args.DIM_DISTANCE)
        slots = Args.BULLET_GSL_SLOT
        idx = slots.index(bullet.slot_type) if bullet.slot_type in slots else len(slots) - 1
        x_gsl_bullet[idx] = 1.0
        return x_gsl_bullet + self.process_position(bullet.position)

    def process_hero(self, hero: HeroInfo, is_enemy: bool, enemy_sub_tower: ActorInfo):
        x_unit = self.process_unit(hero.info)

        x_hero_id = [Args.HERO_ID_VALUE_DICT.get(hero.info.config_id, 0.0)]

        behaves = Args.HERO_BEHAVE
        x_behave = [0.0] * (len(behaves) + 1)
        idx = behaves.index(hero.info.behave) if hero.info.behave in behaves else -1
        x_behave[idx] = 1.0

        unit_size, max_size = Args.EP_UNIT_SIZE, Args.EP_MAX_SIZE
        ep_dim = int(max_size / unit_size) + 1
        x_ep = [0.0] * (1 + ep_dim)
        x_ep[0] = _fedexing_safe_ratio(hero.info.ep, hero.info.ep_max)
        _fedexing_set_one_hot(x_ep, 1 + min(int(math.floor(hero.info.ep / unit_size)), ep_dim - 1))

        s = hero.skill
        x_summoner_skill = [0.0] * Args.DIM_SUMMONER_SKILL
        if s.flash.config_id in Args.SUMMONER_SKILL_ID:
            x_summoner_skill[Args.SUMMONER_SKILL_ID.index(s.flash.config_id)] = 1.0
        else:
            x_summoner_skill[-1] = 1.0
        x_skill1, x_skill2, x_skill3, x_skill_flash, x_skill_recover = (
            self.process_skill(x) for x in [s.first, s.second, s.thrid, s.flash, s.recover]
        )

        x_level = [0.0] * Args.LEVEL_MAX
        _fedexing_set_one_hot(x_level, min(max(hero.level - 1, 0), Args.LEVEL_MAX - 1))
        x_money = self.process_money(hero.money_total, is_enemy)
        x_grass = [float(hero.flag_in_grass)]
        x_organ = self.process_organ(hero, enemy_sub_tower)

        x_buff = [0.0] * Args.DIM_BUFF
        for buff_id in hero.info.buff.skill_ids:
            if buff_id in Args.BUFFS:
                x_buff[Args.BUFFS.index(buff_id)] = 1.0

        x = (
            x_hero_id
            + x_behave
            + x_ep
            + x_summoner_skill
            + x_skill1
            + x_skill2
            + x_skill3
            + x_skill_flash
            + x_skill_recover
            + x_level
            + x_money
            + x_grass
            + x_organ
            + x_buff
            + x_unit
        )
        assert len(x) == Args.DIM_HERO, f"{len(x)=}, {Args.DIM_HERO=}"
        return x

    def process_money(self, money, is_enemy: bool):
        idx = int(is_enemy)
        if self.last_money[idx] is None:
            self.last_money[idx] = money
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        unit_size, max_size = Args.MONEY_UNIT_SIZE, Args.MONEY_MAX_SIZE
        money_dim = int(max_size / unit_size) + 1
        x_money = [0.0] * (2 + money_dim)
        if 0 < delta < 20:
            x_money[-2] = 1.0
        else:
            _fedexing_set_one_hot(x_money, max(min(int(math.floor(delta / unit_size)), money_dim - 1), 0))
        x_money[-1] = min(money / 10000.0, 1.0)
        return x_money

    def process_organ(self, hero: HeroInfo, enemy_sub_tower: ActorInfo):
        tower = enemy_sub_tower
        x_organ = [0.0] * 2
        x_organ[0] = float(math.dist(hero.info.position, tower.position) <= tower.attack_range)
        x_organ[1] = float(hero.info.id == tower.attack_target)
        return x_organ

    def process_position(self, position):
        p = np.asarray(position, dtype=np.float32)
        if p[0] == Info.UNSEEN_PADDING:
            return [0.0] * Args.DIM_DISTANCE

        unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = (max_size / unit_size + 1) / 2
        rpos = [
            int(_fedexing_clip(_fedexing_fix((p[i] - self.pos[i]) / unit_size), -max_idx, max_idx) + max_idx)
            for i in range(2)
        ]
        x_dis = _fedexing_clip(math.dist(p, self.pos) / (max_size / 2), 0.0, 1.0)
        rpos_dim = int(2 * max_idx + 1)
        x_rpos = [0.0] * (rpos_dim * 2 + 1)
        x_rpos[rpos[0]] = 1.0
        x_rpos[rpos[1] + rpos_dim] = 1.0
        x_rpos[-1] = x_dis

        unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
        max_size = Args.WHOLE_DISTANCE_MAX_SIZE
        wpos_dim = int(max_size / unit_size)
        wpos = [
            int(_fedexing_clip(math.floor((p[i] + max_size / 2) / unit_size), 0, wpos_dim - 1))
            for i in range(2)
        ]
        x_wratio = [
            _fedexing_clip((p[i] + max_size / 2) / unit_size - wpos[i], -1.0, 1.0)
            for i in range(2)
        ]
        x_wpos = [0.0] * (wpos_dim * 2 + 2)
        x_wpos[wpos[0]] = 1.0
        x_wpos[wpos[1] + wpos_dim] = 1.0
        x_wpos[-2] = x_wratio[0]
        x_wpos[-1] = x_wratio[1]
        return x_rpos + x_wpos

    def process_river_crab(self):
        behave = Args.RIVER_CRAB_BEHAVE
        x_behave = [0.0] * (len(behave) + 1)
        crab = self.info.river_crab
        mask = [0.0]
        if crab is None:
            return [0.0] * Args.DIM_RIVER_CRAB, mask
        mask[0] = 1.0
        idx = behave.index(crab.behave) if crab.behave in behave else -1
        x_behave[idx] = 1.0
        x = x_behave + self.process_unit(crab)
        assert len(x) == Args.DIM_RIVER_CRAB, f"{len(x)=}, {Args.DIM_RIVER_CRAB=}"
        return x, mask

    def process_skill(self, info: SlotInfo):
        unit_size, max_size = Args.CD_UNIT_SIZE, Args.CD_MAX_SIZE
        cd_dim = int(max_size / unit_size) + 2
        x_cd = [0.0] * (2 + cd_dim)
        x_cd[0] = _fedexing_safe_ratio(info.cd, info.cd_max)
        _fedexing_set_one_hot(x_cd, 1 + min(int(math.ceil(info.cd / unit_size)), cd_dim - 1))
        if not info.usable and info.level == 0:
            x_cd[-1] = 1.0
        return x_cd

    def process_soldier(self, soldiers: List[ActorInfo], opposed_sub_tower: ActorInfo):
        soldiers = sorted(soldiers, key=lambda s: s.runtime_id)
        soldiers = soldiers[: Args.SOLDIER_MAX_NUM]
        behavers = Args.SOLDIER_BEHAVE
        config_ids = Args.SOLDIER_CONFIG_ID
        x_soldiers = []
        mask_soldiers = [0.0] * Args.SOLDIER_MAX_NUM
        for i, soldier in enumerate(soldiers):
            x_soldier = [0.0] * (Args.DIM_SOLDIER - Args.DIM_UNIT)
            base_idx = 0
            idx = behavers.index(soldier.behave) if soldier.behave in behavers else -1
            x_soldier[base_idx + idx] = 1.0
            base_idx += len(behavers) + 1
            for j, ids in enumerate(config_ids):
                if soldier.config_id in ids:
                    x_soldier[base_idx + j] = 1.0
            base_idx += len(config_ids)
            if math.dist(soldier.position, opposed_sub_tower.position) <= opposed_sub_tower.attack_range:
                x_soldier[base_idx] = 1.0
            if soldier.id == opposed_sub_tower.attack_target:
                x_soldier[base_idx + 1] = 1.0
            x_soldiers += x_soldier + self.process_unit(soldier)
            mask_soldiers[i] = 1.0
        x_soldiers += [0.0] * (Args.DIM_SOLDIERS - len(x_soldiers))
        return x_soldiers, mask_soldiers

    def process_sub_tower(self, sub_tower: ActorInfo, cake: CakeInfo, is_enemy: bool):
        x_target = [0.0] * 5
        if sub_tower.attack_target == 0:
            x_target[0] = 1.0
        else:
            actor_type = self.info.id2type.get(sub_tower.attack_target)
            if actor_type == "hero":
                x_target[1] = 1.0
            elif actor_type == "soldier":
                x_target[2] = 1.0
            elif actor_type is not None:
                self._fedexing_warn(f"sub_tower target type={actor_type} not in [hero, soldier]")
            else:
                self._fedexing_warn(f"sub_tower target={sub_tower.attack_target} not in id2type")

        x_target[3] = float(cake is not None)
        enemy_idx = int(is_enemy)
        if cake is not None:
            self.next_cake_frame[enemy_idx] = self.n_frame + 76 * 30
            x_target[4] = 0.0
        else:
            x_target[4] = min((self.next_cake_frame[enemy_idx] - self.n_frame) / (75 * 30), 1.0)
        x = x_target + self.process_unit(sub_tower)
        assert len(x) == Args.DIM_SUB_TOWER, f"{len(x)=}, {Args.DIM_SUB_TOWER=}"
        return x

    def process_unit(self, info: ActorInfo):
        x_pos = self.process_position(info.position)

        unit_size, max_size = Args.HP_UNIT_SIZE, Args.HP_MAX_SIZE
        hp_dim = int(max_size / unit_size) + 2
        x_hp = [0.0] * (1 + hp_dim)
        x_hp[0] = _fedexing_safe_ratio(info.hp, info.hp_max)
        _fedexing_set_one_hot(x_hp, 1 + min(int(math.ceil(info.hp / unit_size)), hp_dim - 1))

        x_mark = [0.0] * Args.DIM_MARK
        now_feature_idx = 0
        for mark_id, max_layer in Args.MARK_ID_LAYERS.items():
            if mark_id in info.buff.marks_ids:
                index = info.buff.marks_ids.index(mark_id)
                layer = max(min(info.buff.marks_layers[index], max_layer), 0)
                x_mark[now_feature_idx + layer] = 1.0
            now_feature_idx += max_layer + 1
        return x_hp + x_mark + x_pos

    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]
