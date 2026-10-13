"""
对环境返回的state_dicts进行解包, 转为易用的类形式,
从属性中直接获取相关信息值

第一次出现时间
river_crab: frame=920, time=30.7s
cake: frame=1778, time=59.3s
car: frame=6254, time=208.5s
河蟹复活时间: 
river crab: 复活时间1878帧, 1min 2s, 
河蟹第二次死亡有102帧, 3s多处于'State_Dead'状态, 然后出现在dead_action中 (WHY?)

behave包含
[
    'State_Dead': 阵亡时, (单位死亡后会一直处于该状态)
    'State_Idle': 空闲 (原地不动, 或者在视野外),
    'Direction_Move': 移动,
    'State_GameOver': 游戏终止, (这个状态无关紧要)
    'Attack_Move': 移动攻击, (防御塔一直处于该状态)
    'Normal_Attack': 普通攻击,
    'State_Revive': 恢复中 (在泉水里回血回蓝)
    'State_Auto': 自动状态 (只有河蟹具有这个状态信息, 表示自动移动)
    'Attack_Path': 小兵在攻击轨迹上
    'State_Born': 河蟹出现时候
    'UseSkill_1': 英雄使用一技能
    'UseSkill_2': 英雄使用二技能
    'UseSkill_3': 英雄使用三技能
]

passive_skill

通用buff_skills (7,)
90015: 可能是泉水的回复buff
10000: 点回复技能时候产生的buff (1.2s先消失)
10010: 回复技能产生的恢复buff (5.8s)
11001: 可能是加速buff
11002: 可能是减速buff
11010: 可能是净化buff (狄仁杰2技能)
11111: 可能某种通用buff (虞姬)
一些未知buff (6,):
911220, 911290, 914110, 914210, 914211, 914250

英雄的buff编号规律:
    前三位是英雄的id, 169, 173, 174分别为后羿, 李元芳, 虞姬
    第四位如果是0~3则表示被动,1,2,3技能的buff

后羿 (buff_skills)
buff_skills (10,):
169000, 169010, 169020, 169040,
169100,
169900, 169901, 169910, 169920, 169963,
buff_marks及其对应层数 (2,)
16900 (3层), 16901 (2层)

李元芳相关: 
buff_skills (20,):
173000, 173040,
173101, 173110, 173120, 173150, 173151, 173152, 173153, 173154, 173155, 173159, 173160, 173170, 173173,
173250,
173920, 173950, 173990, 173999,
buff_marks (2,):
17300 (4层), 17310 (1层)

虞姬相关:
buff_skills (10,)
174000, 174010, 174090,
174100,
174250, 174260,
174360,
174910, 174920, 174950,
buff_marks (0,)

hit_target_info:
游戏开始158 (5s)内会发现每个英雄自身会以自己为target产生slot_id=SLOT_SKILL_VALID的攻击, 不清楚是什么东西, 也可能是开始时泉水的回血?
从蓝色方阵营视角可以看到双方英雄对对方阵营单位的攻击信息, 包含小兵, 英雄, 防御塔的所有攻击信息
因此要判断当前英雄攻击的英雄, 防御塔的奖励需要从蓝方阵营的info.hero_enemy.info.hit_target_infos中查看每个hit_target_info,
通过id2type[hit_target_info.hit_target]转化为'hero', 'soldier', 'organ', 'crab'

防御塔:
子弹信息
    防御塔的子弹信息为来源为防御塔的id, slot_type="SLOT_SKILL_0"
攻击目标
    通过ActorState中的attack_target确定攻击目标的runtime_id, 在通过id2type找到对应的类别

2024.10.21: 官方修复buff_marks的key顺序
"""
from typing import List
import json
import numpy as np
from agent_diy.conf.conf import Config, GameConfig

inverse_position = False    # 是否翻转


TOWER_SUB_TYPES = {"ACTOR_SUB_TOWER", 21}
CRYSTAL_SUB_TYPES = {"ACTOR_SUB_CRYSTAL", 24}
SPRING_SUB_TYPES = {"ACTOR_SUB_TOWER_SPRING", 22, 23}
SOLDIER_SUB_TYPES = {"ACTOR_SUB_SOLDIER", 1}

def actor(obj):
    return obj.get("actor_state", obj)

def camp_to_zero(camp):
    if isinstance(camp, str):
        if camp == "PLAYERCAMP_MID":
            return -1
        if camp.endswith("_1") or camp == "1":
            return 0
        if camp.endswith("_2") or camp == "2":
            return 1
        return -1
    if isinstance(camp, int):
        return camp - 1 if camp in (1, 2) else camp
    return -1

def val(d, key, default=0):
    if key in d:
        return d[key]
    values = d.get("values", {})
    return values.get(key, default) if isinstance(values, dict) else default

def first(d, keys, default=0):
    for key in keys:
        if key in d:
            return d[key]
    return default

class Info:
    UNSEEN_PADDING = 100000    # 对无法看到位置的填充值 (与默认填充值保持一致)
    """将环境返回信息进行解包(将key转化为属性)
    Args:
        state_dict: [dict] 环境返回的某一方信息state_dicts[0/1]
        reverse_camp1: [bool] 是否将红色方阵营获得的所有位置及朝向进行反转

    由于bullet可能是由当前不在视野中的单位发出的, 但是之前见过,
    因此需要记录下当前全部的id类型, 使用id2type进行记忆
    """
    def __init__(self, state_dict=None, reverse_camp1=True):
        # 将能够发子弹的id与类别进行对应
        # 类别包含hero, organ, soldier, crab (用于子弹从属, 攻击目标判定)
        self.id2type = {}
        self.state_dict = state_dict
        self.reverse_camp1 = reverse_camp1
        if state_dict is not None:
            self.update(state_dict)
    
    def reset(self):
        self.id2type = {}
    
    def update(self, state_dict: dict):
        s = self.state_dict = state_dict
        self.player_id = s.get('player_id', 0)    # 玩家编号(用于确定使用的英雄)
        self.player_camp = camp_to_zero(s.get('player_camp', s.get('camp', 1)))    # 0/1 蓝方/红方

        ### 将红色方阵营所有位置及方向进行反转 ###
        global inverse_position
        inverse_position = (self.player_camp == 1 and self.reverse_camp1)

        self.game_id = s.get('env_id', '')    # 游戏编号(每局编号不同)
        self.legal_action = s.get('legal_action', [])    # 全体合法动作 12+16*4+12*9 (最后一维由于与第一维相关, 因此是一个(12, 9)的矩阵)
        if len(self.legal_action) and GameConfig.debug_agent:  # 如果是debug且legal_action存在, 对其进行划分
            cum_label_sizes = np.cumsum(Config.LABEL_SIZE_LIST[:-1])
            self.legal_actions = np.split(self.legal_action, cum_label_sizes)  # [(12,), (16,)*4, (12*9,)]
            self.legal_actions[-1] = self.legal_actions[-1].reshape(12, 9)
        else:  # 如果是eval状态时, legal_action可能是[], 即无法操控, train_workflow不会进行预测
            self.legal_actions = []
        self.sub_action_mask = s.get('sub_action_mask', {})    # 第一维动作对整个动作的mask shape=(12, 6)
        self.frame_state = frame = s.get('frame_state', {})    # 帧内状态信息
        self.n_frame = frame.get('frame_no', frame.get('frameNo', 0))    # 帧数
        self.map_state: bool = frame.get('map_state', False)    # True: 已坍塌, False: 未坍塌 (不会变成未坍塌)

        ### 解包英雄信息HeroState ###
        hero_states = frame.get('hero_states', [])
        self.hero_our = HeroInfo(hero_states, self.player_camp)
        self.hero_enemy = HeroInfo(hero_states, 1-self.player_camp)
        for hero in [self.hero_our, self.hero_enemy]:
            self.id2type[hero.info.id] = 'hero'

        ### 以下处理npc信息 ###
        npc_states = frame.get('npc_states', [])
        ### 解包防御塔信息 ###
        self.organ_our = OrganInfo(npc_states, self.player_camp)
        self.organ_enemy = OrganInfo(npc_states, 1-self.player_camp)
        for organ in [self.organ_our, self.organ_enemy]:
            self.id2type[organ.sub_tower.id] = 'organ'
            self.id2type[organ.crystal.id] = 'organ'
            self.id2type[organ.spring.id] = 'organ'
        ### 解包小兵信息 ###
        self.soldiers_our = SoldierInfo(npc_states, self.player_camp)
        self.soldiers_enemy = SoldierInfo(npc_states, 1-self.player_camp)
        for soldiers in [self.soldiers_our, self.soldiers_enemy]:
            for l in [soldiers.close, soldiers.remote, soldiers.car]:
                for s in l: self.id2type[s.id] = 'soldier'
        ### 河蟹信息 ###
        self.river_crab = None
        for s in npc_states:
            if s.get('config_id') == 6827:    # 河蟹id
                self.river_crab = ActorInfo(s)
                self.id2type[self.river_crab.id] = 'river_crab'
        ### 血包信息 ###
        cakes = frame.get('cakes')    # 血包可能没有出现
        self.cake_our, self.cake_enemy = None, None
        if cakes is not None:
            self.cake_our = CakeInfo(cakes, self.player_camp)
            self.cake_enemy = CakeInfo(cakes, 1-self.player_camp)
            if self.cake_our.position is None: self.cake_our = None    # 可能没有对应塔后的血包
            if self.cake_enemy.position is None: self.cake_enemy = None
        ### 子弹信息 ###
        bullets = frame.get('bullets')    # 子弹可能没有出现
        self.bullets_our = BulletsInfo(bullets, self.player_camp, self.id2type)
        self.bullets_enemy = BulletsInfo(bullets, 1-self.player_camp, self.id2type)
        # TODO: frame_action 死亡事件 (补河蟹, 补兵, 杀死英雄)
        self.deads = DeadsInfo(frame.get('frame_action', {}).get('dead_action', []))    # 死亡事件

class HeroInfo:
    """将state_dicts[0/1]['frame_state']['hero_states']列表取出对应阵营的HeroState"""
    def __init__(self, hero_states, camp):
        self.hero_state = None
        for s in hero_states:
            if camp_to_zero(actor(s).get('camp', s.get('camp'))) == camp:
                self.hero_state = s
        if self.hero_state is None:
            self.hero_state = {}
            if hero_states:
                self.hero_state = hero_states[0]
        s = self.hero_state
        self.player_id: int = s['player_id']    # 玩家编号(控制该英雄的玩家)
        self.info = ActorInfo(actor(s))    # 角色
        self.skill = SkillInfo(s.get('skill_state', actor(s).get('skill_state', {})))    # 技能
        self.equip_state = s.get('equip_state', actor(s).get('equip_state', {}))    # 装备(未解包)
        # self.buff_state = s.get('buff_state', {})    # buff(未解包)与ActorInfo中的buff一模一样, 跳过
        self.level = s.get('level', actor(s).get('level', 1))    # 英雄等级
        self.exp = s.get('exp', actor(s).get('exp', 0))    # 当前经验值(累计经验值需要加上之前等级的)
        self.money = s.get('money', actor(s).get('money', 0))    # 当前金币(不是总金币)
        self.money_total = first(s, ['money_cnt', 'moneyCnt'], self.money)    # 总金币
        self.revive_time = s.get('revive_time', 0)    # 复活时间
        self.kda = [first(s, ['kill_cnt', 'killCnt'], 0), first(s, ['dead_cnt', 'deadCnt'], 0), first(s, ['assist_cnt', 'assistCnt'], 0)]
        self.hurt_total = first(s, ['total_hurt', 'totalHurt'], 0)    # 总输出
        self.hurt_hero_total = first(s, ['total_hurt_to_hero', 'totalHurtToHero'], 0)    # 对英雄总输出
        self.be_hurt_total = first(s, ['total_be_hurt_by_hero', 'totalBeHurtByHero'], 0)    # 承受英雄总伤害
        self.flag_in_grass = first(s, ['is_in_grass', 'isInGrass'], False)    # 是否在草丛中
        self.flag_buy_equip = first(s, ['can_buy_equip', 'canBuyEquip'], False)    # 是否可以买装备
        self.passive_skill = s.get('passive_skill', [])

class ActorInfo:
    """处理actor_state信息"""
    def __init__(self, actor_state):
        s = self.actor_state = actor_state
        self.config_id: int = s.get('config_id', s.get('configId', 0))    # 如果是英雄, 则是英雄编号
        self.id: int = s.get('runtime_id', 0)    # 唯一id
        self.type: List[str] = [s.get('actor_type'), s.get('sub_type')]
        self.camp = cvt_camp_str2int(s.get('camp', 1))
        self.behave: str = s.get('behav_mode', 'State_Idle')    # 当前执行的行为
        self.position = cvt_position(s.get('location', {'x': Info.UNSEEN_PADDING, 'z': Info.UNSEEN_PADDING}))    # 位置
        self.forward = cvt_position(s.get('forward', {'x': 0, 'z': 0}))    # 朝向
        self.hp = s.get('hp', 0)    # 当前生命值
        self.hp_max = s.get('max_hp', 1) or 1    # 最大生命值
        ### 数值信息提取(只提取部分重要信息) ###
        self.values = v = s.get('values', s)    # 数值信息
        self.ep = val(v, 'ep', 0)    # 当前法力值
        self.ep_max = val(v, 'max_ep', 1) or 1    # 当前法力值
        self.hp_recover = val(v, 'hp_recover', 0)    # 生命恢复量
        self.ep_recover = val(v, 'ep_recover', 0)    # 法术恢复量
        self.attack_range = s.get('attack_range', 0)    # 攻击范围
        self.attack_target = s.get('attack_target', 0)    # 攻击目标编号 (0为没有目标)
        self.kill_bonus = s.get('kill_income', 0)    # 被击杀奖励
        # 只能从蓝方阵营才能看到双方英雄的HitTargetInfo信息
        self.hit_target_infos = [HitTargetInfo(x) for x in s.get('hit_target_info', [])]
        self.sight_range = s.get('sight_area', 0)    # 视野范围
        # buff configId: 133001(从头开始就有), 90015(可能是泉水回血)
        self.buff_state = s.get('buff_state', {})    # buff状态信息(TODO: 具体内容未知)
        self.buff = BuffInfo(self.buff_state)    # buff信息

class SkillInfo:
    """处理skill_state信息"""
    def __init__(self, skill_state):
        self.skill_state = skill_state
        slots = list((self.skill_state or {}).get('slot_states', []))
        while len(slots) < 7:
            slots.append({})
        self.noraml_attack = SlotInfo(slots[0])    # 普攻
        self.first = SlotInfo(slots[1])     # 一技能
        self.second = SlotInfo(slots[2])    # 二技能
        self.thrid = SlotInfo(slots[3])     # 三技能
        self.recover = SlotInfo(slots[4])    # 回复
        self.flash = SlotInfo(slots[5])    # 闪现
        self.back = SlotInfo(slots[6])    # 回城

class SlotInfo:
    """slot_states列表中的每个元素, 每一个可用按钮"""
    def __init__(self, slot_state):
        s = self.slot_state = slot_state
        self.config_id = s.get('configId', s.get('config_id', 0))
        self.slot_type = s.get('slot_type', 'SLOT_SKILL_0')    # 技能名称 "SLOT_SKILL_x" (0~6)
        self.level = s.get('level', 0)    # 技能等级 (如果是技能)
        self.usable = s.get('usable', False)    # 是否可用
        self.cd = s.get('cooldown', 0)    # 冷却 (单位: ms)
        self.cd_max = s.get('cooldown_max', 1) or 1    # 最大冷却时间 (单位: ms)
        self.count = s.get('usedTimes', s.get('used_times', 0))    # 使用次数
        self.hit_hero_count = s.get('hitHeroTimes', s.get('hit_hero_times', 0))    # 打中对方英雄次数
        self.flag_used = s.get('succUsedInFrame', s.get('succ_used_in_frame', 0))    # 当前帧成功使用

class OrganInfo:
    """npc_states中找到对应阵营的防御塔信息"""
    def __init__(self, npc_states, camp):
        self.sub_tower = None
        self.crystal = None
        self.spring = None
        for s in npc_states:
            if s.get('sub_type') not in (TOWER_SUB_TYPES | CRYSTAL_SUB_TYPES | SPRING_SUB_TYPES): continue
            if camp_to_zero(s.get('camp')) != camp: continue
            sub_type = s['sub_type']
            if sub_type in TOWER_SUB_TYPES:
                self.sub_tower = ActorInfo(s)
            elif sub_type in CRYSTAL_SUB_TYPES:
                self.crystal = ActorInfo(s)
            elif sub_type in SPRING_SUB_TYPES:
                self.spring = ActorInfo(s)
        fallback = ActorInfo({
            'runtime_id': 0,
            'config_id': 0,
            'camp': camp + 1,
            'location': {'x': Info.UNSEEN_PADDING, 'z': Info.UNSEEN_PADDING},
            'forward': {'x': 0, 'z': 0},
            'hp': 0,
            'max_hp': 1,
            'buff_state': {'buff_skills': [], 'buff_marks': []},
        })
        self.sub_tower = self.sub_tower or fallback
        self.crystal = self.crystal or fallback
        self.spring = self.spring or fallback
            
class SoldierInfo:
    """npc_states中找到对应阵营的小兵信息"""
    def __init__(self, npc_states, camp):
        self.close: List[ActorInfo] = []    # 近战小兵
        self.remote: List[ActorInfo] = []    # 远程小兵
        self.car: List[ActorInfo] = []    # 炮车
        self.merge: List[ActorInfo] = []    # 全部信息
        for s in npc_states:
            if s.get('sub_type') not in SOLDIER_SUB_TYPES and s.get('config_id') not in [6800,6801,6802,6803,6804,6805]: continue
            if camp_to_zero(s.get('camp')) != camp: continue
            if s['config_id'] in [6801, 6804]:    # 蓝方/红方近战编号
                self.close.append(ActorInfo(s))
            elif s['config_id'] in [6800, 6803]:    # 蓝方/红方远程编号
                self.remote.append(ActorInfo(s))
            elif s['config_id'] in [6802, 6805]:    # 蓝方/红方炮车编号 (TODO: 检验下是不是炮车)
                self.car.append(ActorInfo(s))
        for l in [self.close, self.remote, self.car]:
            self.merge.extend(l)

class CakeInfo:
    """cakes中找到对应阵营的血包信息"""
    def __init__(self, cakes, camp):
        self.position, self.camp = None, None
        for cake in cakes:
            collider = cake.get('collider', {})
            location = collider.get('location', {})
            pos_x = location.get('x', 0)
            camp_ = int(pos_x > 0)
            if camp == camp_:
                self.position, self.camp = cvt_position(location), camp_

class BulletsInfo:
    """bullets中找到对应阵营的子弹信息"""
    def __init__(self, bullets, camp, id2type):
        self.id2type = id2type
        self.soldier: List[BulletInfo] = []    # 小兵子弹
        self.hero: List[BulletInfo] = []    # 英雄子弹
        self.organ: List[BulletInfo] = []    # 防御塔子弹
        self.merge = []    # 全部信息
        # TODO: 找到更多子弹信息, 英雄普攻子弹, 技能子弹, 防御塔子弹
        if bullets is None: return
        for bullet in bullets:
            if camp_to_zero(bullet.get('camp')) != camp: continue
            source = bullet['source_actor']
            if source not in self.id2type:
                # logger.info(f"id={source} never seen it before ids={self.id2type.keys()}, skip it!")
                # raise Exception("[ERROR] Bullets key not found!")
                continue
#             assert source in self.id2type, f"[ERROR](Unpack state): \
# source_id: {source} not in {self.id2type=}, \n\n\
# state_dict=\n{json.dumps(state_dict, indent=2)}"
            type = self.id2type[source]
            info = BulletInfo(bullet, type)
            if type == 'soldier':
                self.soldier.append(info)
            if type == 'hero':
                self.hero.append(info)
            if type == 'organ':
                self.organ.append(info)
        for l in [self.soldier, self.hero, self.organ]:
            self.merge.extend(l)

class BulletInfo:
    """bullets中每个元素的信息"""
    def __init__(self, bullet, type):
        self.id = bullet.get('runtime_id', 0)    # id编号
        self.source_id = bullet.get('source_actor', 0)    # 发射子弹的对象id
        self.camp = cvt_camp_str2int(bullet.get('camp'))    # 0/1 蓝方/红方
        st = bullet.get('slot_type', 'SLOT_SKILL_0')    # 由发射对象的哪个按钮产生的 SLOT_SKILL_VALID, SLOT_SKILL_0~6
        if isinstance(st, int):
            st = f"SLOT_SKILL_{st}" if st >= 0 else "SLOT_SKILL_VALID"
        self.slot_type = st
        self.skill_id = bullet.get('skill_id', 0)    # 所属技能 (无用, 一直是0)
        self.position = cvt_position(bullet.get('location', {'x': Info.UNSEEN_PADDING, 'z': Info.UNSEEN_PADDING}))
        self.type = type

class DeadsInfo:
    """frame_action中的dead_action列表进行分类解包 (List[DeadAction])"""
    def __init__(self, deads):
        self.soldier: List[DeadInfo] = []    # 士兵死亡
        self.organ: List[DeadInfo] = []    # 防御塔死亡
        self.hero: List[DeadInfo] = []    # 英雄死亡
        self.river_crab: List[DeadInfo] = []    # 河蟹死亡
        self.merge = []    # 全部信息
        for dead in deads:
            tmp = DeadInfo(dead)
            if tmp.death.type[1] == 'ACTOR_SUB_SOLDIER':
                self.soldier.append(tmp)
            if tmp.death.type[0] == 'ACTOR_ORGAN':
                self.organ.append(tmp)
            if tmp.death.type[0] == 'ACTOR_HERO':
                self.hero.append(tmp)
            if tmp.death.config_id == 6827:
                self.river_crab.append(tmp)
        for l in [self.soldier, self.hero, self.organ, self.river_crab]:
            self.merge.extend(l)

class DeadInfo:
    """对dead_action中每个元素进行解包 (DeadAction)"""
    def __init__(self, dead):
        self.death = ActionActorInfo(dead['death'])    # 死亡者
        self.killer = ActionActorInfo(dead['killer'])    # 击杀者
        # TODO: 助攻 assist_set: 本任务可以忽略

class ActionActorInfo:
    """对dead_action进行解包 (ActionActorInfo)"""
    def __init__(self, info):
        self.config_id = info['config_id']
        self.id = info['runtime_id']
        self.type: List[str] = [info['actor_type'], info['sub_type']]
        self.camp = cvt_camp_str2int(info['camp'])

class BuffInfo:
    """对actor_state中的buff_state进行解包 (BuffState)"""
    def __init__(self, buff_state):
        self.skills: List[BuffSkillInfo] = []    # buff组
        self.skill_ids: List[int] = []  # buff_config_id组
        self.marks: List[BuffMarkInfo] = []    # 印记组
        self.marks_ids: List[int] = []  # mark_id组
        self.marks_layers: List[int] = []  # mark_layer组
        for s in buff_state.get('buff_skills', []):
            self.skills.append(BuffSkillInfo(s))
            self.skill_ids.append(self.skills[-1].id)
        for s in buff_state.get('buff_marks', []):
            self.marks.append(BuffMarkInfo(s))
            self.marks_ids.append(self.marks[-1].id)
            self.marks_layers.append(self.marks[-1].layer)

class BuffSkillInfo:
    """对buff_state中的buff_skills进行解包 (BuffSkillState)"""
    def __init__(self, buff_skill):
        s = buff_skill
        self.id = s['configId']    # 配置id
        self.start_time: str = s['startTime']    # 开始时间 (C中的uint64)
        self.times = s['times']    # 生效次数 (好像没什么意义)
        self.buff_skill_id = self.id

class BuffMarkInfo:
    """对buff_state中的buff_marks进行解包 (BuffMarkState)"""
    def __init__(self, buff_mark):
        s = buff_mark
        self.actor_id = s['origin_actorId']    # 施放者id
        self.id = s['configId']    # 配置id
        self.layer = s['layer']    # 层数
        self.buff_mark_id = self.id

class HitTargetInfo:
    """对hit_target_info中每个元素解包"""
    def __init__(self, info):
        self.hit_target = info['hit_target']    # 命中目标的id
        self.skill_id = info['skill_id']    # 技能ID
        self.slot_type = info['slot_type']    # 槽编号 (SLOT_SKILL_[0,1,2,3] 判断是0,1,2,3分别表示普攻, 1,2,3技能)

def cvt_position(d: dict):
    """处理location信息中的x, z坐标, 并处理无法看见的填充值"""
    flag = -1 if inverse_position else 1
    fn = lambda x: Info.UNSEEN_PADDING if x == 100000 else flag * x
    return [fn(d[i]) for i in ['x', 'z']]

def cvt_camp_str2int(camp):
    return camp_to_zero(camp)

try:
    from agent_diy.utils import show_iter
except ImportError:
    def show_iter(value):
        return str(value)

SHOW_SKIP_KEYS = ['state_dict', 'legal_action', 'reward',
'frame_state', 'actor_state', 'hero_state', 'skill_state',
'slot_state', 'equip_state', 'buff_state',
'values', 'logger']
def info2dict(info: Info, skip_keys=True, keys=SHOW_SKIP_KEYS, depth=0, show=False):
    """将Info中的信息进行简化, 跳过不重要信息, 并可转为str输出出来
    Args:
        info: 待显示的Info
        skip_keys: 是否跳过SHOW_SKIP_KEYS中的key
        depth: 当前递归深度
        show: 是否直接print输出结果
    Returns: (递归第一层)
        d: Info化简后的字典
        s: 可用于打印的str
    """
    d = info
    if hasattr(info, '__dict__'):    # 如果是class
        d = {}
        for key, value in info.__dict__.items():
            if skip_keys and key in keys: continue
            if hasattr(value, '__dict__') or isinstance(value, list):
                d[key] = info2dict(value, skip_keys, keys, depth+1)
            else:
                d[key] = value
    if isinstance(info, list):    # 如果是list
        d = []
        for i in info:
            d.append(info2dict(i, skip_keys, keys, depth+1))
    if depth > 0: return d
    s = show_iter(d)
    if show: print(s)
    return d, s

if __name__ == '__main__':
    import json, time
    from agent_diy.utils.dfs_iterable_struct import dfs_iter_apply_fn
    with open("/data/projects/hok1v1/diy/debug/state_dicts.json", 'r') as file:
    # with open("/data/projects/hok1v1/diy/debug/state_dicts_90.json", 'r') as file:
    # with open("/data/projects/hok1v1/diy/debug/state_dicts_狄仁杰二技能打到多段伤害.json", 'r') as file:
    # with open("/data/projects/hok1v1/diy/debug/state_dicts_268.json", 'r') as file:
        state_dicts = json.load(file)
    from diy.feature.reward_manager import GameRewardManager
    reward_manager = GameRewardManager(None)
    start_time = time.time()
    info = Info(None, state_dicts['0'])
    print(info.hero_our.info.position)
    print(time.time() - start_time)
    start_time = time.time()
    info = Info(None, state_dicts['1'])
    print(info.hero_our.info.position)
    print(time.time() - start_time)
    # reward = reward_manager.result(info, info.hero_our.info.hit_target_infos)
    # state_dicts['1']['reward'] = reward
    # info.update(state_dicts['1'])
    # print(info.sub_action_mask, info.sub_action_mask.shape)
    exit()
    # print(info.hero_our.info.position)
    # print(info.hero_our.info.forward)
    # print(info.soldiers_our.close)
    # print(info.soldiers_our.remote)
    # print(info.soldiers_enemy)
    # info = Info(None, state_dicts['1'])
    # print(info.soldiers_our.close)
    # print(info.soldiers_our.remote)
    # print(info.hero_our.info.position)
    # print(info.hero_our.info.forward)
    info2dict(info, show=True)
    # info2dict(info.legal_actions, show=True)
    ### 查看hit_target_info是否出现 ###
    def fn(x, key, passby: list):
        if key == 'hit_target_infos' and len(x):
            passby.append([(info.id2type[i['hit_target']], i['slot_type']) for i in x])
    tmp = []
    dfs_iter_apply_fn(info2dict(info)[0], fn, only_dict=True, input_key=True, passby=tmp, only_leaf=False)
    if len(tmp):
        print(f"hit_target_info={tmp}")
    exit()
    ### 统计有多少种buff ###
    def fn(x, key, passby: list):
        if key == 'buff_state' and len(x):
            passby.append(x)
    buff_list = []
    dfs_iter_apply_fn(info2dict(info)[0], fn, only_dict=False, input_key=True, passby=buff_list, only_leaf=False)
    print(buff_list)

    ### 统计分类buff ###
    buff_list = {'skills': set(), 'marks': set()}
    def fn(x, key, passby: dict):
        if key == 'buff_skill_id': passby['skills'].add(x)
        if key == 'buff_mark_id': passby['marks'].add(x)
    dfs_iter_apply_fn(info2dict(info, skip_keys=False)[0], fn, only_dict=False, input_key=True, passby=buff_list)
    print(buff_list)

    ### 统计共有多少种behave ###
    def fn(x, key, passby: list):
        if key == 'behave':
            passby.append(x)
    passby = []
    with open("test.yaml", 'w') as file:
        file.write(info2dict(info)[1])
    dfs_iter_apply_fn(info2dict(info, skip_keys=False)[0], fn, only_dict=False, input_key=True, passby=passby)
    behaves = set()
    behaves = behaves.union(passby)
    print(behaves)

    print(len(info.deads.merge))
    frame_state = state_dicts['1']['frame_state']
    key = 'frame_action'
    if key not in frame_state: exit()
    info = frame_state[key]
    if key == 'frame_action':
        key = 'dead_action'
        info = info.get(key)
        if info is None: exit()
    if len(info) == 0: exit()
    print(json.dumps(info, indent=2))
