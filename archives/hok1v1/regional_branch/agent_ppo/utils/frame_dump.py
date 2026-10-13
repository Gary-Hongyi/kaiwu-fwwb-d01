#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Current-protocol frame dumping helpers for Taichu migration analysis."""

import json
import os
import time
from copy import deepcopy

import numpy as np


def _fedexing_env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _fedexing_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _fedexing_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _fedexing_safe_len(value):
    try:
        return len(value)
    except TypeError:
        return None


def _fedexing_summarize_observation(observation):
    if not isinstance(observation, dict):
        return {"type": type(observation).__name__}

    frame_state = observation.get("frame_state") or {}
    hero_states = frame_state.get("hero_states") or []
    npc_states = frame_state.get("npc_states") or []
    bullets = frame_state.get("bullets") or []
    cakes = frame_state.get("cakes") or []
    frame_action = frame_state.get("frame_action") or {}
    legal_action = observation.get("legal_action") or []
    sub_action_mask = observation.get("sub_action_mask") or {}

    npc_sub_type_counts = {}
    for npc in npc_states:
        key = str(npc.get("sub_type")) if isinstance(npc, dict) else type(npc).__name__
        npc_sub_type_counts[key] = npc_sub_type_counts.get(key, 0) + 1

    return {
        "top_keys": sorted(observation.keys()),
        "env_id": observation.get("env_id"),
        "player_id": observation.get("player_id"),
        "player_camp": observation.get("player_camp"),
        "camp": observation.get("camp"),
        "win": observation.get("win"),
        "legal_action_len": len(legal_action),
        "legal_action_sum": int(sum(legal_action)) if legal_action else 0,
        "sub_action_mask_type": type(sub_action_mask).__name__,
        "sub_action_mask_keys": sorted(sub_action_mask.keys()) if isinstance(sub_action_mask, dict) else None,
        "frame_state_keys": sorted(frame_state.keys()),
        "frame_no": frame_state.get("frame_no", frame_state.get("frameNo")),
        "hero_count": len(hero_states),
        "npc_count": len(npc_states),
        "npc_sub_type_counts": npc_sub_type_counts,
        "bullet_count": len(bullets),
        "cake_count": len(cakes),
        "dead_action_count": len(frame_action.get("dead_action") or []),
        "heroes": [_fedexing_unit_summary(hero) for hero in hero_states],
        "npcs_first": [_fedexing_unit_summary(npc) for npc in npc_states[:8]],
        "bullet_keys": sorted(bullets[0].keys()) if bullets and isinstance(bullets[0], dict) else [],
        "cake_keys": sorted(cakes[0].keys()) if cakes and isinstance(cakes[0], dict) else [],
    }


def _fedexing_unit_summary(unit):
    if not isinstance(unit, dict):
        return {"type": type(unit).__name__}
    location = unit.get("location") or {}
    skill_state = unit.get("skill_state") or {}
    return {
        "keys": sorted(unit.keys()),
        "runtime_id": unit.get("runtime_id"),
        "player_id": unit.get("player_id"),
        "config_id": unit.get("config_id"),
        "actor_type": unit.get("actor_type"),
        "sub_type": unit.get("sub_type"),
        "camp": unit.get("camp"),
        "hp": unit.get("hp"),
        "max_hp": unit.get("max_hp"),
        "level": unit.get("level"),
        "money": unit.get("money"),
        "moneyCnt": unit.get("moneyCnt"),
        "location": {"x": location.get("x"), "z": location.get("z")},
        "skill_slot_count": _fedexing_safe_len(skill_state.get("slot_states", [])),
    }


class FedExingFrameDumper:
    """JSONL dumper controlled by HOK_DUMP_FRAMES environment variables."""

    def __init__(self, logger=None):
        self.enabled = _fedexing_env_flag("HOK_DUMP_FRAMES", False)
        self.logger = logger
        self.max_records = _fedexing_env_int("HOK_DUMP_MAX_RECORDS", 80)
        self.interval = max(_fedexing_env_int("HOK_DUMP_INTERVAL", 1), 1)
        self.include_raw = _fedexing_env_flag("HOK_DUMP_RAW", True)
        self.records = 0
        self.step_events_seen = 0
        self.path = None

        if not self.enabled:
            return

        dump_dir = os.environ.get("HOK_DUMP_DIR", "agent_ppo/debug/frame_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(dump_dir, f"current_protocol_{stamp}_{os.getpid()}.jsonl")
        self._fedexing_write(
            {
                "event": "meta",
                "created_at": stamp,
                "pid": os.getpid(),
                "include_raw": self.include_raw,
                "max_records": self.max_records,
                "interval": self.interval,
            }
        )
        if self.logger:
            self.logger.info(f"HOK frame dumper enabled, writing to {self.path}")

    def dump(self, event, episode, env_obs=None, observation=None, actions=None, env_reward=None):
        if not self.enabled or self.records >= self.max_records:
            return
        if event == "step":
            self.step_events_seen += 1
            if (self.step_events_seen - 1) % self.interval != 0:
                return

        if observation is None and isinstance(env_obs, dict):
            observation = env_obs.get("observation")

        obs_payload = {}
        obs_summary = {}
        if isinstance(observation, dict):
            for agent_id, obs in observation.items():
                obs_summary[str(agent_id)] = _fedexing_summarize_observation(obs)
                if self.include_raw:
                    obs_payload[str(agent_id)] = deepcopy(obs)

        record = {
            "event": event,
            "episode": episode,
            "record_index": self.records,
            "env_frame_no": env_obs.get("frame_no") if isinstance(env_obs, dict) else None,
            "terminated": env_obs.get("terminated") if isinstance(env_obs, dict) else None,
            "truncated": env_obs.get("truncated") if isinstance(env_obs, dict) else None,
            "observation_summary": obs_summary,
            "actions": actions,
            "env_reward": env_reward,
        }
        if self.include_raw:
            record["observation"] = obs_payload
            if isinstance(env_obs, dict):
                record["extra_info"] = deepcopy(env_obs.get("extra_info"))

        self._fedexing_write(record)
        self.records += 1

    def _fedexing_write(self, record):
        with open(self.path, "a", encoding="utf-8") as file_obj:
            json.dump(record, file_obj, ensure_ascii=False, default=_fedexing_json_default)
            file_obj.write("\n")

    def should_stop(self):
        return self.enabled and self.records >= self.max_records
