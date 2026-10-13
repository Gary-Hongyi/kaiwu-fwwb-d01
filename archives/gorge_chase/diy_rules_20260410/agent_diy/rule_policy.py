#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Rule policy for Gorge Chase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


MOVE_DIRS: List[Tuple[float, float]] = [
    (1.0, 0.0),
    (1.0, -1.0),
    (0.0, -1.0),
    (-1.0, -1.0),
    (-1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
]


@dataclass
class RuleDecision:
    action: int
    greedy_action: int
    prob: List[float]
    debug: Dict[str, float]


class RulePolicy:
    def decide(self, summary: Dict, legal_action: Sequence[int], prefer_greedy: bool = True) -> RuleDecision:
        legal = [int(v) for v in legal_action]
        scores = self._score_actions(summary, legal)
        legal_indices = [i for i, flag in enumerate(legal) if flag]
        if not legal_indices:
            legal_indices = [0]
            scores[0] = 0.0

        greedy_action = max(legal_indices, key=lambda idx: scores[idx])
        action = greedy_action if prefer_greedy else self._sample_from_scores(scores, legal_indices)
        prob = self._scores_to_prob(scores, legal_indices, greedy_action)
        return RuleDecision(
            action=int(action),
            greedy_action=int(greedy_action),
            prob=prob,
            debug={"best_score": float(scores[greedy_action])},
        )

    def _score_actions(self, summary: Dict, legal_action: Sequence[int]) -> List[float]:
        scores = [-1e9] * len(legal_action)

        hero = summary.get("hero", {})
        monsters = summary.get("monsters", [])
        treasure = summary.get("treasure", {})
        buff = summary.get("buff", {})
        clearance = self._pad_vector(summary.get("clearance", []), 8)
        trap = self._pad_vector(summary.get("trap", []), 4)
        stuck = self._pad_vector(summary.get("stuck", []), 4)
        safe_treasure = self._pad_vector(summary.get("safe_treasure", []), 4)
        flash_ready = bool(summary.get("flash_ready", False))
        progress = float(summary.get("progress", 0.0))
        score_now = float(summary.get("score", 0.0))

        monster_pressure = self._monster_pressure(monsters)
        emergency = monster_pressure > 2.4 or (trap[1] > 0.5 and monster_pressure > 1.6)
        need_escape = monster_pressure > 0.9
        chase_treasure = treasure.get("visible", 0.0) > 0 and (
            safe_treasure[3] > 0.02 or monster_pressure < 0.65 or score_now < 1800
        )
        chase_buff = buff.get("visible", 0.0) > 0 and monster_pressure > 0.55
        stuck_level = max(float(stuck[2]), float(stuck[3]))

        for idx, flag in enumerate(legal_action):
            if not flag:
                continue

            is_flash = idx >= 8
            dir_idx = idx % 8
            move_dx, move_dz = MOVE_DIRS[dir_idx]

            score = 0.0
            score += 2.8 * float(clearance[dir_idx])
            score -= 1.8 * (1.0 - float(clearance[dir_idx])) ** 2
            score -= 1.2 * float(trap[3])

            # Prefer not to oscillate or keep walking into tiny pockets.
            if stuck_level > 0.25:
                score += 1.2 * float(clearance[dir_idx])
                score += 0.8 * self._alignment(move_dx, move_dz, hero.get("escape_bias_dx", 0.0), hero.get("escape_bias_dz", 0.0))

            for order, monster in enumerate(monsters[:2]):
                if monster.get("visible", 0.0) <= 0:
                    continue
                danger = max(0.0, 0.42 - float(monster.get("dist_norm", 1.0))) / 0.42
                away = self._alignment(move_dx, move_dz, -monster.get("dx_norm", 0.0), -monster.get("dz_norm", 0.0))
                side = self._lateral(move_dx, move_dz, monster.get("dx_norm", 0.0), monster.get("dz_norm", 0.0))
                weight = 7.0 if order == 0 else 4.5
                score += weight * danger * away
                score += 0.9 * danger * side

            if chase_treasure:
                safe_gain = 1.5 + 2.8 * float(safe_treasure[3])
                score += safe_gain * self._alignment(
                    move_dx, move_dz, treasure.get("dx_norm", 0.0), treasure.get("dz_norm", 0.0)
                )
                score += 0.8 * float(safe_treasure[2])
            elif treasure.get("visible", 0.0) > 0 and monster_pressure < 0.95:
                score += 1.2 * self._alignment(
                    move_dx, move_dz, treasure.get("dx_norm", 0.0), treasure.get("dz_norm", 0.0)
                )

            if chase_buff:
                score += 2.3 * self._alignment(move_dx, move_dz, buff.get("dx_norm", 0.0), buff.get("dz_norm", 0.0))

            if score_now >= 2500 and monster_pressure > 0.35:
                score += 2.5 * self._alignment(
                    move_dx,
                    move_dz,
                    hero.get("escape_bias_dx", 0.0),
                    hero.get("escape_bias_dz", 0.0),
                )

            if progress > 0.75 and monster_pressure < 0.25:
                # Late game: be slightly more conservative to avoid throwing away a good score.
                score += 0.9 * float(clearance[dir_idx])

            if is_flash:
                if not flash_ready:
                    score -= 50.0
                else:
                    score -= 1.6
                    if emergency:
                        score += 4.5
                    elif need_escape or stuck_level > 0.45:
                        score += 2.4
                    else:
                        score -= 2.5

            else:
                if emergency and flash_ready:
                    score -= 0.5

            scores[idx] = score

        return scores

    def _monster_pressure(self, monsters: Sequence[Dict]) -> float:
        pressure = 0.0
        for idx, monster in enumerate(monsters[:2]):
            if monster.get("visible", 0.0) <= 0:
                continue
            dist = float(monster.get("dist_norm", 1.0))
            threat = max(0.0, 0.42 - dist) / 0.42
            pressure += threat * (1.0 if idx == 0 else 0.7)
        return pressure

    def _alignment(self, move_dx: float, move_dz: float, target_dx: float, target_dz: float) -> float:
        norm = np.sqrt(target_dx * target_dx + target_dz * target_dz)
        if norm < 1e-6:
            return 0.0
        return float((move_dx * target_dx + move_dz * target_dz) / norm)

    def _lateral(self, move_dx: float, move_dz: float, target_dx: float, target_dz: float) -> float:
        norm = np.sqrt(target_dx * target_dx + target_dz * target_dz)
        if norm < 1e-6:
            return 0.0
        cross = move_dx * target_dz - move_dz * target_dx
        return float(abs(cross) / norm)

    def _scores_to_prob(self, scores: Sequence[float], legal_indices: Sequence[int], greedy_action: int) -> List[float]:
        prob = np.zeros(len(scores), dtype=np.float32)
        chosen_scores = np.array([scores[idx] for idx in legal_indices], dtype=np.float32)
        shifted = chosen_scores - np.max(chosen_scores)
        weights = np.exp(np.clip(shifted, -8.0, 0.0))
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-6:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum

        for idx, action_idx in enumerate(legal_indices):
            prob[action_idx] = 0.25 * weights[idx]
        prob[greedy_action] += 0.75
        prob = prob / max(1e-6, float(np.sum(prob)))
        return list(prob.astype(np.float32))

    def _sample_from_scores(self, scores: Sequence[float], legal_indices: Sequence[int]) -> int:
        weights = np.array([max(-20.0, scores[idx]) for idx in legal_indices], dtype=np.float32)
        weights = np.exp(weights - np.max(weights))
        weights = weights / max(1e-6, float(np.sum(weights)))
        return int(np.random.choice(np.array(legal_indices, dtype=np.int64), p=weights))

    def _pad_vector(self, values: Sequence[float], size: int) -> np.ndarray:
        arr = np.array(list(values) if values is not None else [], dtype=np.float32).flatten()
        if arr.size >= size:
            return arr[:size]
        if arr.size == 0:
            return np.zeros(size, dtype=np.float32)
        return np.pad(arr, (0, size - arr.size), mode="constant")
