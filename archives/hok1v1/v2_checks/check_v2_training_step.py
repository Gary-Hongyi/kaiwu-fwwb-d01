from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_ppo.algorithm.algorithm_v2 import AlgorithmV2
from agent_ppo.conf.conf import Config
from agent_ppo.model.model_v2 import ModelV2


def _random_legal_and_action(batch: int):
    legal = np.zeros((batch, Config.LEGAL_ACTION_DIM), dtype=np.float32)
    action = np.zeros((batch, len(Config.LABEL_SIZE_LIST)), dtype=np.int64)
    offset = 0
    rng = np.random.default_rng(20260529)
    for head_idx, size in enumerate(Config.LABEL_SIZE_LIST):
        for row in range(batch):
            choices = rng.choice(size, size=max(1, size // 3), replace=False)
            legal[row, offset + choices] = 1.0
            action[row, head_idx] = int(choices[0])
        offset += size
    return legal, action


def main() -> None:
    torch.manual_seed(20260529)
    np.random.seed(20260529)
    batch = 8
    legal, action = _random_legal_and_action(batch)
    model = ModelV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    algorithm = AlgorithmV2(model, optimizer)
    old_logits = torch.randn(batch, Config.LEGAL_ACTION_DIM)
    old_value = torch.randn(batch, Config.VALUE_HEAD_NUM)
    feature_v2 = torch.randn(batch, Config.FEATURE_DIM_V2)
    before = model.forward_v2(feature_v2, old_logits, old_value).delta_norm.item()
    info = algorithm.learn(
        {
            "feature_v2": feature_v2,
            "old_logits": old_logits,
            "old_value": old_value,
            "legal_action": legal,
            "action": action,
            "return_v2": torch.randn(batch, Config.V2_VALUE_HEAD_NUM),
            "advantage_v2": torch.randn(batch, Config.V2_VALUE_HEAD_NUM),
            "is_train": np.ones(batch, dtype=np.float32),
        }
    )
    after = model.forward_v2(feature_v2, old_logits, old_value).delta_norm.item()
    values = np.asarray(list(info.__dict__.values()), dtype=np.float64)
    print(info)
    print(f"delta_norm_before={before:.8f}, delta_norm_after={after:.8f}")
    assert np.all(np.isfinite(values)), "loss info contains NaN/Inf"
    assert after > before, "delta branch did not start moving after one optimizer step"


if __name__ == "__main__":
    main()
