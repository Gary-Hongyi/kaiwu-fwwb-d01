from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_ppo.conf.conf import Config
from agent_ppo.model.model_v2 import ModelV2


def tensor_equivalence() -> None:
    model = ModelV2()
    batch = 4
    feature_v2 = torch.randn(batch, Config.FEATURE_DIM_V2)
    old_logits = torch.randn(batch, Config.LEGAL_ACTION_DIM)
    old_value = torch.randn(batch, Config.VALUE_HEAD_NUM)
    out = model.forward_v2(feature_v2, old_logits, old_value)
    max_diff = torch.max(torch.abs(out.final_logits - old_logits)).item()
    delta_norm = out.delta_norm.item()
    print(f"tensor_equivalence max_masked_logits_diff={max_diff:.8f}, delta_norm={delta_norm:.8f}")
    assert max_diff < 1e-7, "ModelV2 zero-init delta failed"


def observation_equivalence(old_ckpt: str, observation_json: str) -> None:
    from agent_ppo.agent_v2 import AgentV2
    from agent_ppo.old_policy import OldPolicy

    os.environ["USE_V2_POLICY"] = "true"
    os.environ["USE_V2_DELTA"] = "true"
    os.environ["FALLBACK_TO_OLD"] = "false"
    os.environ["V2_CONFIDENCE_THRESHOLD"] = "0.0"

    observation = json.loads(Path(observation_json).read_text(encoding="utf-8"))
    old = OldPolicy(ckpt_path=old_ckpt)
    v2 = AgentV2()
    v2.old_policy.load_model(old_ckpt)
    old.reset(observation)
    v2.reset(observation)

    old_output = old.forward(observation)
    v2_output = v2._forward_v2(observation, old_output)
    diff = torch.max(torch.abs(v2_output.final_logits.cpu() - torch.as_tensor(old_output.logits))).item()
    old_action = old.exploit(observation)
    v2_action = v2.exploit(observation)
    print(f"observation_equivalence max_masked_logits_diff={diff:.8f}")
    print(f"old_exploit={old_action}")
    print(f"v2_initial_exploit={v2_action}")
    assert diff < 1e-6, "v2 initial logits are not equal to old logits"
    assert old_action == v2_action, "v2 initial exploit action differs from old policy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ckpt", default="")
    parser.add_argument("--observation-json", default="")
    args = parser.parse_args()
    tensor_equivalence()
    if args.old_ckpt and args.observation_json:
        observation_equivalence(args.old_ckpt, args.observation_json)
    else:
        print("skip observation equivalence: provide --old-ckpt and --observation-json to run end-to-end check")


if __name__ == "__main__":
    main()
