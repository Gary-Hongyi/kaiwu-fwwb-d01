#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Self-contained smoke runner for the v2 residual policy.

This script verifies the whole v2 stack **without** needing the kaiwudrl /
common_python training framework or a real checkpoint. When those frameworks
are not importable (e.g. on a laptop or in CI) lightweight stand-ins are
injected into ``sys.modules`` *before* importing ``agent_ppo`` so that the old
``Agent`` / ``OldPolicy`` classes can still be constructed with random weights.

It checks the acceptance criteria from the design:

  1. ModelV2 zero-init delta  -> final_logits == old_logits at init.
  2. GRU memory threading is finite and shape-stable.
  3. One AlgorithmV2 train step is finite and moves delta_logits.
  4. obs_builder_v2 emits FEATURE_DIM_V2 finite features.
  5. reward_manager_v2 emits V2_VALUE_HEAD_NUM heads and warms up correctly.
  6. AgentV2 integration: build, legacy-sample learn, save/load round-trip,
     dual checkpoint format detection, confidence + legal-action fallback.

Run:  python3 agent_ppo/debug/run_v2_smoke.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Optional framework stubs (only used when the real ones are unavailable).
# --------------------------------------------------------------------------- #
def _install_framework_stubs() -> dict[str, bool]:
    used = {"kaiwudrl": False, "common_python": False}

    try:
        import kaiwudrl.interface.agent  # noqa: F401
    except Exception:
        kaiwudrl = types.ModuleType("kaiwudrl")
        interface = types.ModuleType("kaiwudrl.interface")
        agent_mod = types.ModuleType("kaiwudrl.interface.agent")

        class BaseAgent:  # minimal stand-in for the platform base class
            def __init__(self, *args, **kwargs):
                pass

            def send_sample_data(self, *args, **kwargs):
                pass

        agent_mod.BaseAgent = BaseAgent
        interface.agent = agent_mod
        kaiwudrl.interface = interface
        sys.modules["kaiwudrl"] = kaiwudrl
        sys.modules["kaiwudrl.interface"] = interface
        sys.modules["kaiwudrl.interface.agent"] = agent_mod
        used["kaiwudrl"] = True

    try:
        import common_python.utils.common_func  # noqa: F401
    except Exception:
        common_python = types.ModuleType("common_python")
        utils = types.ModuleType("common_python.utils")
        common_func = types.ModuleType("common_python.utils.common_func")

        def create_cls(cls_name, **fields):
            def __init__(self, **kwargs):
                for key, default in fields.items():
                    setattr(self, key, kwargs.get(key, default))

            return type(cls_name, (), {"__init__": __init__})

        class Frame:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        common_func.create_cls = create_cls
        common_func.Frame = Frame
        utils.common_func = common_func
        common_python.utils = utils
        sys.modules["common_python"] = common_python
        sys.modules["common_python.utils"] = utils
        sys.modules["common_python.utils.common_func"] = common_func
        used["common_python"] = True

    return used


# --------------------------------------------------------------------------- #
# Individual checks. Each returns (passed: bool, detail: str).
# --------------------------------------------------------------------------- #
def check_tensor_equivalence():
    from agent_ppo.conf.conf import Config
    from agent_ppo.model.model_v2 import ModelV2

    torch.manual_seed(0)
    model = ModelV2()
    model.eval()
    feature = torch.randn(6, Config.FEATURE_DIM_V2)
    old_logits = torch.randn(6, Config.LEGAL_ACTION_DIM)
    old_value = torch.randn(6, Config.VALUE_HEAD_NUM)
    out = model.forward_v2(feature, old_logits, old_value)
    max_diff = torch.max(torch.abs(out.final_logits - old_logits)).item()
    delta_norm = out.delta_norm.item()
    ok = max_diff < 1e-7 and delta_norm < 1e-7
    return ok, f"max|final-old|={max_diff:.3e}, delta_norm={delta_norm:.3e}"


def check_memory_threading():
    from agent_ppo.conf.conf import Config
    from agent_ppo.model.model_v2 import ModelV2

    torch.manual_seed(1)
    model = ModelV2()
    feature = torch.randn(3, Config.FEATURE_DIM_V2)
    old_logits = torch.randn(3, Config.LEGAL_ACTION_DIM)
    old_value = torch.randn(3, Config.VALUE_HEAD_NUM)
    state = model.initial_state(3)
    out1 = model.forward_v2(feature, old_logits, old_value, state)
    out2 = model.forward_v2(feature, old_logits, old_value, out1.hidden_state)
    finite = bool(torch.isfinite(out1.hidden_state).all() and torch.isfinite(out2.hidden_state).all())
    shape_ok = tuple(out2.hidden_state.shape) == (3, model.memory_dim)
    ok = finite and shape_ok
    return ok, f"hidden_shape={tuple(out2.hidden_state.shape)}, finite={finite}"


def check_training_step():
    from agent_ppo.algorithm.algorithm_v2 import AlgorithmV2
    from agent_ppo.conf.conf import Config
    from agent_ppo.model.model_v2 import ModelV2

    torch.manual_seed(7)
    np.random.seed(7)
    batch = 8
    legal = np.zeros((batch, Config.LEGAL_ACTION_DIM), dtype=np.float32)
    action = np.zeros((batch, len(Config.LABEL_SIZE_LIST)), dtype=np.int64)
    offset = 0
    rng = np.random.default_rng(7)
    for head_idx, size in enumerate(Config.LABEL_SIZE_LIST):
        for row in range(batch):
            choices = rng.choice(size, size=max(1, size // 3), replace=False)
            legal[row, offset + choices] = 1.0
            action[row, head_idx] = int(choices[0])
        offset += size

    model = ModelV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    algorithm = AlgorithmV2(model, optimizer)
    old_logits = torch.randn(batch, Config.LEGAL_ACTION_DIM)
    old_value = torch.randn(batch, Config.VALUE_HEAD_NUM)
    feature = torch.randn(batch, Config.FEATURE_DIM_V2)
    before = model.forward_v2(feature, old_logits, old_value).delta_norm.item()
    info = algorithm.learn(
        {
            "feature_v2": feature,
            "old_logits": old_logits,
            "old_value": old_value,
            "legal_action": legal,
            "action": action,
            "return_v2": torch.randn(batch, Config.V2_VALUE_HEAD_NUM),
            "advantage_v2": torch.randn(batch, Config.V2_VALUE_HEAD_NUM),
            "is_train": np.ones(batch, dtype=np.float32),
        }
    )
    after = model.forward_v2(feature, old_logits, old_value).delta_norm.item()
    values = np.asarray(list(info.__dict__.values()), dtype=np.float64)
    finite = bool(np.all(np.isfinite(values)))
    moved = after > before
    ok = finite and moved
    return ok, (
        f"finite={finite}, delta_before={before:.3e}, delta_after={after:.3e}, "
        f"ppo={info.ppo_loss:.3f}, value={info.value_loss:.3f}, distill={info.distill_loss:.3f}, "
        f"kl={info.kl_to_old:.3e}"
    )


def check_obs_builder_v2():
    from agent_ppo.conf.conf import Config
    from agent_ppo.feature.obs_builder_v2 import FedExingObsBuilderV2

    builder = FedExingObsBuilderV2(None)
    result = builder.smoke_test()
    ok = result.feature_dim == Config.FEATURE_DIM_V2 and result.finite
    return ok, (
        f"feature_dim={result.feature_dim} (want {Config.FEATURE_DIM_V2}), "
        f"finite={result.finite}, range=[{result.min_value:.3f},{result.max_value:.3f}]"
    )


def _dummy_frame_data():
    return {
        "frame_no": 200,
        "hero_states": [
            {
                "runtime_id": 1, "camp": 0, "hp": 5000, "max_hp": 8000, "ep": 120,
                "max_ep": 400, "level": 4, "money_cnt": 800, "kill_cnt": 1, "dead_cnt": 0,
                "total_hurt_to_hero": 1200, "exp": 500, "location": {"x": 4000, "z": 0},
                "buff_skills": [],
            },
            {
                "runtime_id": 2, "camp": 1, "hp": 2200, "max_hp": 8000, "ep": 60,
                "max_ep": 400, "level": 3, "money_cnt": 600, "kill_cnt": 0, "dead_cnt": 1,
                "total_hurt_to_hero": 400, "exp": 300, "location": {"x": 9000, "z": 0},
                "buff_skills": [],
            },
        ],
        "npc_states": [
            {"sub_type": 21, "camp": 0, "hp": 9000, "max_hp": 10000, "location": {"x": -8000, "z": 0}},
            {"sub_type": 21, "camp": 1, "hp": 2500, "max_hp": 10000, "location": {"x": 12000, "z": 0}},
        ],
        "cakes": [],
    }


def check_reward_manager_v2():
    from agent_ppo.conf.conf import Config
    from agent_ppo.feature.reward_manager_v2 import FedExingGameRewardManagerV2

    frame = _dummy_frame_data()
    mgr0 = FedExingGameRewardManagerV2(1)
    mgr0.set_train_step(0)
    out0 = mgr0.result(frame)
    vec0 = np.asarray(out0["v2_reward_vector"], dtype=np.float64)

    mgr1 = FedExingGameRewardManagerV2(1)
    mgr1.set_train_step(Config.REWARD_HEAD_WARMUP_STEPS * 4)
    out1 = mgr1.result(_dummy_frame_data())
    vec1 = np.asarray(out1["v2_reward_vector"], dtype=np.float64)

    imitation_idx = list(Config.V2_VALUE_HEAD_NAMES).index("imitation")
    dim_ok = vec0.size == Config.V2_VALUE_HEAD_NUM
    finite = bool(np.all(np.isfinite(vec0)) and np.all(np.isfinite(vec1)))
    head_meta_ok = out0.get("stay_close_to_old_head") == "imitation"
    warmup_ok = vec0[imitation_idx] > vec1[imitation_idx] + 1e-9  # imitation decays
    ok = dim_ok and finite and head_meta_ok and warmup_ok
    return ok, (
        f"dim={vec0.size}/{Config.V2_VALUE_HEAD_NUM}, finite={finite}, "
        f"imitation@0={vec0[imitation_idx]:.4f} -> @warm={vec1[imitation_idx]:.4f}, "
        f"head_meta={head_meta_ok}"
    )


def _make_legacy_samples(batch: int):
    """Build SampleData-shaped rows that the old sample protocol expects."""
    from agent_ppo.conf.conf import Config
    from agent_ppo.feature.definition import SampleData

    width = int(sum(shape[0] for shape in Config.data_shapes))
    rng = np.random.default_rng(11)
    samples = []
    for _ in range(batch):
        arr = rng.standard_normal(width).astype(np.float32) * 0.1
        samples.append(SampleData(sample=torch.as_tensor(arr)))
    return samples


def check_agent_v2_integration():
    from agent_ppo.agent_v2 import AgentV2
    from agent_ppo.conf.conf import Config

    torch.manual_seed(3)
    agent = AgentV2(device=torch.device("cpu"))

    # 1) legacy-sample learn path: frozen old model -> v2 batch -> one train step
    delta_before = float(
        torch.linalg.vector_norm(agent.model.delta_head[-1].weight).item()
    )
    info = agent.learn(_make_legacy_samples(4))
    delta_after = float(
        torch.linalg.vector_norm(agent.model.delta_head[-1].weight).item()
    )
    learn_finite = np.all(np.isfinite(np.asarray(list(info.__dict__.values()), dtype=np.float64)))
    learn_moved = delta_after > delta_before

    # 2) save / load round-trip (v2 format)
    with tempfile.TemporaryDirectory() as tmp:
        agent.save_model(tmp, id="smoke")
        v2_ckpt = Path(tmp) / "model_v2.ckpt-smoke.pkl"
        compat_ckpt = Path(tmp) / "model.ckpt-smoke.pkl"
        files_ok = v2_ckpt.exists() and compat_ckpt.exists()
        reloaded = AgentV2(device=torch.device("cpu"))
        reloaded.load_model(tmp, id="smoke")
        w_a = agent.model.delta_head[1].weight
        w_b = reloaded.model.delta_head[1].weight
        load_ok = torch.allclose(w_a, w_b)

        # 3) dual-format: a raw old state_dict must route to OldPolicy, not ModelV2
        old_only = Path(tmp) / "model.ckpt-oldonly.pkl"
        torch.save(agent.old_policy.model.state_dict(), old_only)
        dual = AgentV2(device=torch.device("cpu"))
        dual.load_model(tmp, id="oldonly")  # should NOT raise
        dual_ok = True

    # 4) confidence helper + legal-action gate (pure logic, no env protocol needed)
    prob = [[0.0] * Config.LEGAL_ACTION_DIM]
    offsets = np.cumsum([0] + Config.LABEL_SIZE_LIST)
    test_action = [0, 15, 15, 15, 15, 0]
    for idx, val in enumerate(test_action):
        prob[0][offsets[idx] + val] = 0.4 + 0.1 * idx
    conf = AgentV2._selected_action_confidence(test_action, prob)
    conf_ok = abs(conf - 0.4) < 1e-6  # min selected prob == head-0 prob

    # legal-action check used by the illegal_action fallback reason
    raw_len = sum(Config.LABEL_SIZE_LIST[:-1]) + Config.LABEL_SIZE_LIST[0] * Config.LABEL_SIZE_LIST[-1]
    legal = np.zeros(raw_len, dtype=np.float32)
    off = 0
    for size in Config.LABEL_SIZE_LIST[:-1]:
        legal[off] = 1.0  # head index 0 legal
        off += size
    legal[off + 0 * Config.LABEL_SIZE_LIST[-1] + 0] = 1.0  # button 0 -> target 0 legal
    old_agent = agent.old_policy.agent
    legal_action_ok = old_agent._fedexing_is_legal_action(legal, [0, 0, 0, 0, 0, 0])
    illegal_action_bad = not old_agent._fedexing_is_legal_action(legal, [5, 0, 0, 0, 0, 0])

    # 5) env-flag fallback switch
    os.environ["FORCE_OLD_POLICY"] = "true"
    force_ok = agent._force_old_policy() is True
    os.environ.pop("FORCE_OLD_POLICY", None)

    ok = all(
        [
            learn_finite, learn_moved, files_ok, load_ok, dual_ok,
            conf_ok, legal_action_ok, illegal_action_bad, force_ok,
        ]
    )
    return ok, (
        f"learn_finite={bool(learn_finite)}, delta {delta_before:.3e}->{delta_after:.3e}, "
        f"save_files={files_ok}, reload_match={load_ok}, dual_format={dual_ok}, "
        f"confidence={conf:.3f}({conf_ok}), legal_gate={legal_action_ok}/{illegal_action_bad}, "
        f"force_old={force_ok}"
    )


CHECKS = [
    ("1. model_v2 zero-init equivalence", check_tensor_equivalence),
    ("2. model_v2 GRU memory threading", check_memory_threading),
    ("3. algorithm_v2 one train step", check_training_step),
    ("4. obs_builder_v2 feature dim", check_obs_builder_v2),
    ("5. reward_manager_v2 heads+warmup", check_reward_manager_v2),
    ("6. agent_v2 integration", check_agent_v2_integration),
]


def main() -> int:
    used = _install_framework_stubs()
    stub_note = [name for name, on in used.items() if on]
    print("=" * 74)
    print("v2 residual policy - smoke test")
    print(f"torch={torch.__version__}, numpy={np.__version__}")
    if stub_note:
        print(f"NOTE: using lightweight stubs for missing modules: {', '.join(stub_note)}")
    print("=" * 74)

    results = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - smoke runner must report, not crash
            ok, detail = False, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        results.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"        {detail}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("=" * 74)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
