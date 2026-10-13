from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_jsonl(path: str) -> None:
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {path}")
    old_win = np.mean([float(r.get("old_win", 0.0)) for r in records])
    v2_win = np.mean([float(r.get("v2_win", 0.0)) for r in records])
    fallback = np.mean([float(r.get("fallback", 0.0)) for r in records])
    illegal = np.mean([float(r.get("illegal", 0.0)) for r in records])
    print(f"episodes={len(records)}")
    print(f"old_win_rate={old_win:.4f}")
    print(f"v2_win_rate={v2_win:.4f}")
    print(f"fallback_rate={fallback:.4f}")
    print(f"illegal_action_rate={illegal:.4f}")
    print("decision=PASS" if v2_win >= old_win and illegal < 0.01 else "decision=FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", default="", help="jsonl fields: old_win, v2_win, fallback, illegal")
    args = parser.parse_args()
    if args.eval_jsonl:
        summarize_jsonl(args.eval_jsonl)
    else:
        print("Provide --eval-jsonl to summarize arena/evaluation records.")
        print('Example line: {"old_win": 1, "v2_win": 1, "fallback": 0, "illegal": 0}')


if __name__ == "__main__":
    main()

