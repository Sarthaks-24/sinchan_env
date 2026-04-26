#!/usr/bin/env python3
"""
Stage 3 — Full (or scaled) training: longer run, same stack as stage 2, heavier logging.

Usage:
  python training/stage3_full_train.py --env-url $env:ENV_URL --max-steps 300
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: longer GRPO training run.")
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--output-dir", default="training/artifacts/run1")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-qlora", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_script = root / "training" / "train_sinchan.py"
    if not train_script.is_file():
        raise SystemExit(f"Missing {train_script}")

    cmd: list[str] = [
        sys.executable,
        str(train_script),
        "--env-url",
        args.env_url,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--max-steps",
        str(args.max_steps),
        "--dataset-size",
        str(args.dataset_size),
        "--learning-rate",
        "1e-5",
        "--num-generations",
        "2",
        "--seed",
        str(args.seed),
        "--precision",
        "auto",
    ]
    if args.use_qlora:
        cmd.append("--use-qlora")

    print("Stage 3 — full training\n  " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
    if rc != 0:
        raise SystemExit(rc)
    print(f"\n[OK] Stage 3 finished. Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
