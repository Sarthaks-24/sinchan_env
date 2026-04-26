#!/usr/bin/env python3
"""
Stage 2 — Minimal training (sanity check): small SLM, few steps, checkpoints + metrics.

Delegates to `train_sinchan.py` with conservative defaults for Colab free tier / local CPU+GPU.

Usage:
  python training/stage2_minimal_train.py --env-url $env:ENV_URL
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_train = Path(__file__).resolve().parent
if str(_train) not in sys.path:
    sys.path.insert(0, str(_train))
import utf8_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: short GRPO sanity run.")
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument(
        "--output-dir",
        default="training/artifacts/stage2_minimal",
        help="Checkpoints and trainer_state.json",
    )
    parser.add_argument("--max-steps", type=int, default=80, help="Keep 50–120 for a quick proof run")
    parser.add_argument("--dataset-size", type=int, default=40)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="4-bit QLoRA via PEFT (requires bitsandbytes + peft; GPU recommended).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_script = root / "training" / "train_sinchan.py"
    if not train_script.is_file():
        raise SystemExit(f"Missing {train_script}")

    cmd: list[str] = [
        *utf8_bootstrap.py_child_args(),
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

    print("Stage 2 — minimal training\n  " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
    if rc != 0:
        raise SystemExit(rc)
    print(f"\n[OK] Stage 2 finished. Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
