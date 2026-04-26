#!/usr/bin/env python3
"""
Run all stages in order: validate → optional minimal train → full train → evaluate → plots.

This is a thin orchestrator; each stage is a standalone script for debugging.

Usage:
  python training/run_pipeline.py --env-url $env:ENV_URL --skip-train
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


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n=>", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(cwd), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged training + eval pipeline.")
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-minimal-train", action="store_true", help="Skip stage 2 (keep stage 3).")
    parser.add_argument("--skip-full-train", action="store_true", help="Skip stage 3 (eval/plots only).")
    parser.add_argument("--skip-train", action="store_true", help="Skip both stage 2 and 3.")
    parser.add_argument("--use-qlora", action="store_true", help="Pass through to training stages (GPU).")
    parser.add_argument(
        "--run-dir",
        default="training/artifacts/run1",
        help="Where full training (stage 3) writes artifacts; also used for plots.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = utf8_bootstrap.py_child_args()
    use_qlora = ["--use-qlora"] if args.use_qlora else []

    if not args.skip_validate:
        _run(
            [*py, str(root / "training" / "stage1_validate_env.py"), "--env-url", args.env_url],
            root,
        )

    if not args.skip_train and not args.skip_minimal_train:
        cmd = [
            *py,
            str(root / "training" / "stage2_minimal_train.py"),
            "--env-url",
            args.env_url,
        ] + use_qlora
        _run(cmd, root)

    if not args.skip_train and not args.skip_full_train:
        cmd = [
            *py,
            str(root / "training" / "stage3_full_train.py"),
            "--env-url",
            args.env_url,
            "--output-dir",
            args.run_dir,
        ] + use_qlora
        _run(cmd, root)

    _run(
        [
            *py,
            str(root / "training" / "stage4_evaluate.py"),
            "--env-url",
            args.env_url,
            "--run-dir",
            args.run_dir,
        ],
        root,
    )

    _run(
        [
            *py,
            str(root / "training" / "plot_metrics.py"),
            "--run-dir",
            args.run_dir,
            "--eval-summary",
            str(root / "training" / "artifacts" / "eval_summary.json"),
            "--assets-dir",
            str(root / "assets"),
        ],
        root,
    )
    print("\n[OK] Pipeline complete. Check assets/ for PNGs and training/artifacts/ for JSON/MD.")


if __name__ == "__main__":
    main()
