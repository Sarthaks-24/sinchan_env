#!/usr/bin/env python3
"""
Stage 4 — Evaluation: random (untrained-style) vs rule-based baseline; training-run reward stats.

Writes a human-readable report for the README before/after table and hackathon evidence.

Usage:
  python training/stage4_evaluate.py --env-url ... --run-dir training/artifacts/run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

_train = Path(__file__).resolve().parent
if str(_train) not in sys.path:
    sys.path.insert(0, str(_train))
import utf8_bootstrap  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(nums: list[float]) -> float:
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def _extract_train_rewards(run_dir: Path) -> tuple[list[float], list[float], list[int]]:
    """Return (rewards, losses, steps) from metrics.jsonl or trainer_state log_history."""
    rewards: list[float] = []
    losses: list[float] = []
    steps: list[int] = []

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = row.get("step")
            if st is not None:
                try:
                    steps.append(int(st))
                except (TypeError, ValueError):
                    pass
            for key in ("reward", "rewards/mean", "train/reward"):
                if key in row and isinstance(row[key], (int, float)):
                    rewards.append(float(row[key]))
                    break
            for key in ("loss", "train/loss"):
                if key in row and isinstance(row[key], (int, float)):
                    losses.append(float(row[key]))
                    break
        return rewards, losses, steps

    ts = run_dir / "trainer_state.json"
    if ts.is_file():
        data = _load_json(ts)
        history = data.get("log_history") or []
        if isinstance(history, list):
            for row in history:
                if not isinstance(row, dict):
                    continue
                st = row.get("step")
                if st is not None:
                    try:
                        steps.append(int(st))
                    except (TypeError, ValueError):
                        pass
                for key in ("reward", "rewards/mean", "train/reward"):
                    if key in row and isinstance(row[key], (int, float)):
                        rewards.append(float(row[key]))
                        break
                for key in ("loss", "train/loss"):
                    if key in row and isinstance(row[key], (int, float)):
                        losses.append(float(row[key]))
                        break
    return rewards, losses, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: baselines + training summary.")
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-output",
        default="training/artifacts/eval_summary.json",
        help="JSON from evaluate_scenarios.py",
    )
    parser.add_argument(
        "--run-dir",
        default="training/artifacts/run1",
        help="Training output (metrics.jsonl or trainer_state.json)",
    )
    parser.add_argument(
        "--report",
        default="training/artifacts/stage4_report.md",
        help="Markdown report for README / submission",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_script = root / "training" / "evaluate_scenarios.py"
    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_script.is_file():
        raise SystemExit(f"Missing {eval_script}")

    print("Stage 4 — running scripted baselines (random vs rule-based)...")
    rc = subprocess.run(
        [
            *utf8_bootstrap.py_child_args(),
            str(eval_script),
            "--env-url",
            args.env_url,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(eval_path),
        ],
        cwd=str(root),
    ).returncode
    if rc != 0:
        raise SystemExit(rc)

    eval_data = _load_json(eval_path) if eval_path.is_file() else {}
    avg_random = float(eval_data.get("avg_reward_random", float("nan")))
    avg_rule = float(eval_data.get("avg_reward_rule_based", float("nan")))

    run_dir = Path(args.run_dir)
    tr_rewards, tr_losses, tr_steps = _extract_train_rewards(run_dir)
    tail = 20
    last_r = tr_rewards[-tail:] if tr_rewards else []
    train_reward_mean = _mean(last_r) if last_r else float("nan")
    last_loss = tr_losses[-tail:] if tr_losses else []
    train_loss_mean = _mean(last_loss) if last_loss else float("nan")

    r_mean_s = f"{train_reward_mean:.4f}" if tr_rewards and not math.isnan(train_reward_mean) else "- (no reward in logs)"
    l_mean_s = f"{train_loss_mean:.4f}" if tr_losses and not math.isnan(train_loss_mean) else "- (no loss in logs)"

    lines: list[str] = [
        "# Stage 4 evaluation report",
        "",
        f"- Environment: `{args.env_url}`",
        f"- Episodes per baseline policy: {args.episodes}",
        f"- Training run directory: `{run_dir}`",
        "",
        "## Baseline policies (environment rollouts)",
        "",
        f"| Policy | Avg episode reward |",
        f"|---|---:|",
        f"| Random (untrained-style) | {avg_random:.4f} |",
        f"| Rule-based heuristic | {avg_rule:.4f} |",
        "",
        "## Training run statistics (GRPO / TRL)",
        "",
        f"- Logged train reward points: {len(tr_rewards)}",
        f"- Mean train reward (last {tail} logged points): {r_mean_s}",
        f"- Mean train loss (last {tail} logged points): {l_mean_s}",
        "",
        "**Interpretation:** Random policy is a practical *before training* baseline on the live environment. ",
        "GRPO optimizes the same step reward signal; use `plot_metrics.py` on the run directory for curves, ",
        "and compare rising training reward vs the random baseline average.",
        "",
    ]

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[OK] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
