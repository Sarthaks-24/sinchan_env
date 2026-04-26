import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: py -3 -m pip install matplotlib"
        ) from exc


def _extract_log_history(run_dir: Path) -> list[dict]:
    """Prefer `metrics.jsonl` (written by `train_sinchan.py`), else `trainer_state.json`."""
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        rows: list[dict] = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
        if rows:
            return rows

    trainer_state = run_dir / "trainer_state.json"
    if not trainer_state.exists():
        return []
    data = _load_json(trainer_state)
    history = data.get("log_history", [])
    return history if isinstance(history, list) else []


def _plot_training_curves(plt, history: list[dict], assets_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if not history:
        return outputs

    steps = []
    rewards = []
    losses = []
    entropies: list[tuple[int, float]] = []
    for row in history:
        step = row.get("step") if isinstance(row, dict) else None
        if step is None:
            continue
        try:
            step_val = int(step)
        except Exception:
            continue

        reward_keys = ["reward", "rewards/mean", "train/reward", "episode_reward", "objective"]
        reward_val = None
        for key in reward_keys:
            if key in row and isinstance(row[key], (int, float)):
                reward_val = float(row[key])
                break

        loss_keys = ["loss", "train/loss", "policy_loss", "objective/kl", "train_loss"]
        loss_val = None
        for key in loss_keys:
            if key in row and isinstance(row[key], (int, float)):
                loss_val = float(row[key])
                break

        if reward_val is not None:
            steps.append(step_val)
            rewards.append(reward_val)
        if loss_val is not None:
            losses.append((step_val, loss_val))
        if "entropy" in row and isinstance(row["entropy"], (int, float)):
            entropies.append((step_val, float(row["entropy"])))

    if rewards:
        plt.figure(figsize=(9, 5))
        plt.plot(steps, rewards, marker="o", linewidth=1.5)
        flat_r = len(rewards) > 0 and all(r == rewards[0] for r in rewards)
        plt.title(
            "Training reward vs step"
            + (" (flat — short run; use eval/baselines for env-scale signal)" if flat_r else "")
        )
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        out = assets_dir / "reward_curve_total.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        outputs.append(out)

    if losses:
        x = [xv for xv, _ in losses]
        y = [yv for _, yv in losses]
        # TRL sometimes logs 0 loss on very short GRPO runs; entropy still moves — plot it
        # so the chart is a faithful readout, not a blank line.
        if all(v == 0.0 for v in y) and entropies:
            x = [a for a, _ in entropies]
            y = [b for _, b in entropies]
            y_label = "Entropy (proxy — reported loss was 0)"
            plot_title = "Policy entropy vs step (short run; loss column was 0 in logs)"
        else:
            y_label = "Loss"
            plot_title = "Training loss vs step"
        plt.figure(figsize=(9, 5))
        plt.plot(x, y, marker="o", linewidth=1.5, color="tab:red")
        plt.title(plot_title)
        plt.xlabel("Step")
        plt.ylabel(y_label)
        plt.grid(alpha=0.3)
        out = assets_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        outputs.append(out)

    return outputs


def _plot_eval_summary(plt, eval_summary_path: Path, assets_dir: Path) -> list[Path]:
    if not eval_summary_path.exists():
        return []

    data = _load_json(eval_summary_path)
    policies = []
    avg_rewards = []
    for key in ["avg_reward_random", "avg_reward_rule_based"]:
        if key in data and isinstance(data[key], (int, float)):
            label = key.replace("avg_reward_", "")
            policies.append(label)
            avg_rewards.append(float(data[key]))

    if not policies:
        return []

    plt.figure(figsize=(7, 5))
    bars = plt.bar(policies, avg_rewards, color=["tab:gray", "tab:green"][: len(policies)])
    plt.title("Baseline Policy Comparison")
    plt.xlabel("Policy")
    plt.ylabel("Average Episode Reward")
    plt.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, avg_rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )
    out = assets_dir / "baseline_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return [out]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reward/loss/eval PNG charts for hackathon evidence."
    )
    parser.add_argument(
        "--run-dir",
        default="training/artifacts/run1",
        help="Training output directory containing trainer_state.json",
    )
    parser.add_argument(
        "--eval-summary",
        default="training/artifacts/eval_summary.json",
        help="JSON output produced by training/evaluate_scenarios.py",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Directory where PNG charts will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_summary = Path(args.eval_summary)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt = _safe_import_matplotlib()

    produced: list[Path] = []
    produced.extend(_plot_training_curves(plt, _extract_log_history(run_dir), assets_dir))
    produced.extend(_plot_eval_summary(plt, eval_summary, assets_dir))

    if produced:
        print("Generated charts:")
        for path in produced:
            print(f"- {path}")
    else:
        print(
            "No charts generated. Ensure trainer_state.json and/or eval_summary.json exist "
            "at the provided paths."
        )


if __name__ == "__main__":
    main()
