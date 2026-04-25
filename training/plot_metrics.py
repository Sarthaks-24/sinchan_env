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

        loss_keys = ["loss", "train/loss", "policy_loss", "objective/kl"]
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

    if rewards:
        plt.figure(figsize=(9, 5))
        plt.plot(steps, rewards, marker="o", linewidth=1.5)
        plt.title("Training Reward vs Step")
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
        plt.figure(figsize=(9, 5))
        plt.plot(x, y, marker="o", linewidth=1.5, color="tab:red")
        plt.title("Training Loss vs Step")
        plt.xlabel("Step")
        plt.ylabel("Loss")
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
