import argparse
import json
import random
from pathlib import Path

from openenv.core.env_server.mcp_types import CallToolAction

from sinchan_env import SinChanEnv


def _random_agent_choice(actions: list[dict]) -> tuple[str, str, str]:
    action = random.choice(actions)
    return (
        action["name"],
        "Random baseline: selecting an arbitrary action.",
        "Buri buri~ I'm doing something random!",
    )


def _rule_based_choice(actions: list[dict]) -> tuple[str, str, str]:
    # Prioritize safe/responsible signals for a stronger baseline.
    preferred = ("responsible", "honest", "safe", "smart", "teamwork", "showing_empathy")
    avoid = ("dangerous", "stealing", "bullying", "selfish", "lying")

    best_score = -10
    best_action = actions[0]
    for action in actions:
        tags = action.get("tags", [])
        score = sum(2 for t in tags if t in preferred) - sum(2 for t in tags if t in avoid)
        if score > best_score:
            best_score = score
            best_action = action

    reasoning = (
        "I should pick an action that avoids harm, helps people, and keeps trust for tomorrow."
    )
    dialogue = "Ora ora~ I'll do the smart thing while still being Shin-chan!"
    return best_action["name"], reasoning, dialogue


def run_episode(env: SinChanEnv, policy: str) -> dict:
    env.reset()
    info = env.call_tool("get_scenario_info")
    steps = 0
    total_reward = 0.0
    done = False
    trajectory = []

    while not done and steps < 6:
        actions = info.get("available_actions") or []
        if not actions:
            break

        if policy == "random":
            action_name, reasoning, dialogue = _random_agent_choice(actions)
        else:
            action_name, reasoning, dialogue = _rule_based_choice(actions)

        step_res = env.step(
            CallToolAction(
                tool_name="choose_action",
                arguments={
                    "action_name": action_name,
                    "reasoning": reasoning,
                    "dialogue": dialogue,
                },
            )
        )
        obs = getattr(step_res, "observation", step_res)
        reward = float(getattr(step_res, "reward", getattr(obs, "reward", 0.0)) or 0.0)
        total_reward += reward
        done = bool(getattr(obs, "done", False))
        result = getattr(obs, "result", {}) or {}
        trajectory.append(
            {
                "step": steps + 1,
                "action": action_name,
                "reward": reward,
                "status": result.get("status", ""),
            }
        )
        steps += 1

        # Refresh available actions if scenario continues.
        if not done:
            info = env.call_tool("get_scenario_info")

    return {
        "scenario_title": info.get("title", "unknown"),
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "trajectory": trajectory,
    }


def evaluate(env_url: str, episodes: int, seed: int, output_path: Path) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {"random": [], "rule_based": []}
    client = SinChanEnv(base_url=env_url)
    sync_ctx = client.sync() if hasattr(client, "sync") else client
    with sync_ctx as env:
        for _ in range(episodes):
            results["random"].append(run_episode(env, policy="random"))
        for _ in range(episodes):
            results["rule_based"].append(run_episode(env, policy="rule_based"))

    def avg_reward(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return round(sum(row["total_reward"] for row in rows) / len(rows), 4)

    summary = {
        "env_url": env_url,
        "episodes_per_policy": episodes,
        "seed": seed,
        "avg_reward_random": avg_reward(results["random"]),
        "avg_reward_rule_based": avg_reward(results["rule_based"]),
        "results": results,
    }

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote evaluation results to: {output_path}")
    print(
        "Average rewards -> random: "
        f"{summary['avg_reward_random']} | rule_based: {summary['avg_reward_rule_based']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShinChan environment policies.")
    parser.add_argument("--env-url", default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="training/artifacts/eval_summary.json",
        help="Output JSON path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        env_url=args.env_url,
        episodes=args.episodes,
        seed=args.seed,
        output_path=Path(args.output),
    )
