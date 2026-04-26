#!/usr/bin/env python3
"""
Stage 1 — Environment validation: reset, random agent, log rewards and transitions.

Usage:
  python training/stage1_validate_env.py --env-url http://localhost:8000 --episodes 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from openenv.core.env_server.mcp_types import CallToolAction

try:
    from sinchan_env import SinChanEnv
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from client import SinChanEnv  # type: ignore


def _connect(env_url: str) -> SinChanEnv:
    use_http = env_url.lower().startswith("https://")
    client = SinChanEnv(base_url=env_url, prefer_http_mcp=use_http)
    return client


def run_random_episode(env, seed: int | None, episode_idx: int, log_lines: list[str]) -> dict:
    rng = random.Random((seed or 0) * 1000 + episode_idx)
    lines: list[str] = []

    env.call_tool("new_episode", seed=seed)
    info = env.call_tool("get_scenario_info")
    if not isinstance(info, dict):
        lines.append(f"[ep {episode_idx}] ERROR: get_scenario_info not a dict: {info!r}")
        return {"episode": episode_idx, "error": "bad_info", "total_reward": 0.0, "steps": 0}

    title = info.get("title", "?")
    lines.append(f"\n=== Episode {episode_idx} | {title} ===")
    lines.append(f"narrative (excerpt): {str(info.get('narrative', ''))[:200]}...")

    total_reward = 0.0
    steps = 0
    done = False
    max_steps = 8

    while not done and steps < max_steps:
        actions = info.get("available_actions") or []
        if not actions:
            lines.append("No actions; breaking.")
            break
        pick = rng.choice(actions)
        name = pick.get("name", "unknown")
        lines.append(f"  step {steps + 1}: RANDOM action={name!r}")

        step_res = env.step(
            CallToolAction(
                tool_name="choose_action",
                arguments={
                    "action_name": name,
                    "reasoning": "Random policy (stage1 validation).",
                    "dialogue": "Buri buri~ random move for testing!",
                },
            )
        )
        obs = getattr(step_res, "observation", step_res)
        r = float(getattr(step_res, "reward", getattr(obs, "reward", 0.0)) or 0.0)
        total_reward += r
        done = bool(getattr(obs, "done", False))
        result = getattr(obs, "result", {}) or {}
        if isinstance(result, dict):
            lines.append(
                f"    reward={r:.4f} done={done} status={result.get('status', '')!r}"
            )
        else:
            lines.append(f"    reward={r:.4f} done={done}")
        steps += 1
        if not done:
            info = env.call_tool("get_scenario_info")

    summary = {
        "episode": episode_idx,
        "scenario_title": title,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "done": done,
    }
    lines.append(f"  >>> episode total_reward={summary['total_reward']}")
    log_lines.extend(lines)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: validate env with a random agent.")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="training/artifacts/stage1_validation.json",
        help="JSON summary path",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    log_lines.append(f"stage1_validate_env started | env_url={args.env_url} seed={args.seed}")
    summaries: list[dict] = []

    attempts = 8
    delay_s = 4
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client = _connect(args.env_url)
            if hasattr(client, "sync"):
                with client.sync() as env:
                    for ep in range(1, args.episodes + 1):
                        summaries.append(
                            run_random_episode(
                                env, seed=args.seed + ep, episode_idx=ep, log_lines=log_lines
                            )
                        )
            else:
                env = client
                for ep in range(1, args.episodes + 1):
                    summaries.append(
                        run_random_episode(
                            env, seed=args.seed + ep, episode_idx=ep, log_lines=log_lines
                        )
                    )
            break
        except Exception as exc:
            last_err = exc
            wait = min(45, delay_s * attempt)
            log_lines.append(f"[retry {attempt}/{attempts}] {exc}; sleep {wait}s")
            time.sleep(wait)
    else:
        print("\n".join(log_lines), file=sys.stderr)
        raise RuntimeError(f"Could not connect after {attempts} attempts: {last_err}") from last_err

    payload = {
        "stage": 1,
        "env_url": args.env_url,
        "episodes": args.episodes,
        "seed": args.seed,
        "summaries": summaries,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n".join(log_lines))
    print(f"\n[OK] Wrote {out_path}")
    print("Stage 1 complete: environment responds to reset, random actions, and rewards.")


if __name__ == "__main__":
    main()
