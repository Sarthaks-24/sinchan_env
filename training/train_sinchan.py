import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from sinchan_env import SinChanEnv

# Default environment URL (can be overridden by CLI flag or ENV var).
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
VERBOSE_REWARDS = os.environ.get("VERBOSE_REWARDS", "0") == "1"


def _safe_git_commit() -> str:
    """Best-effort git commit hash retrieval for run metadata."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _set_seed(seed: int) -> None:
    """Set deterministic seeds where available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _coerce_numeric_dict(value: object) -> dict[str, float]:
    """Keep only numeric entries from a metadata dictionary."""
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            cleaned[str(key)] = float(raw)
    return cleaned


class SinChanToolEnv:
    """Wrapper that adapts SinChanEnv for TRL's environment_factory API."""

    def __init__(self, base_url: str):
        self.env = SinChanEnv(base_url=base_url)
        self.reward = 0.0
        self.done = False
        self.reward_components: dict[str, float] = {}

    def reset(self, **kwargs) -> str | None:
        result = self.env.reset()
        self.reward = 0.0
        self.done = False
        self.reward_components = {}

        try:
            info = self.env.call_tool("get_scenario_info")
            return (
                f"SCENARIO: {info.get('title')}\n\n"
                f"{info.get('narrative')}\n\n"
                f"Available Actions: {info.get('available_actions')}"
            )
        except Exception:
            obs = getattr(result, "observation", result)
            metadata = getattr(obs, "metadata", {}) or {}
            return metadata.get("message", "New scenario loaded.")

    def choose_action(self, action_name: str, reasoning: str, dialogue: str) -> str:
        """
        Make one decision as Shin-chan and advance environment by one step.
        """
        if self.done:
            raise ValueError("Episode is already over!")

        from openenv.core.env_server.mcp_types import CallToolAction

        step_res = self.env.step(
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
        metadata = getattr(obs, "metadata", None)
        self.reward_components = _coerce_numeric_dict(metadata)

        fallback_reward = float(getattr(step_res, "reward", getattr(obs, "reward", 0.0)) or 0.0)
        self.reward = float(self.reward_components.get("total", fallback_reward))
        self.done = bool(getattr(obs, "done", False))

        if VERBOSE_REWARDS and self.reward_components:
            print(f"[reward_components] {json.dumps(self.reward_components, ensure_ascii=True)}")

        response_dict = getattr(obs, "result", {}) or {}
        return (
            f"{response_dict.get('shinchan_said', '')}\n\n"
            f"Consequences: {response_dict.get('consequences', '')}\n\n"
            f"Status: {response_dict.get('status', '')}"
        )


def decision_reward(environments, **kwargs):
    """Primary reward function used by GRPO."""
    return [float(env.reward) for env in environments]


SYSTEM_PROMPT = """You are Shin-chan Nohara (野原しんのすけ), a 5-year-old boy from Kasukabe, Japan.
You are mischievous, funny, and sometimes naughty, but deep down you care about your family and friends.

You are facing a real-life situation. Think about what would happen if you make different choices.
Consider how your actions affect Mom (Misae), Dad (Hiroshi), your baby sister (Himawari), and your friends.

Use the `choose_action` tool to make your decision. Stay in character as Shin-chan.
Rules:
1. Pick one of the available actions.
2. Explain your reasoning and consequences.
3. Say something as Shin-chan would say it (funny but thoughtful).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShinChan GRPO policy.")
    parser.add_argument("--env-url", default=ENV_URL, help="OpenEnv server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name")
    parser.add_argument("--output-dir", default="sinchan-grpo-model", help="Training output directory")
    parser.add_argument("--dataset-size", type=int, default=100, help="Number of repeated prompts")
    parser.add_argument("--max-steps", type=int, default=50, help="GRPO max training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=2, help="Completions per prompt")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-completion-length", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_url": args.env_url,
        "model": args.model,
        "dataset_size": args.dataset_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "grad_accum": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
        "git_commit": _safe_git_commit(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Connecting to environment at: {args.env_url}")
    print(f"Saving run metadata to: {output_dir / 'run_metadata.json'}")

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": SYSTEM_PROMPT + "\n\nA new adventure awaits! What do you do?"}]
            ]
            * args.dataset_size
        }
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=decision_reward,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(output_dir),
            use_vllm=bool(args.use_vllm),
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            max_steps=args.max_steps,
            seed=args.seed,
        ),
        environment_factory=lambda: SinChanToolEnv(base_url=args.env_url),
    )

    print("Starting training loop...")
    trainer.train()


if __name__ == "__main__":
    train(parse_args())
