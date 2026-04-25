import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

try:
    from sinchan_env import SinChanEnv
except ModuleNotFoundError:
    # Colab fallback when package is not installed but project files are present.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from client import SinChanEnv

# Default environment URL (can be overridden by CLI flag or ENV var).
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
VERBOSE_REWARDS = os.environ.get("VERBOSE_REWARDS", "0") == "1"


def _assert_tokenizer_supports_grpo_tools(model_id: str) -> None:
    """
    TRL raises ValueError at GRPOTrainer init if the tokenizer chat template
    cannot render user → assistant(tool_calls) → tool. Qwen2.5-Instruct often
    fails; Qwen3 base/instruct templates include tool branches.
    """
    try:
        from transformers import AutoTokenizer
        from trl.chat_template_utils import supports_tool_calling
    except ImportError as exc:
        print(
            "ERROR: Need a recent `trl` with `chat_template_utils.supports_tool_calling`.\n"
            "  pip install -U trl transformers\n"
            f"  Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    except Exception as exc:
        print(
            f"ERROR: Could not load tokenizer for {model_id!r}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if supports_tool_calling(tok):
        return

    print(
        "ERROR: This model's chat template does not support TRL's GRPO tool-calling format.\n"
        "  GRPOTrainer requires templates that render assistant tool_calls and tool messages.\n"
        "  Try: --model Qwen/Qwen3-0.6B   (or another model whose tokenizer passes TRL's check)\n"
        f"  Model requested: {model_id}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _require_trl_openenv_stack() -> None:
    """
    TRL's GRPOTrainer with environment_factory needs a new enough transformers
    and the jmespath extra; see huggingface/trl grpo_trainer.py.
    """
    import importlib.util
    import sys

    import re

    import transformers

    def _parse_semver(s: str) -> tuple[int, int, int]:
        m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", s)
        if not m:
            return 0, 0, 0
        a, b, c = m.group(1), m.group(2), m.group(3) or "0"
        return int(a), int(b), int(c)

    if _parse_semver(transformers.__version__) < (5, 2, 0):
        print(
            "ERROR: GRPO + OpenEnv needs transformers>=5.2.0 (TRL requirement).\n"
            "  pip install -U 'transformers>=5.2.0'\n"
            f"  Current: {transformers.__version__}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if importlib.util.find_spec("jmespath") is None:
        print(
            "ERROR: pip install jmespath  (required by TRL when using environment_factory).",
            file=sys.stderr,
        )
        raise SystemExit(1)


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


def _resolve_precision(precision: str, force_cpu: bool = False) -> dict[str, bool]:
    """
    Resolve TrainingArguments precision flags for the current runtime.

    On many Colab runtimes, bf16 is not available and GRPOConfig raises:
    "Your setup doesn't support bf16/gpu".
    """
    import torch

    has_cuda = torch.cuda.is_available() and not force_cpu
    if not has_cuda:
        return {"use_cpu": True, "bf16": False, "fp16": False}

    if precision == "bf16":
        return {"use_cpu": False, "bf16": True, "fp16": False}
    if precision == "fp16":
        return {"use_cpu": False, "bf16": False, "fp16": True}
    if precision == "fp32":
        return {"use_cpu": False, "bf16": False, "fp16": False}

    # auto
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_supported:
        return {"use_cpu": False, "bf16": True, "fp16": False}
    return {"use_cpu": False, "bf16": False, "fp16": True}


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
        self.base_url = base_url.rstrip("/")
        self._client = None
        self._sync_cm = None
        self.env = self._connect_with_retry(base_url)
        self.reward = 0.0
        self.done = False
        self.reward_components: dict[str, float] = {}

    def _connect_with_retry(self, base_url: str):
        """
        HF Spaces can return transient 503 while waking up.
        Retry a few times before failing hard.
        """
        import requests
        
        attempts = 8
        delay_s = 5
        last_err = None
        
        # 1. Wake up the space via HTTP GET if possible
        health_url = f"{base_url.rstrip('/')}/health"
        for idx in range(1, attempts + 1):
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(min(10, delay_s * idx))
            
        # 2. Validate MCP tool connectivity (HTTP-first in hosted setups)
        for idx in range(1, attempts + 1):
            try:
                use_http = base_url.lower().startswith("https://")
                client = SinChanEnv(base_url=base_url, prefer_http_mcp=use_http)
                if hasattr(client, "sync"):
                    sync_cm = client.sync()
                    env = sync_cm.__enter__()
                    # Validate tool listing without requiring websocket step traffic.
                    env.list_tools(use_cache=False)
                    self._client = client
                    self._sync_cm = sync_cm
                    return env
                client.list_tools(use_cache=False)
                self._client = client
                return client
            except Exception as exc:
                last_err = exc
                wait = min(30, delay_s * idx)
                print(
                    f"[connect retry {idx}/{attempts}] MCP not ready at {base_url}. "
                    f"Waiting {wait}s. Error: {exc}"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to connect to environment after {attempts} attempts. Last error: {last_err}"
        )

    def _http_reset(self) -> None:
        """Reset via HTTP endpoint to avoid websocket dependency."""
        import requests

        reset_url = f"{self.base_url}/reset"
        response = requests.post(reset_url, json={}, timeout=15)
        response.raise_for_status()

    def close(self) -> None:
        """Release client resources cleanly."""
        try:
            if self._sync_cm is not None:
                self._sync_cm.__exit__(None, None, None)
            elif hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def reset(self, **kwargs) -> str | None:
        result = None
        try:
            self._http_reset()
        except Exception:
            # Fallback for local/dev servers where client reset is preferred.
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

        response_dict = self.env.call_tool(
            "choose_action",
            action_name=action_name,
            reasoning=reasoning,
            dialogue=dialogue,
        )

        response_dict = response_dict if isinstance(response_dict, dict) else {}
        self.reward_components = _coerce_numeric_dict(
            response_dict.get("reward_components")
        )
        fallback_reward = float(response_dict.get("reward", 0.0) or 0.0)
        self.reward = float(self.reward_components.get("total", fallback_reward))
        self.done = bool(response_dict.get("done", False))

        if VERBOSE_REWARDS and self.reward_components:
            print(f"[reward_components] {json.dumps(self.reward_components, ensure_ascii=True)}")

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
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF model id (must pass TRL tool-calling chat template check; Qwen3 recommended)",
    )
    parser.add_argument("--output-dir", default="sinchan-grpo-model", help="Training output directory")
    parser.add_argument("--dataset-size", type=int, default=100, help="Number of repeated prompts")
    parser.add_argument("--max-steps", type=int, default=50, help="GRPO max training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=2, help="Completions per prompt")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-completion-length", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Training precision. 'auto' picks bf16 if supported, else fp16 on CUDA, else CPU fp32.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU mode (disables bf16/fp16). Useful for unsupported Colab runtimes.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Keep at 1 for remote HF Space (TRL creates one env per generation slot).",
    )
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=1,
        help="GRPOConfig steps_per_generation; use 1 unless you know you need more.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    _require_trl_openenv_stack()
    _assert_tokenizer_supports_grpo_tools(args.model)
    # Quieter TRL experimental warnings for environment_factory
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision_flags = _resolve_precision(args.precision, force_cpu=bool(args.use_cpu))

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
        "precision": args.precision,
        "resolved_precision_flags": precision_flags,
        "git_commit": _safe_git_commit(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Connecting to environment at: {args.env_url}")
    print(f"Saving run metadata to: {output_dir / 'run_metadata.json'}")
    print(f"Resolved precision flags: {precision_flags}")

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
            use_cpu=precision_flags["use_cpu"],
            bf16=precision_flags["bf16"],
            fp16=precision_flags["fp16"],
            per_device_train_batch_size=args.per_device_train_batch_size,
            steps_per_generation=args.steps_per_generation,
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
    try:
        train(parse_args())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from None
