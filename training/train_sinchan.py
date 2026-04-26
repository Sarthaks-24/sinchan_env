import argparse
import math
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows: TRL loads UTF-8 .jinja; cp1252 causes UnicodeDecodeError (e.g. byte 0x9d).
_train_dir = Path(__file__).resolve().parent
if str(_train_dir) not in sys.path:
    sys.path.insert(0, str(_train_dir))
import utf8_bootstrap  # noqa: E402

utf8_bootstrap.ensure_utf8_text_mode()

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

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


def _detect_world_size() -> int:
    """
    Best-effort process count for GRPOConfig (matches TRL's self.world_size at init).

    Single-process Colab is 1; `accelerate` / torch.distributed set WORLD_SIZE.
    """
    raw = os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def _normalize_grpo_batch_shape(
    *,
    per_device_train_batch_size: int,
    steps_per_generation: int,
    num_generations: int,
    world_size: int,
) -> int:
    """
    Ensure generation_batch_size is divisible by num_generations.

    TRL computes:
      generation_batch_size = per_device_train_batch_size * steps_per_generation * world_size
    and requires generation_batch_size % num_generations == 0.

    Minimally increase steps_per_generation (preferred over raising per_device batch,
    which would spawn more parallel envs for remote Spaces).
    """
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")

    base = per_device_train_batch_size * world_size * steps_per_generation
    if base % num_generations == 0:
        return steps_per_generation

    # Smallest integer factor so (base * factor) % num_generations == 0.
    factor = num_generations // math.gcd(base, num_generations)
    new_steps = steps_per_generation * factor
    new_base = per_device_train_batch_size * world_size * new_steps
    print(
        "Adjusted steps_per_generation to satisfy TRL: "
        f"generation_batch_size (= {per_device_train_batch_size} * {world_size} * steps) "
        f"must be divisible by num_generations={num_generations}. "
        f"Was batch_size={base}; now steps_per_generation={new_steps} -> batch_size={new_base}.",
        file=sys.stderr,
    )
    return new_steps


def _qlora_config(args: argparse.Namespace, precision_flags: dict[str, bool]) -> tuple[dict | None, object | None]:
    """
    Return (model_init_kwargs, peft_config) for QLoRA, or (None, None) if disabled.
    """
    if not args.use_qlora:
        return None, None
    if precision_flags.get("use_cpu"):
        print(
            "ERROR: --use-qlora requires a GPU runtime (bitsandbytes 4-bit). "
            "Omit --use-qlora on CPU or pass GPU flags.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        import torch
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        print(
            "ERROR: QLoRA needs `peft`, `bitsandbytes`, and a working torch build.\n"
            "  pip install peft bitsandbytes",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    compute_dtype = torch.bfloat16 if precision_flags.get("bf16") else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_init_kwargs: dict = {
        "quantization_config": bnb,
        "device_map": "auto",
        "trust_remote_code": False,
    }
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model_init_kwargs, peft_config


def _normalize_num_generations(num_generations: int) -> int:
    """TRL GRPO requires at least two samples per prompt group for advantages."""
    if num_generations >= 2:
        return num_generations
    print(
        f"Adjusted num_generations from {num_generations} to 2 "
        "(TRL GRPO requires num_generations >= 2).",
        file=sys.stderr,
    )
    return 2


def _coerce_numeric_dict(value: object) -> dict[str, float]:
    """Keep only numeric entries from a metadata dictionary."""
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            cleaned[str(key)] = float(raw)
    return cleaned


class _MetricsJsonlCallback(TrainerCallback):
    """Append each on_log step to output_dir/metrics.jsonl for reliable plotting."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._path = output_dir / "metrics.jsonl"

    def on_train_begin(self, args, state, control, **kwargs):
        if self._path.exists():
            self._path.unlink()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        row = {"step": int(state.global_step or 0)}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))})
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return control


class _TrainingProgressCallback(TrainerCallback):
    """Show live training progress in notebooks/Colab while GRPO runs."""

    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self._bar = None
        self._last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if tqdm is None:
            return control
        self._bar = tqdm(
            total=self.total_steps,
            desc="GRPO training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._bar is None:
            return control
        current_step = min(int(state.global_step or 0), self.total_steps)
        if current_step > self._last_step:
            self._bar.update(current_step - self._last_step)
            self._last_step = current_step
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._bar is None or not logs:
            return control
        summary_parts = []
        for key in ("loss", "reward", "learning_rate"):
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    summary_parts.append(f"{key}={value:.4g}")
                else:
                    summary_parts.append(f"{key}={value}")
        if summary_parts:
            self._bar.set_postfix_str(" | ".join(summary_parts))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.n = self.total_steps
            self._bar.close()
            self._bar = None
        return control


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

    def _reconnect(self) -> None:
        """Recreate client/session after transient network failures."""
        self._close()
        self._client = None
        self._sync_cm = None
        self.env = self._connect_with_retry(self.base_url)

    def _call_tool_with_retry(self, tool_name: str, **kwargs):
        """
        Call a tool with retries for intermittent HF Space transport failures.
        """
        attempts = 4
        last_err = None
        for idx in range(1, attempts + 1):
            try:
                return self.env.call_tool(tool_name, **kwargs)
            except Exception as exc:
                last_err = exc
                if idx == attempts:
                    break
                wait = min(20, 2 * idx)
                print(
                    f"[tool retry {idx}/{attempts}] {tool_name} failed: {exc}. "
                    f"Reconnecting in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                try:
                    self._reconnect()
                except Exception as reconnect_exc:
                    last_err = reconnect_exc
        raise RuntimeError(f"Tool call failed after retries: {tool_name}. Last error: {last_err}")

    def _close(self) -> None:
        """Release client resources cleanly."""
        try:
            if self._sync_cm is not None:
                self._sync_cm.__exit__(None, None, None)
            elif hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass

    def __del__(self):
        self._close()

    def reset(self, **kwargs) -> str | None:
        result = None
        seed = kwargs.get("seed")
        try:
            if seed is not None:
                self._call_tool_with_retry("new_episode", seed=seed)
            else:
                self._call_tool_with_retry("new_episode")
        except Exception:
            try:
                self._http_reset()
            except Exception:
                # Fallback for local/dev servers where client reset is preferred.
                try:
                    result = self.env.reset()
                except Exception:
                    self._reconnect()
                    result = self.env.reset()
        self.reward = 0.0
        self.done = False
        self.reward_components = {}

        try:
            info = self._call_tool_with_retry("get_scenario_info")
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

        Args:
            action_name: Name of the selected action from `available_actions`.
            reasoning: Short explanation of why Shin-chan chose this action.
            dialogue: In-character Shin-chan line spoken while taking the action.

        Returns:
            Human-readable consequence text used as the tool observation.
        """
        if self.done:
            raise ValueError("Episode is already over!")

        try:
            response_dict = self._call_tool_with_retry(
                "choose_action",
                action_name=action_name,
                reasoning=reasoning,
                dialogue=dialogue,
            )
        except Exception as exc:
            # Keep training alive across occasional remote errors.
            self.reward_components = {"total": -1.0}
            self.reward = -1.0
            self.done = True
            return (
                f"Tool failure: {exc}\n\n"
                "Consequences: Connection issue while contacting environment.\n\n"
                "Status: episode terminated with fallback penalty."
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
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="4-bit QLoRA via PEFT (install bitsandbytes + peft; requires a CUDA GPU).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if args.per_device_train_batch_size < 1 or args.steps_per_generation < 1:
        print(
            "ERROR: --per-device-train-batch-size and --steps-per-generation must be >= 1.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.num_generations < 1:
        print("ERROR: --num-generations must be >= 1.", file=sys.stderr)
        raise SystemExit(1)

    _require_trl_openenv_stack()
    _assert_tokenizer_supports_grpo_tools(args.model)
    # Quieter TRL experimental warnings for environment_factory
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision_flags = _resolve_precision(args.precision, force_cpu=bool(args.use_cpu))
    model_init_kwargs, peft_config = _qlora_config(args, precision_flags)
    world_size = _detect_world_size()
    num_generations_effective = _normalize_num_generations(args.num_generations)
    normalized_steps_per_generation = _normalize_grpo_batch_shape(
        per_device_train_batch_size=args.per_device_train_batch_size,
        steps_per_generation=args.steps_per_generation,
        num_generations=num_generations_effective,
        world_size=world_size,
    )

    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_url": args.env_url,
        "model": args.model,
        "dataset_size": args.dataset_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "num_generations": num_generations_effective,
        "num_generations_requested": args.num_generations,
        "num_generations_effective": num_generations_effective,
        "detected_world_size": world_size,
        "steps_per_generation_requested": args.steps_per_generation,
        "steps_per_generation_effective": normalized_steps_per_generation,
        "grad_accum": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
        "precision": args.precision,
        "resolved_precision_flags": precision_flags,
        "use_qlora": bool(args.use_qlora),
        "has_peft_config": peft_config is not None,
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

    grpo_kwargs: dict = dict(
        output_dir=str(output_dir),
        use_vllm=bool(args.use_vllm),
        use_cpu=precision_flags["use_cpu"],
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        steps_per_generation=normalized_steps_per_generation,
        max_completion_length=args.max_completion_length,
        num_generations=num_generations_effective,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        eval_strategy="no",
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    if model_init_kwargs is not None:
        grpo_kwargs["model_init_kwargs"] = model_init_kwargs

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=decision_reward,
        train_dataset=dataset,
        args=GRPOConfig(**grpo_kwargs),
        peft_config=peft_config,
        callbacks=[
            _MetricsJsonlCallback(output_dir),
            _TrainingProgressCallback(args.max_steps),
        ],
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
