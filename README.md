---
title: ShinChan Life Simulator
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Shin-chan OpenEnv RL with TRL GRPO training and Gradio UI.
---

# ShinChan Life Simulator

**OpenEnv**-based RL environment: an agent plays as Shin-chan through social / family / school dilemmas and learns to reduce harmful outcomes while staying in character. Training uses **Hugging Face TRL GRPO** with optional **QLoRA (4-bit + PEFT)** for low memory.

| Resource | Link |
|----------|------|
| **Hugging Face Space (cards)** | [https://huggingface.co/spaces/Gladiator-codes/sinchan-env](https://huggingface.co/spaces/Gladiator-codes/sinchan-env) |
| **Space runtime (primary UI—best UX)** | [https://gladiator-codes-sinchan-env.hf.space/play](https://gladiator-codes-sinchan-env.hf.space/play) → `/sinchan-ui/` |
| **Space runtime (OpenEnv lab / Gradio)** | [https://gladiator-codes-sinchan-env.hf.space/web/](https://gladiator-codes-sinchan-env.hf.space/web/) · [`/gradio`](https://gladiator-codes-sinchan-env.hf.space/gradio) |
| **Colab (end-to-end training)** | [Open submission notebook in Google Colab](https://colab.research.google.com/drive/1BF3I0M1Md2hG_rq7oldPVD-dJHsdLl0g?usp=sharing) (source: `training/ShinChan_GRPO_Training.ipynb` in this repo) |
| **Repository** | [https://github.com/Sarthaks-24/sinchan_env](https://github.com/Sarthaks-24/sinchan_env) |
| **Mini blog (served on the Space)** | [https://gladiator-codes-sinchan-env.hf.space/blog.md](https://gladiator-codes-sinchan-env.hf.space/blog.md) |
| **Blog source in repo** | [`blog.md`](https://github.com/Sarthaks-24/sinchan_env/blob/main/blog.md) (same text as `GET /blog.md` on the Space) |

> **No large model weights in git.** Upload checkpoints to a Hub model repo and link them here; this repo only stores scripts, small JSON/MD evidence, and small PNGs.

---

## 1. Problem motivation

Short-horizon social dilemmas have **conflicting incentives** (fun now vs trust tomorrow). A policy must trade off personality (dialogue) against outcomes (rewards, relationships). The environment makes those tradeoffs explicit and teaches an LLM to act through **tool calls** (`choose_action`).

---

## 2. Environment design

- **MCP / HTTP** via OpenEnv: tools `new_episode`, `get_scenario_info`, `get_relationships`, `choose_action`.
- **Episodes** sample scenarios with multiple steps; the UI shows narrative + legal actions.
- **Concurrency:** OpenEnv `create_app` with `max_concurrent` sessions (see `server/app.py`).

---

## 3. Reward function logic (summary)

Rewards are **dense, multi-term** and returned on each `choose_action` (e.g. responsibility, relationship alignment, in-character style). The exact weights live in `server/reward_engine.py` and are echoed as `reward_components` in the tool JSON.

---

## 4. Training setup

| Item | Choice |
|------|--------|
| **Stack** | **TRL** `GRPOTrainer` + OpenEnv `environment_factory` (remote or local) |
| **Model** | `Qwen/Qwen3-0.6B` (tool-friendly chat template; small enough for Colab) |
| **Memory** | Optional **`--use-qlora`**: 4-bit + PEFT LoRA (install `peft`, `bitsandbytes`; **GPU** required) |
| **Repro** | `run_metadata.json` + `metrics.jsonl` in each run directory; `seed` in `GRPOConfig` |

**Staged pipeline (modular, debuggable):**

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `training/stage1_validate_env.py` | Reset, random policy, log transitions and rewards |
| 2 | `training/stage2_minimal_train.py` | 50-100-step-style sanity GRPO (delegates to `train_sinchan.py`) |
| 3 | `training/stage3_full_train.py` | Longer run -> `training/artifacts/run1` (default) |
| 4 | `training/stage4_evaluate.py` | Baselines (random vs rule) + report vs training logs |
| All | `training/run_pipeline.py` | Runs stages in order, then `plot_metrics.py` |

**Core trainer:** `training/train_sinchan.py` (shared by Colab and local).

**Install (dev + training):**

```bash
pip install -e ".[training]"

# optional W&B: export WANDB_API_KEY=...  (local PNGs are always written via plot_metrics)
```

**Preflight the Space (HTTP, no WebSocket):**

```bash
python training/preflight_space.py --base-url https://gladiator-codes-sinchan-env.hf.space --retries 3
```

---

## 5. Results (evidence: real training)

**Committed sample charts:** `assets/reward_curve_total.png`, `assets/loss_curve.png`, and `assets/baseline_comparison.png` are produced by `training/plot_metrics.py` from a real short local/CPU run and a small eval — refresh them after longer training if you want stronger curves (see `assets/README.md`).

After a run, you should have:

- `training/artifacts/<run>/metrics.jsonl` - one JSON object per `on_log` (from `train_sinchan.py`)
- `training/artifacts/<run>/trainer_state.json` - TRL state (may include `log_history`)
- `training/artifacts/eval_summary.json` - from `training/evaluate_scenarios.py`
- **Plots** in `assets/` (generate; then commit the PNGs if required):

```bash
python training/plot_metrics.py --run-dir training/artifacts/run1 --eval-summary training/artifacts/eval_summary.json --assets-dir assets
```

| Plot | File |
|------|------|
| Reward curve | `assets/reward_curve_total.png` |
| Loss curve | `assets/loss_curve.png` |
| Baseline bar chart | `assets/baseline_comparison.png` |

---

## 6. Before vs after (how to read it)

| View | "Before / untrained" | "After / trained signal" |
|------|------------------------|---------------------------|
| **Env rollouts** | **Random** average reward in `eval_summary.json` | **Rule-based** average (informed baseline) in the same file |
| **RL optimization** | N/A | **Training reward** from `metrics.jsonl` / `log_history` (trend up vs random baseline) |

Per-scenario numbers change every episode sample; the **reported aggregates** in `eval_summary.json` and the **training curves** are the stable submission evidence.

| Scenario (example) | Before (random policy) | After (see evidence) |
|--------------------|------------------------|------------------------|
| Last Chocobi | *from `eval_summary.json` runs* | *training curve + last-K mean reward* |
| Homework Dilemma | ^ | ^ |
| Broken Window Trouble | ^ | ^ |
| Teacher in Tears | ^ | ^ |
| Candy from a Stranger | ^ | ^ |

Fill the numeric columns after you run `stage4_evaluate.py` and paste from `eval_summary` + your training summary.

---

## 7. Hugging Face Space / UI

- **Main runtime UI (recommended):** **`/play`** — redirects to **`/sinchan-ui/`** (crayon-style experience; use this for demos and the “best” end-user UI).
- **OpenEnv lab:** `/web` (when `ENABLE_WEB_INTERFACE` is on) — full OpenEnv playground.
- **Gradio (simple state / action / reward):** **`/gradio`** — new episode, action, state, **reward** JSON.
- **Probes:** `/health` for load checks.

**Deploy:** from repo root, use OpenEnv push (or your existing Docker Space):

```bash
openenv push --repo-id Gladiator-codes/sinchan-env .
```

**Blog on Space:** `GET /blog.md` (serves repository root `blog.md` when present, else `server/static/blog.md`).

### Docker Space: generic Hub page / "app not loading"

This repository's Space metadata uses **`sdk: docker`** (see the YAML front matter at the top of this file). **Hugging Face does not look for a root `app.py` with `gr.Interface(...).launch()`** here -- that pattern is for **Gradio-SDK** Spaces. In a Docker Space, the **entrypoint is the `CMD` in the `Dockerfile`** at the repo root: it runs **Uvicorn** on `server.app:app`.

If the Space shows a **generic Hugging Face shell** or your UI never appears:

1. Open **Build logs** and **Runtime (container) logs** -- import errors, missing dependencies, or a crash on startup mean the proxy has nothing to route to; fix those first.
2. In the Space **Settings -> App**, set the **port** to **7860** so it matches `app_port` in this README, `openenv.yaml`, `EXPOSE` in the `Dockerfile`, and the `PORT` your container listens on.
3. After deploy, wait out a cold start (free tier can take a few minutes), then check `GET /health` (should be `200`) and open **`/play`** (primary UI) or **`/gradio`** / **`/web`** on the same host — the main UIs are **not** necessarily at the repo name's short URL without a path.
4. **Local proof** the image runs: from the repo root, `docker build -t sinchan-test .` then `docker run --rm -e PORT=7860 -p 7860:7860 sinchan-test` and open `http://127.0.0.1:7860/health`.

---

## 8. Local quickstart

```bash
pip install -e .
# PowerShell: $env:ENV_URL="http://127.0.0.1:8000"
uv run server
# Open: http://localhost:8000/play  |  http://localhost:8000/web  |  http://localhost:8000/gradio  |  http://localhost:8000/blog.md
```

```bash
python -m pytest -q tests/test_smoke.py
```

**Client example (hosted `https://`):** use `SinChanEnv(..., prefer_http_mcp=True)` (see `client.py`).

---

## 9. Unsloth

Primary instruction path is **TRL** (per hackathon). **Unsloth** is optional for fast LoRA on supported models; this repo is wired for **GRPO + OpenEnv** via TRL. If you add an Unsloth path, keep it in a *separate* script so TRL/Colab stays the default.

---

## 10. After you finish: next actions

Step-by-step post-submission checks are in **`WHAT_TO_DO_NEXT.md`** in the repository root (deployment, Colab, plots, Hub).

---

*Commit small PNG/JSON only; do not store multi-GB checkpoints in git.*
