# Shin-chan Life Simulator — hackathon write-up

**Live on Space:** [https://gladiator-codes-sinchan-env.hf.space/blog.md](https://gladiator-codes-sinchan-env.hf.space/blog.md)

---

## Problem

Short-horizon social and family dilemmas have **conflicting incentives** (immediate fun vs trust, grades, or safety). A useful policy must keep Shin-chan’s personality while **reducing harmful outcomes**. The project frames this as an **OpenEnv** tool-calling loop: the model chooses actions through `choose_action`, and the environment returns **dense rewards** that make tradeoffs explicit.

## Environment

The server exposes **HTTP/MCP tools** (`new_episode`, `get_scenario_info`, `get_relationships`, `choose_action`) via **OpenEnv** `create_app`. Episodes sample multi-step scenarios; rewards and `reward_components` come from `server/reward_engine.py`. For the **main runtime UI** (strongest art/UX), open **`/play`** (redirects to **`/sinchan-ui/`**). The OpenEnv playground is at **`/web`**; a compact **Gradio** console is at **`/gradio`**.

## Training

The default training path is **Hugging Face TRL** `GRPOTrainer` against the live environment (`training/train_sinchan.py`), with **`Qwen/Qwen3-0.6B`** and optional **QLoRA** for memory. The public Colab notebook runs the same stack end-to-end. Metrics land under `training/artifacts/<run>/` (`metrics.jsonl`, `trainer_state.json`, `run_metadata.json`); plots are generated with `training/plot_metrics.py` into `assets/`.

## Results

Report **aggregate** evidence: training **reward / loss trends** from `metrics.jsonl` (and plots `assets/reward_curve_total.png`, `assets/loss_curve.png` after you run the plot script), plus **evaluation aggregates** from `training/artifacts/eval_summary.json` when you run the eval stage. Per-episode samples vary; submission strength comes from **curves + eval summary**, not single rollouts.

## Why it matters

**Tool-grounded RL** on a **served environment** is closer to real deployment than a static dataset: the same API you train on is what the Space runs. OpenEnv makes that boundary clear; GRPO fits **tool-calling** models where each step is an action with measurable feedback.
