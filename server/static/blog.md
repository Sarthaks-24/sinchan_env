# Shin-chan life simulator — mini blog

**Hosted on the Hugging Face Space (same host as the OpenEnv server).**  
This file is the hackathon “mini write-up” you can link from the project README:  
https://gladiator-codes-sinchan-env.hf.space/blog.md (OpenEnv / Gradio: https://gladiator-codes-sinchan-env.hf.space/web/)

## What is this environment?

A social/family *choose-your-outcome* loop: Shin-chan faces dilemmas, picks an action, and gets a **dense reward** (relationships, responsibility, in-character behavior). The goal of RL here is to learn a policy that keeps Shin-chan’s personality while **reducing bad outcomes** on average.

## Why OpenEnv + TRL GRPO?

- **OpenEnv** exposes the world as **HTTP/MCP tools** so you can train against a *live* environment (local or on a Space).
- **TRL GRPO** fits **tool-calling** LLMs: the policy proposes `choose_action` calls; rewards come from the same engine judges use in evaluation.

## What I actually trained

- **Model:** `Qwen/Qwen3-0.6B` (small, reliable on Colab; optional **QLoRA** for memory).
- **Signal:** per-step `reward` / `reward_components` returned by the environment.
- **Evidence:** `training/artifacts/**` in the git repo: `run_metadata.json`, `metrics.jsonl`, `trainer_state.json`, plus `assets/reward_curve_total.png` and `assets/loss_curve.png` from `plot_metrics.py`.

## How to verify the Space

1. Open `/gradio` for a simple human UI (state, action, reward).  
2. Open `/web` for the OpenEnv playground.  
3. `GET /health` should return `{"status": "ok"}`.

## Honest note

This post is a **short narrative + pointers to artifacts**; the README and the Colab notebook document exact commands, seeds, and where plots are produced.

---

*Update this file on the Hub with your run IDs and Space URL when you publish.*
