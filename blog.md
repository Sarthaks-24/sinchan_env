# Shin-chan Life Simulator - deeper hackathon write-up

**Served on the Space:** [https://gladiator-codes-sinchan-env.hf.space/blog.md](https://gladiator-codes-sinchan-env.hf.space/blog.md)

This post explains what we built, *why* the design looks this way, and how the training loop, rewards, and deployment surface fit together.

---

## 1. Problem we are actually solving

Many interesting decisions are not one-shot multiple-choice questions. They are **short stories**: you have a few turns, the situation moves, and what felt funny or harmless in turn one can create fallout in turn three. In this project, an agent plays as Shin-chan in **family / school / social** mini-scenarios. The goals pull in different directions:

- **Stay in character** (the policy should sound like the character, not like a safety brochure).
- **Limit harm** to others and to the character's long-term relationships (grades, trust, risk).

That is a **constrained control** problem, not a pure imitation game. A static "say the right thing" dataset can teach tone, but it does not give you a *closed loop* from action to outcome over several steps. We use **reinforcement learning in a tool-calling environment** so every step is an action, the simulator updates state, and the reward makes the tradeoffs **measurable**.

---

## 2. How the world is represented

Each **episode** draws a **scenario** with several legal actions per step, narrative context, and characters involved. The model does not "freeform chat" the world into existence; it picks from **tool-defined** actions and supplies short **reasoning** and **dialogue** for that pick.

The server is a **FastAPI** app (see `openenv.yaml`: `app: server.app:app`, port `7860` on the Space) built with **OpenEnv**'s `create_app` pattern. The agent interacts over **MCP-style tools** (HTTP JSON-RPC) such as `new_episode`, `get_scenario_info`, `get_relationships`, and `choose_action`. Concretely:

- **`new_episode`** resets latent relationship / progress flags and samples a new scenario.
- **`get_scenario_info`** and **`get_relationships`** expose what the model is allowed to know without cheating the next reward.
- **`choose_action`** is the only step that advances the episode and returns **observation + reward** for RL.

That boundary matters: the **same** tool names, payloads, and session behavior you use in training are the same ones the Space runs in production. There is no separate "Kaggle version" and "Space version" of the rules.

**Where to try it (same container, different UIs):**

- **`/play` -> `/sinchan-ui/`** - primary crayon-style UI; best for demos.
- **`/web`** - full OpenEnv playground (when the web interface flag is on).
- **`/gradio`** - compact state / action / reward console for quick debugging.

---

## 3. Reward design (why it is "dense" and multi-term)

The scalar reward is not a single opaque label. `server/reward_engine.py` turns each `(scenario, step, action, reasoning, dialogue)` into a **vector of components** that are summed (with penalties) into **`total`**, and the tool response surfaces those as **`reward_components`**.

Conceptually, the terms separate **what** you did, **how** you justified it, and **whether** you stayed on-brand:

- **Decision quality** - overlap with a scenario's **optimal path** and tag-based heuristics (e.g. selfish vs responsible tags) when the path is not followed exactly.
- **Social awareness** - does the **reasoning** refer to the people in the scene and to consequences, not just generic moral filler?
- **Reasoning and repetition penalties** - pushes the model away from copy-paste actions and one-line "because it is good" answers.
- **Personality** - light signals that the dialogue still looks like the character (including keyword heuristics around known motifs).
- **Responsibility, relationship impact, long-term thinking, creativity** - additional structured pressure so "funny" does not always beat "sustainable."

Dense, interpretable parts help **debugging** (you can see *which* term broke when a rollout looks wrong) and **stability** (the optimizer gets feedback every action, not only at episode end).

---

## 4. Training: GRPO + a live environment

The main training path is **Hugging Face TRL** `GRPOTrainer` in `training/train_sinchan.py`. **GRPO (Group Relative Policy Optimization)** is well suited to **turn-level** problems where you can sample multiple continuations and compare them in the *same* context: the algorithm uses **relative** quality within a group of samples rather than relying on a fixed scalar baseline from a separate critic. That lines up with tool-calling: each "turn" is a concrete action with a reward from the environment.

**Model:** `Qwen/Qwen3-0.6B` (small, tool-friendly chat template; feasible on modest GPUs and in Colab).

**QLoRA (optional):** 4-bit loading + PEFT when you need to fit training into tight VRAM. It is a **practical** knob, not a different task definition - the environment contract does not change.

**Artifacts** land under `training/artifacts/<run>/`: typically `metrics.jsonl`, `trainer_state.json`, and `run_metadata.json`. The staged scripts (`stage1` ... `stage4`, and `run_pipeline.py`) are there to keep the path **modular** - validate, short sanity run, longer run, then eval - so a failure is easier to localize than a single monolithic job.

**Plots:** `training/plot_metrics.py` turns logged curves into the PNGs under `assets/` (reward, loss, baselines) when you need figures for a write-up or deck.

---

## 5. Evaluation and what "counts" as evidence

Single episodes are **high variance**: sampling and stochastic dialogue can make one trajectory look great and another poor for the same policy. We care about **aggregates**: trends in `metrics.jsonl` and bar-level comparisons in `training/artifacts/eval_summary.json` after you run the evaluation stage (e.g. random vs rule-based baselines from `stage4_evaluate.py` / related scripts). The submission story is stronger when you show **curves + summary stats**, not a cherry-picked chat log.

---

## 6. Why this stack (opinionated but honest)

- **OpenEnv** makes the "environment as a service" idea explicit: tools, sessions, and HTTP/MCP are first-class, which is how many agents will be deployed anyway.
- **Tool-grounded RL** reduces the gap between "we trained on JSONL" and "we ship an API" - the policy sees the same action space at train and serve time.
- **GRPO** fits a regime where you get **frequent, comparable** rollouts in the same context group - exactly what a multi-step scenario with per-step rewards provides.

The uncomfortable truth is that the reward engine is still a **scaffold**: it encodes heuristics and author intent. The deeper research direction is to tighten that scaffold with human feedback, richer scenario coverage, and calibration so "personality" and "safety" do not fight each other in the loss by accident. This project is a **working end-to-end slice** of that story: a served env, a clear API, a training loop, and eval hooks you can actually run.

---

## 7. Quick map of the repository

| Area | Role |
|------|------|
| `server/app.py`, `openenv.yaml` | HTTP app + OpenEnv integration |
| `server/reward_engine.py` | Per-step reward decomposition |
| `server/sinchan_environment.py` | Env wiring to MCP / tools |
| `training/train_sinchan.py` | TRL GRPO + env factory |
| `training/stage*.py` / `run_pipeline.py` | Staged train and eval |
| `client.py` | `SinChanEnv` client for local or remote |

If you are reading this on the Space, the file you are viewing is the same **markdown** the repo serves at `GET /blog.md` (repository root), so updates deploy with the next Docker build.
