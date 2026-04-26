---
title: ShinChan Life Simulator
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Shin-chan RL environment built with OpenEnv.
---

# ShinChan Life Simulator

OpenEnv-compliant RL environment where an agent plays as Shin-chan through social/family/school dilemmas and learns to reduce mistakes while keeping personality.

## Submission Links

- Hugging Face Space (card): [https://huggingface.co/spaces/Gladiator-codes/sinchan-env](https://huggingface.co/spaces/Gladiator-codes/sinchan-env)
- Hugging Face runtime URL: [https://gladiator-codes-sinchan-env-a446abd.hf.space](https://gladiator-codes-sinchan-env-a446abd.hf.space)
- Colab notebook (GRPO): [ShinChan_GRPO_Training.ipynb](https://colab.research.google.com/github/Sarthaks-24/sinchan_env/blob/main/training/ShinChan_GRPO_Training.ipynb)
- GitHub repository: [https://github.com/Sarthaks-24/sinchan_env](https://github.com/Sarthaks-24/sinchan_env)
- Mini-blog or video (<2 min): `TODO_ADD_PUBLIC_URL`

## Minimum Requirement Status

- [x] Uses OpenEnv (`openenv-core[core]` in `pyproject.toml`)
- [x] OpenEnv environment hosted on Hugging Face Space
- [x] TRL training script (`training/train_sinchan.py`)
- [x] Colab notebook (`training/ShinChan_GRPO_Training.ipynb`)
- [x] Deployment + connectivity runbook (`RUNBOOK.md`)
- [ ] Reward/loss plot images committed under `assets/`
- [ ] Before vs after results table filled with real run outputs
- [ ] Public mini-blog or <2 minute video link added above

## Why This Environment

- **Innovation:** Multi-step social dilemmas with conflicting incentives, not a simple toy game loop.
- **Storytelling:** Shin-chan setting makes behavior shifts easy to understand for non-technical reviewers.
- **Learnability:** Reward function is dense and multi-component, so policy updates get informative signal each step.
- **Evaluation-ready:** Includes scripted preflight + evaluation pipeline to show reward improvement.

## What The Agent Learns

- Balance short-term temptation vs long-term social consequences.
- Improve relationship-aware decision making.
- Keep in-character dialogue while reducing harmful outcomes.
- Generalize across scenario families (temptation, school, family, social conflict).

## Project Structure

```text
sinchan_env/
├── __init__.py
├── client.py
├── models.py
├── Dockerfile
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── characters.py
│   ├── scenario_data.py
│   ├── scenarios.py
│   ├── reward_engine.py
│   ├── sinchan_environment.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/
│   ├── train_sinchan.py
│   ├── preflight_space.py
│   └── ShinChan_GRPO_Training.ipynb
└── tests/
    └── test_smoke.py
```

## Local Run

```bash
pip install -e .
uv run server
```

- Open OpenEnv web: `http://localhost:8000/web`
- Open custom UI: `http://localhost:8000/play`

Run tests:

```bash
python -m pytest -q tests/test_smoke.py
```

## Deployment (Hugging Face Space)

```bash
openenv push --repo-id Gladiator-codes/sinchan-env .
```

Windows fallback if `openenv` not in PATH:

```bash
py -3 -m openenv.cli push --repo-id Gladiator-codes/sinchan-env .
```

Preflight check (recommended before training):

```bash
python training/preflight_space.py --base-url https://gladiator-codes-sinchan-env-a446abd.hf.space --retries 3
```

## Training (TRL / GRPO)

Primary training entry points:
- Script: `training/train_sinchan.py`
- Colab: `training/ShinChan_GRPO_Training.ipynb`

Set environment URL:

```bash
# PowerShell
$env:ENV_URL = "https://gladiator-codes-sinchan-env-a446abd.hf.space"
```

Run a configurable training job:

```bash
python training/train_sinchan.py --env-url $env:ENV_URL --max-steps 200 --output-dir training/artifacts/run1
```

## Evaluation And Evidence

Generate evaluation summary:

```bash
python training/evaluate_scenarios.py --env-url $env:ENV_URL --episodes 10 --output training/artifacts/eval_summary.json
```

Generate plot assets:

```bash
python training/plot_metrics.py --run-dir training/artifacts/run1 --eval-summary training/artifacts/eval_summary.json --assets-dir assets
```

Expected submission evidence files:
- `assets/reward_curve_total.png`
- `assets/baseline_comparison.png`
- `assets/loss_curve.png` (if loss logs available)
- `training/artifacts/eval_summary.json`
- `training/artifacts/run1/run_metadata.json`

### Before vs After (Fill With Real Numbers)

| Scenario | Before Training | After Training |
|---|---:|---:|
| Last Chocobi | TODO | TODO |
| Homework Dilemma | TODO | TODO |
| Broken Window Trouble | TODO | TODO |
| Teacher in Tears | TODO | TODO |
| Candy from a Stranger | TODO | TODO |

## Judge-Criteria Mapping

- **Environment Innovation (40%)**: Rich social/family dilemma space with character relationships and curriculum progression.
- **Storytelling (30%)**: Themed environment + custom `/play` UI + clear behavior examples in table/video/blog.
- **Reward Improvement (20%)**: Reward/loss plots + baseline comparison JSON + before/after table.
- **Reward & Pipeline (10%)**: Multi-component reward engine + reproducible TRL GRPO script + notebook rerun path.

## Client Example

```python
from sinchan_env import SinChanEnv

with SinChanEnv(base_url="http://localhost:8000") as env:
    env.call_tool("new_episode")
    info = env.call_tool("get_scenario_info")
    print(info["title"])

    result = env.call_tool(
        "choose_action",
        action_name=info["available_actions"][0]["name"],
        reasoning="I should think about tomorrow and others' feelings.",
        dialogue="Buri buri~ I'll do the right thing my way!",
    )
    print(result)
```

For hosted Spaces, this client defaults to HTTP MCP for `https://` URLs.

```python
from sinchan_env import CallToolAction, SinChanEnv

# prefer_http_mcp=True is the default for https:// bases
with SinChanEnv(
    base_url="https://gladiator-codes-sinchan-env-a446abd.hf.space",
    prefer_http_mcp=True,
) as env:
    env.call_tool("new_episode")
    step_result = env.step(
        CallToolAction(
            tool_name="get_scenario_info",
            arguments={},
        )
    )
    print(step_result.observation)
```

See [RUNBOOK.md](RUNBOOK.md) for deployment/debug triage and final handoff steps.

## Final Pre-Submit Checklist

- [ ] Add public mini-blog/video URL in this README (`TODO_ADD_PUBLIC_URL`)
- [ ] Commit generated plot images in `assets/`
- [ ] Fill before/after table with real metrics
- [ ] Re-run `training/preflight_space.py` on live Space
- [ ] Verify Space + notebook links open publicly without auth
