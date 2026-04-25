# ShinChan Env Runbook (Friend Handoff)

This is the complete execution guide to run, deploy, train, evaluate, and submit the project.

## 0) Prerequisites

- Windows with `py -3` available
- GitHub repo access
- Hugging Face account
- Google Colab account (GPU runtime)

## 1) Clone + Setup

```powershell
git clone https://github.com/Sarthaks-24/sinchan_env.git
cd sinchan_env

py -3 -m pip install -e .
py -3 -m pip install pytest
py -3 -m pip install "openenv-core[core]>=0.2.2"
```

## 2) Verify Locally

### 2.1 Run tests

```powershell
py -3 -m pytest -q tests/test_smoke.py
```

Expected: `3 passed`.

### 2.2 Start server

```powershell
py -3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000/web`

Quick manual checks:
- scenario loads on reset
- tool actions work
- reward updates

## 3) Deploy to Hugging Face Space

### 3.1 Create Space (one-time)

- Name: `sinchan-env`
- SDK: `Docker`
- Visibility: `Public`

### 3.2 Login from terminal

```powershell
hf auth login
```

If login fails, create a fresh write token and retry.

### 3.3 Push environment

```powershell
py -3 -m openenv push --repo-id YOUR_USERNAME/sinchan-env .
```

### 3.4 Check Space health

- Open Space runtime URL: `https://YOUR_USERNAME-sinchan-env.hf.space`
- Check logs until app is healthy

## 4) Colab Training Flow

Notebook path:
- `training/ShinChan_GRPO_Training.ipynb`

In Colab:
1. Set runtime to **T4 GPU**
2. Open notebook
3. Update `ENV_URL` to your HF Space runtime URL
4. Run all cells in order

The notebook now includes:
- dependency install
- health check (`/health`)
- training command
- evaluation JSON generation
- plot PNG generation

## 5) CLI Training (Alternative)

```powershell
set ENV_URL=https://YOUR_USERNAME-sinchan-env.hf.space
py -3 training/train_sinchan.py --env-url %ENV_URL% --max-steps 200 --dataset-size 200 --learning-rate 1e-5 --num-generations 2 --output-dir training/artifacts/run1
```

Outputs:
- `training/artifacts/run1/run_metadata.json`
- trainer logs/checkpoints

## 6) Evaluation + Plot Generation

### 6.1 Evaluation summary

```powershell
set ENV_URL=https://YOUR_USERNAME-sinchan-env.hf.space
py -3 training/evaluate_scenarios.py --env-url %ENV_URL% --episodes 10 --output training/artifacts/eval_summary.json
```

### 6.2 Plot assets

```powershell
py -3 -m pip install matplotlib
py -3 training/plot_metrics.py --run-dir training/artifacts/run1 --eval-summary training/artifacts/eval_summary.json --assets-dir assets
```

Generated:
- `assets/reward_curve_total.png`
- `assets/loss_curve.png` (if loss logs exist)
- `assets/baseline_comparison.png`

## 7) Final Submission Checklist

- [ ] HF Space link works publicly
- [ ] Colab notebook link works
- [ ] README has final links + screenshots + plots
- [ ] Before/after behavior table added
- [ ] Demo video uploaded (< 2 mins)
- [ ] Form submitted before deadline

## 8) Common Issues

### `ModuleNotFoundError: openenv`
Install dependency in the same Python:

```powershell
py -3 -m pip install "openenv-core[core]>=0.2.2"
```

### HF auth `Bad request` or rate limit
- Wait a bit
- generate fresh token
- re-login once (avoid repeated attempts)

### Space stuck in `Starting...`
- check Build + Container logs
- confirm app listening port matches runtime expectation
- restart Space once after successful build

## 9) Security Note

- Never paste tokens in shared screenshots/chats/commits.
- If exposed, revoke immediately and create a new token.
