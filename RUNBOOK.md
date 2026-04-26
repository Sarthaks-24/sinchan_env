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

Expected: `7 passed` (or the count printed by pytest; increase if new tests are added).

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
- **App port:** In Space **Settings → App**, set port to **7860** (must match `README` `app_port`, `Dockerfile` `EXPOSE`, and `uvicorn` — if they differ, the UI can show “space in error” while container logs look fine).

### 3.2 Login from terminal

```powershell
hf auth login
```

If login fails, create a fresh write token and retry.

### 3.3 Push environment

```powershell
py -3 -m openenv.cli push --repo-id Gladiator-codes/sinchan-env .
```

If `openenv` is on your `PATH`, you can use `openenv push ...` instead. Do **not** use `python -m openenv` — that package has no `__main__` module.

### 3.4 Check Space health

- Space card: [Gladiator-codes/sinchan-env](https://huggingface.co/spaces/Gladiator-codes/sinchan-env)
- **Web UI (Gradio + OpenEnv):** [https://gladiator-codes-sinchan-env.hf.space/web/](https://gladiator-codes-sinchan-env.hf.space/web/)  
- **`ENV_URL` / preflight (HTTP base only, no path):** `https://gladiator-codes-sinchan-env.hf.space`  
  (If the short URL is stuck in “space in error” while a build is healthy, try the **revision** hostname from the browser address bar; it can include a middle hash segment that **changes** after each rebuild. Copy that host for `ENV_URL` if preflight or training cannot connect.)
- Check logs until app is healthy

### 3.5 Release verification (deployment drift / “A”)

Local tests passing does **not** mean the public Space is running the same code. Before you debug Colab or clients:

1. **Hub build:** In the Space, open **Settings → App** (or the build tab) and confirm a **build completed after** the commit you care about. If there is no recent build, your changes are not live yet.
2. **Match Colab to Space:** The notebook clones `main` by default. If you need a specific fix, `git checkout <commit>` in Colab or change `REPO_URL` / use your fork.
3. **Browser checks (warm then cold if possible):**
   - `https://gladiator-codes-sinchan-env.hf.space/health` should return `200` (if you get `503`, wait 30–60s and retry; that is often sleep, not your FastAPI).
   - `.../docs` or `.../openapi.json` should load when the app is up.

### 3.6 HTTP-only connectivity (infrastructure / “B”)

Hugging Face can serve `/web` (Gradio) while **WebSocket**-based `step` traffic to `/ws` is flaky. Training and the client in this repo are intended to use **HTTP** to `/mcp` and `POST /reset` for `https://*.hf.space`.

**From Colab or any machine (copy-paste, stop at first failure):**

```python
import requests
BASE = "https://gladiator-codes-sinchan-env.hf.space".rstrip("/")
print("health", requests.get(f"{BASE}/health", timeout=20).status_code, requests.get(f"{BASE}/health", timeout=20).text[:200])
print("reset", requests.post(f"{BASE}/reset", json={}, timeout=30).status_code)
```

Or use the preflight script (classifies the same path as the client):

```powershell
py -3 training/preflight_space.py --base-url https://gladiator-codes-sinchan-env.hf.space --retries 3
```

If `/health` is `503`, retry with `--retries` or wait; that labels **A/B (wake/proxy)**, not “wrong Python in the app.”

### 3.7 When “Quick Start” in the UI fails (API drift / “C”)

- **Do not** use generated Playground text like `CallToolAction(message="...")` for this project. Tools are defined in `server/sinchan_environment.py` (e.g. `choose_action` with `action_name`, `reasoning`, `dialogue`).
- Prefer `from sinchan_env import SinChanEnv, CallToolAction` (this package re-exports names). A bare `from openenv import CallToolEnv` can fail depending on `openenv-core` version.
- For hosted URLs, use `SinChanEnv(base_url="https://...", prefer_http_mcp=True)` (default for `https` is HTTP MCP in this client).

### 3.8 Hub CLI and “deploy never happened” (“D”)

If `hf auth login` returns **rate limit** or **bad request**, wait, use a **write** token, avoid repeated logins, then run `hf whoami` once. If the Space **Build** tab shows no new build, your `git` changes never reached the image: fix push/auth first, then re-run the checks in **3.5** above.

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
set ENV_URL=https://gladiator-codes-sinchan-env.hf.space
py -3 training/train_sinchan.py --env-url %ENV_URL% --max-steps 200 --dataset-size 200 --learning-rate 1e-5 --num-generations 2 --output-dir training/artifacts/run1
```

Outputs:
- `training/artifacts/run1/run_metadata.json`
- trainer logs/checkpoints

## 6) Evaluation + Plot Generation

### 6.1 Evaluation summary

```powershell
set ENV_URL=https://gladiator-codes-sinchan-env.hf.space
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

### **All routes return `503` for a long time** (`/`, `/web`, `/health` never become `200`)

That is **not** a normal “cold start” after many minutes. The edge proxy has **no healthy container** behind it. Colab cannot fix that; fix the Space.

Do this **in order** on [huggingface.co/spaces](https://huggingface.co/spaces):

1. **Build log (Docker):** Open the Space → **Files and versions** or **Build** (or “Build logs”). Did the **last build succeed** (green)? If it **failed**, the image is wrong or missing—fix the error (dependency, Dockerfile, base image pull) and **rebuild**. No amount of retries in Colab helps.
2. **App / Container log:** Open **Logs** (runtime). Look for **Python Traceback**, `ModuleNotFoundError`, `Address already in use`, **OOM / Killed**, or the process **exiting**. If uvicorn **never** prints `Application startup complete`, the app **crashes on boot**—that matches endless `503`.
3. **Settings → Space hardware:** Confirm the Space is not **paused** (if your plan pauses unused Spaces) and that it has **enough RAM/CPU** for the Docker image (very heavy base images can OOM on small hardware).
4. **Port:** This repo’s [`Dockerfile`](Dockerfile) runs `uvicorn` on `0.0.0.0` and `${PORT:-7860}`. In Space settings, **Port** should match what HF expects (often **7860** for Docker SDK). A mismatch can leave the platform routing to nothing → `503`.
5. **Factory reboot** after a **green** build: Space menu → **Restart** / “Factory reboot” so a fresh container actually starts.
6. **Verify locally (best proof):** From your repo root, `docker build -t sinchan-test .` then `docker run --rm -e PORT=7860 -p 7860:7860 sinchan-test` and open `http://localhost:7860/health`. If that fails, fix **before** pushing to HF again.

**Until the Space returns `200` in a normal browser,** set Colab to the **local fallback** (notebook starts `uvicorn` on `127.0.0.1:8000`) or train only against a server you control.

## 9) Security Note

- Never paste tokens in shared screenshots/chats/commits.
- If exposed, revoke immediately and create a new token.
