# What to do next (post-project checklist)

After this repository and Space are in good shape, follow these steps in order. Check each box as you go.

## A. Local sanity (5–10 minutes)

1. [ ] In the project root, create/activate a venv and run `pip install -e ".[dev,training]"` (or `uv sync` if you use the lockfile).
2. [ ] Start the app: `uv run server` (or `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000` with `PYTHONPATH` set to the repo root if needed).
3. [ ] Open in a browser:
   - [ ] `http://localhost:8000/health` → `{"status":"ok"}` (or equivalent)
   - [ ] `http://localhost:8000/gradio` → Gradio UI loads; **New episode** + one action returns JSON with a numeric **reward**
   - [ ] `http://localhost:8000/blog.md` → markdown loads (same content you link from the README)
4. [ ] `python -m pytest -q tests/test_smoke.py` passes.
5. [ ] (Optional) `python training/stage1_validate_env.py --env-url http://127.0.0.1:8000` while the server runs — you should see printed transitions and `training/artifacts/stage1_validation.json` created.

## B. Hugging Face Space deployment

1. [ ] `hf auth login` on the machine you deploy from.
2. [ ] From the **repository root** (where `openenv.yaml` and `Dockerfile` live), run:  
   `openenv push --repo-id <your-username-or-org>/<your-space> .`  
   (On Windows, `py -3 -m openenv.cli push ...` if the CLI is not on `PATH`.)
3. [ ] In the Space **Settings**, confirm: Docker build, **port 7860** (or the port your `CMD` / `PORT` uses), and enough CPU/RAM to boot.
4. [ ] When the build finishes, open the **Space URL** in a new tab to wake the container. Cold starts on free tier can take **1–3+ minutes** and may return 503 on `/health` briefly.
5. [ ] Copy the **runtime** base URL (the one that actually loads the UI, often `https://<name>-<hash>.hf.space` if the short URL errors).
6. [ ] Run:  
   `python training/preflight_space.py --base-url <RUNTIME_BASE_URL> --retries 5`  
   from this repo. All checks should pass.
7. [ ] In the browser, verify: `<RUNTIME_BASE_URL>/gradio`, `<RUNTIME_BASE_URL>/blog.md`, and `<RUNTIME_BASE_URL>/web` (or your chosen OpenEnv route).

## C. Colab: end-to-end training proof

1. [ ] Open `training/ShinChan_GRPO_Training.ipynb` in **Google Colab** (link in README).
2. [ ] **Runtime → Change runtime type → GPU** (T4 is enough). CPU training is possible but very slow and QLoRA will not work.
3. [ ] First cell: updates `openenv-core[core]>=0.2.3` and installs **TRL, transformers, jmespath, matplotlib** (and for QLoRA: `peft`, `bitsandbytes` when you add the optional cell).
4. [ ] Set `ENV_URL` to the **Space runtime** URL from B.5 (not the `huggingface.co/spaces/...` repo page).
5. [ ] Run the connectivity probe cell until it succeeds (or the notebook’s local fallback, if you use it).
6. [ ] Run **Stage 1** (optional) `stage1_validate_env.py` to confirm the remote environment responds.
7. [ ] Run **Stage 2** (short GRPO) first if you need a **quick proof**; then **Stage 3** for a longer run. Confirm `tqdm` / logs stream (do not `capture_output=True` in subprocess).
8. [ ] After training, run **eval** and **plot** cells so `assets/reward_curve_total.png` and `assets/loss_curve.png` exist. Open the files in the Colab file tree to verify they are **non-placeholder** (axes, multiple points).
9. [ ] If you use **W&B**, set `WANDB_API_KEY` in Colab; otherwise rely on the saved PNGs + JSON/JSONL in `training/artifacts/`.
10. [ ] **Download** `training/artifacts/run1/`, `training/artifacts/eval_summary.json`, and `assets/*.png` to your machine if you will commit them to Git.

## D. Repository hygiene for judges

1. [ ] **Do not** commit: large checkpoints, `.safetensors` blobs, `venv/`, or anything under `trash/` (that folder is gitignored).
2. [ ] **Do** commit: small JSON (`eval_summary`, `run_metadata`, `stage4_report` if you keep it), small PNGs, and updated `README` links.
3. [ ] In the README, set the **Colab** link, **Space** link, and **blog** link (`/blog.md` on the Space).
4. [ ] Update the “before / after” table with numbers from your **eval JSON** and training summary (see README section 6).

## E. If something breaks

- **Remote 503 on `/health`:** wait, refresh the Space, open the Space in a normal browser tab to wake the proxy; retry `preflight_space.py` with a higher `--retries`.
- **Connection errors in training:** use `prefer_http_mcp=True` for `https` (already the default in the client) and check Space logs for Python stack traces.
- **No reward/loss curves:** ensure `train_sinchan.py` ran with `logging_steps=1` and that `metrics.jsonl` exists under the run directory; re-run `plot_metrics.py` with the correct `--run-dir`.
- **Gradio 404 on Space:** confirm you deployed the current `server/app.py` and that the container rebuilt; the route is `/gradio`.

## F. Submission day

1. [ ] Re-run B (preflight) and C.7–C.8 on a **clean** Colab runtime (or document exact versions in `run_metadata.json`).
2. [ ] Tag the git commit you submit: `v1.0-hackathon` (optional).
3. [ ] Double-check the hackathon form fields: **Space URL**, **Colab URL**, **repo URL**, and **one short write-up** (this repo uses `blog.md` on the Space as that write-up; add a public video link only if the rules require it).

You are done when: **Space is up**, **Colab runs end-to-end without manual patches**, and **curves + JSON** are in the repo (or on Hub) with no multi‑gigabyte files in Git.
