# Plot assets (submission)

PNG charts in this folder are small and safe to commit for judges. **Current committed PNGs** (regenerate any time) come from a **real short local run** (CPU, few GRPO steps) plus a quick `evaluate_scenarios.py` pass, then `plot_metrics.py`. Training artifacts under `training/artifacts/` stay gitignored.

**Windows + TRL:** if import fails with `UnicodeDecodeError` on `.jinja` files, set `PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) for the training command.

**Regenerate (typical / Colab) run:**

```bash
# After training, from repo root
python training/plot_metrics.py \
  --run-dir training/artifacts/run1 \
  --eval-summary training/artifacts/eval_summary.json \
  --assets-dir assets
```

**Files:**

- `reward_curve_total.png` — from `metrics.jsonl` / `trainer_state.json` reward keys. Very short runs can look **flat**; the title notes that if so.
- `loss_curve.png` — from `loss` in logs, or (when loss is all zeros) **policy entropy** from the same rows so a short run still has a useful curve.
- `baseline_comparison.png` — from `evaluate_scenarios.py` JSON (`random` vs `rule_based`).

The Colab notebook can run the same plotting command after your full training. Keep total PNG size small (well under 1 MB combined).
