# Plot assets (submission)

PNG charts are **not** committed as large binaries from CI by default. Generate them locally or in Colab after a real run:

```bash
python training/plot_metrics.py \
  --run-dir training/artifacts/run1 \
  --eval-summary training/artifacts/eval_summary.json \
  --assets-dir assets
```

Expected outputs:

- `reward_curve_total.png` — from `metrics.jsonl` / `trainer_state.json` reward keys.
- `loss_curve.png` — from logged `loss` / `train/loss` keys.
- `baseline_comparison.png` — from `evaluate_scenarios.py` JSON (`random` vs `rule_based`).

The Colab notebook includes a cell that runs the same plotting command. Commit the resulting small PNGs for judges (typically well under 1 MB total).
