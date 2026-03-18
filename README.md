# 617-project — Energy Load Forecasting

Group project (4 members) to predict day-ahead electricity demand using an LSTM baseline vs. a Temporal Fusion Transformer (TFT), with uncertainty estimates and stress tests on extreme weather days.

## Quick start
1. **Create env & install**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. **Data layout**
   - `data/raw/` — downloads from load + weather sources (not tracked)
   - `data/processed/` — joined/cleaned features (not tracked)
3. **Data pull**
   - Use `notebooks/01_data_pull.ipynb` (to be added) or `scripts/data_pull.py` to download:
     - Load: chosen region (e.g., NYISO system load)
     - Weather: Meteostat hourly for nearest stations
   - Save a single parquet/CSV with aligned timestamps, holiday flags, and extreme-weather flags.
4. **Training**
   - Baseline LSTM: `src/models/lstm_baseline.py` (todo)
   - TFT: `src/models/tft.py` (todo)
   - Track runs with MLflow or W&B; metrics: MAE, RMSE, MAPE, pinball loss, coverage.
5. **Evaluation**
   - Time-based train/val/test (hold out most recent year).
   - Extra slice for extreme weather (e.g., top/bottom 5% temperature).
   - Plots/tables land in `reports/`.

## Proposed folders
```
data/
  raw/
  processed/
notebooks/
  01_data_pull.ipynb
  02_eda.ipynb
src/
  data/
  features/
  models/
  utils/
reports/
scripts/
```

## Suggested roles
- Data scout
- Baseline builder (LSTM + simple stats)
- TFT specialist
- Evaluator/storyteller (metrics, slices, report)

## Next actions
- Pick region + date range; fix timezone.
- Commit the data pull notebook/script.
- Define train/val/test splits and extreme-weather thresholds in a shared config.


## How to Run

- Activate env: `source .venv/bin/activate`
- Data (optional refresh): `python scripts/data_pull.py --force --refresh-weather`
- Train TFT (example): `python scripts/train_tft.py --max_epochs 15`
- Evaluate TFT: `python scripts/eval_tft.py --checkpoint reports/tft/lightning_logs/version_3/checkpoints/epoch=14-step=10170.ckpt --device cuda --batch_size 256 --num_workers 0`
- Plots: `python scripts/report_plots.py`
- Notebook: open `notebooks/notebook.ipynb` (Jupyter) for metrics/plots walkthrough.
- Runbook: see `617-project-group-3.md` for full details.
