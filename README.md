# Nashik air-quality prediction: KNN imputation + CNN-LSTM

This project predicts the next hour's concentration of seven pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3) at four Nashik monitoring stations. Missing inputs are filled with KNN imputation, and each station gets its own CNN-LSTM. The latest calendar year of each station is held out as the test set, and every result is compared against a persistence baseline.

## Files

| Path | What it is |
|---|---|
| `Nashik_AQ_CNN_LSTM.ipynb` | Colab notebook (executed, outputs included) |
| `nashik_aq_pipeline.py` | The same pipeline as a script. The notebook is generated from it. |
| `scripts/build_report.py` | Builds the DOCX report from `outputs/`. Every number in it is read from the result files. |
| `scripts/verify_outputs.py` | Independent checks: no leakage, row alignment, metrics recomputed by hand |
| `scripts/make_notebook.py` | Regenerates the notebook from the script |
| `outputs/Nashik_AQ_CNN_LSTM_Report.docx` / `.pdf` | Academic report |
| `outputs/predictions/actual_vs_predicted.csv` | One row per station × hour × pollutant with an observed value |
| `outputs/metrics/station_pollutant_metrics.csv` | MSE / RMSE / MAE / R² per station and pollutant, CNN-LSTM vs persistence |
| `outputs/metrics/overall_pollutant_metrics.csv` | Pooled per-pollutant summary (unflagged series) |
| `outputs/metrics/model_selection_validation.csv` | Validation losses used to choose the target design |
| `outputs/figures/*.png` | Figures 1-8 |
| `outputs/tables/*.csv` | Data dictionary, inspection, cleaning log, splits, imputation counts |
| `outputs/models/` | Trained `.keras` models (both designs) and per-station scaler + imputer (`.joblib`) |
| `colab_run/` | Report and metrics from the Google Colab run (Python 3.13, TensorFlow 2.21) |

## Two runs, two reports

The notebook was run twice with the same code, data and seed, and each run has its own report.

| Run | Environment | Report | PM2.5 RMSE |
|---|---|---|---|
| Local | Python 3.12, TensorFlow 2.19 | `outputs/Nashik_AQ_CNN_LSTM_Report.docx` | 8.674 |
| Google Colab | Python 3.13, TensorFlow 2.21 | `colab_run/Nashik_AQ_CNN_LSTM_Report.docx` | 8.627 |

The CNN-LSTM numbers differ slightly (RMSE by at most 0.05) because TensorFlow versions differ numerically. Persistence numbers and all conclusions are identical. Quote one run consistently, not a mix.

## Run in Google Colab

1. Open `Nashik_AQ_CNN_LSTM.ipynb` in Colab (File → Upload notebook). A CPU runtime is enough.
2. Choose Runtime → Run all.
3. When the upload cell prompts, select the four station CSVs (or the Nashik zip) and also `scripts/build_report.py` if you want the report rebuilt:
   `MIDCAMBAD_NASHIK.CSV`, `combined_gangapur_road_nashik.csv`, `combined_pandav_nagari_nashik.csv`, `hirwadi_nashik.csv`.
   Stations are matched by file name (`midcambad`, `gangapur`, `pandav`, `hirwadi`), so keep those words in the names.
4. The last cell builds the report if `build_report.py` was uploaded, then downloads `nashik_outputs.zip`. Training takes a few minutes on a CPU runtime.

On a GPU runtime, the numbers can differ slightly from the reported ones because cuDNN LSTM kernels are not bit-identical to the CPU kernels.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the four CSVs in data/
.venv/bin/python nashik_aq_pipeline.py          # writes outputs/ and builds the DOCX
.venv/bin/python scripts/verify_outputs.py      # independent checks
soffice --headless --convert-to pdf --outdir outputs outputs/Nashik_AQ_CNN_LSTM_Report.docx
```

The reported results came from executing the notebook with Python 3.12, TensorFlow 2.19.0, pandas 2.2.2, NumPy 2.0.2 and scikit-learn 1.6.1 on CPU, with seed 42. Running the script with TensorFlow 2.19.1 and pandas 3.0 gave the same metrics.

## Design in one paragraph

Each station is cleaned using documented rules: the ≥999 PM analyser ceiling is set to missing, as is any value repeated for 12 or more hours (a stuck sensor). The data is then split by time. Test is the station's latest year: Jan-Aug 2026 (July missing) for three stations, and 2025 for Pandav Nagari. Validation is the six months before the test year, and training is everything earlier. The scaler and KNN imputer (k=5) are fitted on training rows only. Inputs are 24 hours of 7 pollutants plus 6 calendar encodings. The model is Conv1D×2 → MaxPool → LSTM(64) → Dropout → Dense → 7 outputs, trained with masked MSE so missing targets are never imputed into the loss. Two target designs are trained: predicting the level, and predicting the change from the last observation. The one with the lower validation loss is kept (change, at all four stations). Metrics are computed in original units on observed hours only. Persistence is evaluated on the same rows.
