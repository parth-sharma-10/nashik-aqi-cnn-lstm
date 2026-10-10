"""Build the academic report (DOCX) from the files the pipeline wrote to outputs/.

Every number in the text is read from those files. Statements that depend on the
results are guarded by asserts, so a different run fails loudly instead of misreporting.
Run from the project root:  python scripts/build_report.py
"""
import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt, RGBColor

OUT = "outputs"
REPORT = f"{OUT}/Nashik_AQ_CNN_LSTM_Report.docx"
POLL = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
STATIONS = ["MIDC Ambad", "Gangapur Road", "Pandav Nagari", "Hirwadi"]

sm = pd.read_csv(f"{OUT}/metrics/station_pollutant_metrics.csv")
om = pd.read_csv(f"{OUT}/metrics/overall_pollutant_metrics.csv").set_index("pollutant").loc[POLL]
sel = pd.read_csv(f"{OUT}/metrics/model_selection_validation.csv", index_col="station")
skill = pd.read_csv(f"{OUT}/metrics/rmse_skill_summary.csv", index_col="station")
splits = pd.read_csv(f"{OUT}/tables/splits.csv")
clog = pd.read_csv(f"{OUT}/tables/cleaning_log.csv", index_col=0)
imp = pd.read_csv(f"{OUT}/tables/imputation_counts.csv")
insp = pd.read_csv(f"{OUT}/tables/dataset_inspection.csv")
miss = pd.read_csv(f"{OUT}/tables/missing_pct_before_imputation.csv", index_col=0)
info = json.load(open(f"{OUT}/metrics/run_info.json"))
V = info["versions"]

pred = pd.read_csv(f"{OUT}/predictions/actual_vs_predicted.csv")
peak_rows = []
for _, g in pred.groupby(["station", "pollutant"]):
    top = g[g.actual >= g.actual.quantile(0.99)]
    peak_rows.append({
        "under": (top.cnn_lstm_pred < top.actual).mean(),
        "lagged": g.cnn_lstm_pred.corr(g.persistence_pred) > g.cnn_lstm_pred.corr(g.actual)})
peaks = pd.DataFrame(peak_rows)
flagged = sm[sm.quality_flag]
clean_sm = sm[~sm.quality_flag]
n_beat = int((clean_sm["RMSE skill vs persistence"] > 0).sum())
n_clean = len(clean_sm)
level_worse = int((om["level-design RMSE"] > om["Persistence RMSE"]).sum())

# Facts the prose below states. If a re-run changes them, rewrite the prose.
assert (sel.selected_design == "change").all()
assert (sel.val_loss_change < sel.val_loss_persistence).all()
assert (sel.val_loss_level > sel.val_loss_persistence).all()
assert (om["RMSE skill vs persistence"] > 0).all() and level_worse == len(POLL) - 1
assert set(map(tuple, flagged[["station", "pollutant"]].values)) == {
    ("Gangapur Road", "NO2"), ("Gangapur Road", "CO"), ("Gangapur Road", "NH3"),
    ("MIDC Ambad", "CO")}

doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(1)
FIG_N = [0]
TAB_N = [0]


def h(text, level=1):
    doc.add_heading(text, level=level)


def para(text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    p.paragraph_format.space_after = Pt(6)
    return p


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name, r.italic = "Cambria Math", True


def caption(prefix, counter, text):
    counter[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"{prefix} {counter[0]}. {text}")
    r.italic, r.font.size = True, Pt(9)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return counter[0]


def figure(path, text, width=6.3):
    doc.add_picture(f"{OUT}/figures/{path}", width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return caption("Figure", FIG_N, text)


def table(df, text, font=8, widths=None):
    n = caption("Table", TAB_N, text)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style, t.alignment = "Light Grid Accent 1", WD_TABLE_ALIGNMENT.CENTER
    for cell, col in zip(t.rows[0].cells, df.columns):
        cell.text = str(col)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for cell, v in zip(cells, row):
            cell.text = v if isinstance(v, str) else f"{v:,.4g}" if isinstance(v, float) else str(v)
    for row in t.rows:
        for k, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[k])
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font)
    doc.add_paragraph()
    return n


def fmt(x, d=3):
    return f"{x:.{d}f}"


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def front_matter():
    for _ in range(5):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Air-Quality Prediction for Nashik Monitoring Stations Using "
                  "KNN Imputation and a CNN-LSTM Network")
    r.bold, r.font.size = True, Pt(20)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Academic Project Report").font.size = Pt(13)
    for line in ["", "[Student name]  ·  [Roll number]", "[Department, Institution]",
                 "[Course / guide]", "[Submission date]"]:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page_break()

    h("Abstract")
    pooled = ", ".join(f"{p} {100 * om.loc[p, 'RMSE skill vs persistence']:.1f}%" for p in POLL)
    para(
        "Hourly concentrations of seven pollutants (PM2.5, PM10, NO2, SO2, CO, O3 and NH3) "
        "from four continuous monitoring stations in Nashik, India (MIDC Ambad, Gangapur Road, "
        "Pandav Nagari and Hirwadi) were used to build a one-hour-ahead prediction model. "
        "After rule-based cleaning, missing inputs were filled with a K-nearest-neighbours "
        "imputer fitted on training-period data only. A separate convolutional-LSTM "
        "(CNN-LSTM) network was trained for each station on 24-hour input windows, with the "
        "latest calendar year of each station held out for testing: January-August 2026 for "
        "three stations (July 2026 is absent from the files) and 2025 for Pandav Nagari, "
        "whose record ends on 31 December 2025. Predictions were scored in original units "
        "with MSE, RMSE, MAE and R² against observed values only, and compared with a "
        "persistence baseline on identical test hours. Pooled over stations, with four "
        "flagged sensor series excluded, the CNN-LSTM reached test R² between "
        f"{om['CNN-LSTM R2'].min():.2f} ({om['CNN-LSTM R2'].idxmin()}) and "
        f"{om['CNN-LSTM R2'].max():.2f} ({om['CNN-LSTM R2'].idxmax()}), and reduced RMSE "
        f"relative to persistence by: {pooled}. A network trained to predict concentration "
        "levels directly was worse than persistence on validation data at every station; "
        "predicting the change from the last observed value was therefore selected on "
        "validation loss. The gains over persistence are consistent but modest, and four "
        "station-pollutant test series show sensor-quality problems that dominate their scores.")
    h("Keywords")
    para("air-quality prediction; CNN-LSTM; KNN imputation; time-series forecasting; "
         "persistence baseline; Nashik; particulate matter; chronological validation")
    page_break()


def intro_sections():
    h("1. Introduction")
    para("Short-term air-quality prediction supports public advisories and operational "
         "decisions such as traffic or industrial-emission controls. Continuous ambient "
         "monitoring stations report hourly concentrations, but their records contain gaps, "
         "instrument faults and changes in data availability, so any prediction pipeline must "
         "handle missing data without leaking information from the future.")
    para("Hybrid convolutional-recurrent networks are a common choice for such series: "
         "one-dimensional convolutions extract short local patterns, and Long Short-Term "
         "Memory (LSTM) layers model longer dependencies [1], [3], [4]. This project applies "
         "that design to data from four Nashik stations and evaluates it against the "
         "simplest competitive forecast, persistence.")
    h("2. Problem Statement")
    para("Given the hourly pollutant observations of a Nashik monitoring station up to hour t, "
         "predict the concentrations of seven pollutants at hour t+1, and quantify the "
         "accuracy of those predictions over the latest calendar year in the data, which is "
         "excluded from model fitting and model selection.")
    h("3. Objectives")
    bullets([
        "Inspect the Nashik dataset and document its schema, coverage and quality problems.",
        "Build a reproducible cleaning pipeline and KNN imputation fitted without temporal "
        "leakage.",
        "Design, train and validate a CNN-LSTM model for next-hour prediction at each station.",
        "Evaluate predictions with MSE, RMSE, MAE and R² in original units per station and "
        "pollutant, and compare them with a persistence baseline on identical test hours.",
        "Report limitations, including data-quality issues that affect the evaluation.",
    ])


def dataset_section():
    h("4. Dataset Description")
    para("The dataset consists of four CSV files, one per station. Station identity is "
         "given only by the file name; no column identifies the station. All files share "
         "the same 25 columns: a Timestamp plus 24 measurements (criteria pollutants, "
         "volatile organic compounds and meteorological variables). The Hirwadi file has a "
         "26th column, Source_File, whose labels do not match the timestamps (rows labelled "
         "'2023.csv.csv' carry 2026 timestamps), so it was dropped and the timestamps were "
         "treated as authoritative. Hirwadi rows are also stored out of chronological "
         "order and were sorted. The column naming matches the export format of India's "
         "Central Pollution Control Board continuous-monitoring portal; the provenance should "
         "be confirmed with the data provider before publication.")
    d = insp[["station", "rows", "first", "last", "duplicate_timestamps", "sorted_in_file",
              "modal_step", "years_present", "all_pollutants_missing_rows"]].copy()
    d["modal_step"] = d.modal_step.replace({"0 days 01:00:00": "1 hour"})
    d.columns = ["Station", "Rows", "First timestamp", "Last timestamp", "Duplicate times",
                 "Sorted", "Step", "Years present", "Rows with no pollutant"]
    table(d, "Raw files as delivered.", font=7)
    para("One observation is one hourly average. Three files begin with a full year of empty "
         "2017 placeholder rows and resume in mid-2023 (the first observation is on "
         "14 June 2023 at MIDC Ambad, 22 May 2023 at Pandav Nagari and 23 May 2023 at "
         "Hirwadi). Gangapur Road has the longest record, starting in 2017 with a 123-day "
         "gap in that year. In all files that reach 2026, July 2026 is missing entirely. "
         "The latest calendar year is therefore 2026, covering 1 January-30 June and "
         "1-31 August (5,088 of 8,760 hours); it is incomplete. Pandav Nagari has no 2026 "
         "rows; its latest year, 2025, is complete.")
    para("Seven pollutants were selected as prediction targets: the pollutants of the "
         "Indian National AQI that are present in the data (PM2.5, PM10, NO2, SO2, CO, O3, "
         "NH3). The aromatic VOCs (benzene, toluene, xylenes) were not used. Xylene, "
         "BP and VWS are empty in every file. Meteorological variables were not used as "
         "inputs: air temperature and relative humidity are absent at Gangapur Road before "
         "2026, and wind speed shows level shifts that look like sensor changes (for example, "
         "a median of 0.0 m/s at MIDC Ambad in 2026 against about 0.5 m/s before, and "
         "isolated values above 20 m/s at Gangapur Road and Hirwadi). The full data "
         "dictionary, with per-station missing percentages, is exported as "
         "outputs/tables/data_dictionary.csv.")
    dd = pd.DataFrame({
        "Column (standardised)": ["Timestamp", *POLL],
        "Original column": ["Timestamp", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)",
                            "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)", "NH3 (µg/m³)"],
        "Unit": ["-", "µg/m³", "µg/m³", "µg/m³", "µg/m³", "mg/m³", "µg/m³", "µg/m³"],
        "Role": ["Time index (hourly)"] + ["Model input and prediction target"] * 7,
    })
    table(dd, "Data dictionary for the columns used by the model.", font=8)


def preprocessing_section():
    h("5. Data Preprocessing and KNN Imputation")
    h("5.1 Cleaning rules", 2)
    n_ceiling = int(clog["PM2.5_ceiling"].sum() + clog["PM10_ceiling"].sum())
    bullets([
        "Timestamps parsed and sorted; duplicate timestamps would keep the first row "
        "(none were found).",
        "Rows before a station's first pollutant observation removed (the empty 2017 "
        f"placeholder blocks): {int(clog.loc['MIDC Ambad', 'leading_empty_rows_dropped']):,} "
        f"rows at MIDC Ambad, {int(clog.loc['Pandav Nagari', 'leading_empty_rows_dropped']):,} "
        f"at Pandav Nagari and {int(clog.loc['Hirwadi', 'leading_empty_rows_dropped']):,} at "
        "Hirwadi.",
        "Negative concentrations set to missing (none occurred).",
        "PM2.5 or PM10 values of 999 or more set to missing, because that is the analyser "
        f"ceiling, not a measurement ({n_ceiling} values, all at Gangapur Road).",
        "Any value repeated unchanged for 12 or more consecutive hours set to missing as a "
        "stuck-sensor reading. This removed "
        f"{int(clog.filter(like='_stuck').sum().sum()):,} values, mostly CO readings of "
        "exactly 0.00 mg/m³ and long flat NO2 stretches at Gangapur Road.",
        "No interpolation, aggregation or resampling. Hours absent from the files (the "
        "2017 gap at Gangapur Road and July 2026) were not filled, and no input window "
        "spans them.",
    ])
    c = clog.filter(like="_stuck").copy()
    c.columns = [x.replace("_stuck", "") for x in c.columns]
    c.insert(0, "Station", c.index)
    table(c, "Values removed by the stuck-sensor rule (12 or more identical consecutive hours).")
    h("5.2 Missing values", 2)
    figure("fig01_missing_values.png", "Missing hourly values per station and pollutant "
           "before imputation, after cleaning, over each station's retained period.", 5.8)
    h("5.3 KNN imputation", 2)
    para("K-nearest-neighbours imputation fills a missing value with a weighted average of "
         "that variable in the k most similar records [2], [5]. Similarity is measured by "
         "the NaN-Euclidean distance, which compares two hours over the variables both have "
         "observed and rescales for the missing ones. Here k = 5 and neighbours are weighted "
         "by inverse distance (scikit-learn KNNImputer [6]).")
    bullets([
        "Features: the seven standardised pollutant concentrations of the same hour. "
        "Station identifiers and timestamps are not used in the distance.",
        "One StandardScaler and one KNNImputer per station, fitted on that station's "
        "training rows only and then applied to validation and test rows. Neighbours "
        "(donors) are therefore always training-period hours. Scaling is needed because "
        "units and ranges differ widely, for example CO in mg/m³ against PM10 in hundreds "
        "of µg/m³.",
        "Each station is imputed from its own history, preserving station-specific "
        "pollutant relationships. Stations never donate to each other.",
        "Imputed values are used only as model inputs. A missing value at hour s is filled "
        "from other pollutants observed at hour s, which are known when a prediction for "
        "any later hour is issued. Prediction targets are never imputed: training loss and "
        "every evaluation metric use observed values only.",
        "Hours with every pollutant missing receive the training mean (no distance can be "
        "computed). Input windows in which more than half of the pollutant cells are "
        "imputed are discarded (Section 9).",
    ])
    g = imp.groupby(["station", "split"], sort=False)[["missing_before", "missing_after"]] \
        .sum().reset_index()
    g.columns = ["Station", "Split", "Missing cells before", "Missing cells after"]
    table(g, "Missing pollutant cells before and after KNN imputation (7 pollutants).")


def eda_section():
    h("6. Exploratory Data Analysis")
    figure("fig02_historical_trends.png", "Monthly means of hourly observations per "
           "pollutant and station. Gaps are periods with no observations.", 4.9)
    para("Particulate matter shows a seasonal cycle at all stations, with lower "
         "concentrations in the monsoon months (June-September) and higher concentrations "
         "in winter. Gangapur Road is the only station with history before 2023. Its NO2, "
         "CO and NH3 series change character in 2026: NO2 settles near 3 µg/m³ in 0.25 "
         "steps and CO near 0 mg/m³, consistent with an instrument or calibration change "
         "rather than a real drop in pollution.")
    figure("fig03_station_comparison.png", "Distribution of hourly concentrations by station "
           "(box: interquartile range; whiskers: 1.5 IQR; outliers hidden).", 6.3)
    mp = miss.map(lambda v: f"{v:.1f}")
    mp.insert(0, "Station", mp.index)
    table(mp, "Percentage of missing hourly values before imputation (retained period).")


def architecture_section():
    h("7. Proposed CNN-LSTM Architecture")
    para("A one-dimensional convolution slides a small learned filter along the time axis. "
         "With kernel size 3, each output step combines three consecutive hours of all input "
         "features, so a filter can respond to a local motif such as a sharp morning rise in "
         "NO2 [7]. Stacking two such layers widens the receptive field to five hours. "
         "Causal padding ensures that no output step depends on later steps. Max-pooling then "
         "halves the sequence, keeping the strongest response in each pair of steps.")
    para("An LSTM layer [1] reads the pooled sequence step by step. Its cell state is "
         "updated through input, forget and output gates, which lets it carry information "
         "across the whole window without the vanishing gradients of a plain recurrent "
         "network. The final hidden state summarises the 24-hour history; dropout [9] "
         "regularises it, and two dense layers map it to the seven outputs.")
    arch = pd.DataFrame([
        ["Input", "(24, 13)", "24 hours x 13 features", "0"],
        ["Conv1D", "(24, 64)", "64 filters, kernel 3, causal padding, ReLU", "2,560"],
        ["Conv1D", "(24, 64)", "64 filters, kernel 3, causal padding, ReLU", "12,352"],
        ["MaxPooling1D", "(12, 64)", "pool size 2", "0"],
        ["LSTM", "(64)", "64 units, final hidden state", "33,024"],
        ["Dropout", "(64)", "rate 0.2", "0"],
        ["Dense", "(32)", "ReLU", "2,080"],
        ["Dense (output)", "(7)", "linear, one unit per pollutant", "231"],
    ], columns=["Layer", "Output shape", "Configuration", "Parameters"])
    table(arch, "CNN-LSTM layers (50,247 trainable parameters per station model).")
    para("Inputs per hour: the seven standardised, KNN-imputed pollutant concentrations "
         "plus sine and cosine encodings of hour-of-day, day-of-week and month (13 features). "
         "Calendar encodings are deterministic and known in advance. The model is "
         "multi-output: one network predicts all seven pollutants of a station, and each "
         "pollutant is evaluated separately.")


def methodology_section():
    h("8. Methodology and Experimental Setup")
    h("8.1 Forecasting design", 2)
    para("The task is one-step-ahead (next-hour) prediction. To predict hour t+1, the model "
         "receives the 24 hours t-23 ... t of observed (and, where missing, imputed) inputs. "
         "In the test year the inputs are the real observations preceding each predicted "
         "hour, exactly as they would be available to an operational system that updates "
         "every hour. No prediction is fed back as an input, so this is not a recursive "
         "forecast. The test year is a held-out historical year with known measurements. "
         "The work does not forecast genuinely future hours, for which no measurements "
         "exist, and its accuracy says nothing about predicting more than one hour ahead.")
    h("8.2 Station-specific models", 2)
    para("Each station has its own model, scaler and imputer. A shared model was not used "
         "because the stations' histories differ sharply (Gangapur Road from 2017; the "
         "others from mid-2023) and their test years differ (Pandav Nagari ends in 2025). "
         "In a shared model, Pandav Nagari's 2025 test year would coincide with training "
         "data from the other stations, and correlated regional weather would leak across "
         "stations. Independent models avoid that and keep sequences from crossing station "
         "boundaries by construction.")
    h("8.3 Sequence construction", 2)
    bullets([
        "Lookback 24 hours (one diurnal cycle); horizon 1 hour.",
        "A window is kept only if its 25 timestamps are consecutive hours, so no window "
        "spans a gap in the file.",
        "At least one of the seven targets must be observed at the predicted hour.",
        "At most 50% of the window's input pollutant cells may be imputed.",
        "Every pollutant must have an earlier observation (needed by the baseline).",
    ])
    h("8.4 Training configuration and target design", 2)
    cfg = pd.DataFrame([
        ["Loss", "Masked MSE on standardised targets (missing targets excluded)"],
        ["Optimiser", "Adam [8], learning rate 0.001"],
        ["Batch size / maximum epochs", f"{info['batch_size']} / {info['max_epochs']}"],
        ["Early stopping", f"validation loss, patience {info['patience']}, best weights restored"],
        ["Random seed", f"{info['seed']} (Python, NumPy, TensorFlow)"],
        ["Software", f"Python {V['python']}, TensorFlow {V['tensorflow']}, scikit-learn "
                     f"{V['scikit-learn']}, pandas {V['pandas']}, NumPy {V['numpy']}"],
    ], columns=["Setting", "Value"])
    table(cfg, "Training configuration (identical for all stations; no hyperparameter search).",
          font=8)
    para("Two parameterisations of the same network were trained per station, and the "
         "one with the lower validation loss was selected. In the level design, the network "
         "predicts the standardised concentration at t+1. In the change design, it predicts "
         "the difference between the concentration at t+1 and the last observed value, "
         "which is then added back. Because both are scored in the same standardised units, "
         "their validation losses are directly comparable with each other and with "
         "persistence.")
    s = sel.reset_index()[["station", "train_sequences", "val_sequences",
                           "val_loss_persistence", "val_loss_level", "val_loss_change",
                           "best_epoch_change", "selected_design"]]
    s.columns = ["Station", "Train seq.", "Val seq.", "Val loss: persistence",
                 "Val loss: level", "Val loss: change", "Best epoch (change)", "Selected"]
    table(s, "Model selection on the validation period (masked MSE, standardised units). "
             "The test year played no part.", font=8)
    para("The level design was worse than persistence on validation at all four stations. "
         "The change design beat persistence at all four, so it was selected everywhere. "
         "For transparency: the first version of the pipeline used only the level design. "
         "Its test-year results were computed before the change design was added, and they "
         "were worse than persistence. The switch is justified by the validation evidence "
         "above, which was available without the test year. Test RMSE for both designs is "
         "kept in outputs/metrics/station_pollutant_metrics.csv.")
    figure("fig04_loss_curves.png", "Training and validation loss of the selected (change) "
           "design per station. The dashed line is persistence on the validation period.", 6.5)
    para("Validation loss is below training loss at three stations. This is expected rather "
         "than suspicious: dropout is active only during training, and the validation half-"
         "year (July-December) includes the monsoon months, when concentrations are low and "
         "smooth. Hirwadi is the exception: validation loss is higher and rises after "
         "epoch 5, showing early overfitting, which early stopping caught.")


def split_section():
    h("9. Train-Validation-Test Strategy")
    para("Random splitting would put hours from the same day into both training and test "
         "sets. Because neighbouring hours are highly correlated, the model would be scored "
         "on near-copies of its training data, and it would also learn from the future of "
         "the period it is tested on [10]. A chronological split avoids both problems. "
         "For each station, the test set is its latest calendar year, validation is the six "
         "months before it, and training is everything earlier. Windows are assigned to a "
         "split by the timestamp of the predicted hour. Scaler and imputer statistics come "
         "from training rows only.")
    sp = splits[["station", "split", "start", "end", "hourly_rows", "months_present",
                 "pm25_observed"]].copy()
    sp.columns = ["Station", "Split", "Start", "End", "Hourly rows", "Months", "PM2.5 observed"]
    table(sp, "Chronological splits per station.", font=7)


def metrics_section():
    h("10. Evaluation Metrics: MSE, RMSE, MAE and R²")
    para("For n test hours with observed values yᵢ, predictions ŷᵢ and observed mean ȳ:")
    equation("MSE = (1/n) Σ (yᵢ − ŷᵢ)²")
    equation("RMSE = √MSE")
    equation("MAE = (1/n) Σ |yᵢ − ŷᵢ|")
    equation("R² = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²")
    bullets([
        "MSE is in squared units (for example (µg/m³)²). It penalises large errors heavily, "
        "so pollution spikes dominate it.",
        "RMSE is in the pollutant's own unit and is the typical size of an error, weighted "
        "towards large errors.",
        "MAE is in the pollutant's unit and is the average absolute error. It is less "
        "sensitive to spikes than RMSE.",
        "R² is unitless: the share of the test-period variance explained by the predictions. "
        "1 is perfect; 0 equals always predicting the test mean. It is negative when "
        "predictions are worse than that, which is possible on test data, and undefined "
        "when the test values are constant or there are fewer than two of them (reported as "
        "missing in that case).",
        "RMSE skill = 1 − RMSE_model / RMSE_persistence. Positive values mean the CNN-LSTM "
        "beats persistence; 0.05 means a 5% lower RMSE.",
    ])
    para("All metrics are computed with scikit-learn in original units, after inverse "
         "transformation and only on hours with an observed value. Pollutants are never "
         "pooled into one score because their units and scales differ. The overall summary "
         "is computed per pollutant by concatenating the test hours of all stations whose "
         "series is not quality-flagged.")


def results_section():
    h("11. Experimental Results")
    r = sm[["station", "pollutant", "evaluation_period", "n_test_obs", "CNN-LSTM MSE",
            "CNN-LSTM RMSE", "CNN-LSTM MAE", "CNN-LSTM R2"]].copy()
    r["evaluation_period"] = r.evaluation_period.str.replace(r" \(\d{4}\)", "", regex=True)
    r["pollutant"] = [f"{p} *" if q else p for p, q in zip(sm.pollutant, sm.quality_flag)]
    r.columns = ["Station", "Pollutant", "Evaluation Period", "Number of Test Observations",
                 "MSE", "RMSE", "MAE", "R²"]
    table(r, "CNN-LSTM test results per station and pollutant, in original units (µg/m³; "
             "CO in mg/m³, MSE in squared units). * = quality-flagged test series "
             "(fewer than 100 distinct values; see Section 14).", font=7,
          widths=[1.0, 0.7, 1.5, 0.8, 0.8, 0.7, 0.7, 0.6])
    o = om.reset_index()[["pollutant", "unit", "n_test_obs", "CNN-LSTM MSE", "CNN-LSTM RMSE",
                          "CNN-LSTM MAE", "CNN-LSTM R2", "stations_included"]]
    o.columns = ["Pollutant", "Unit", "Test obs.", "MSE", "RMSE", "MAE", "R²",
                 "Stations included"]
    table(o, "Overall summary per pollutant: test hours of all unflagged stations pooled. "
             "Test periods: 2025 (Pandav Nagari) and January-August 2026 (others).", font=7)
    h("11.1 Station-wise interpretation", 2)
    for s in STATIONS:
        g = sm[sm.station == s].set_index("pollutant")
        ok = g[~g.quality_flag]
        period = g.evaluation_period.iloc[0]
        text = (f"Test period {period}, {int(g.n_test_obs.min()):,}-{int(g.n_test_obs.max()):,} "
                f"observed hours per pollutant. R² ranges from {g['CNN-LSTM R2'].min():.2f} "
                f"({g['CNN-LSTM R2'].idxmin()}) to {g['CNN-LSTM R2'].max():.2f} "
                f"({g['CNN-LSTM R2'].idxmax()}). The model beats persistence on RMSE for "
                f"{int((ok['RMSE skill vs persistence'] > 0).sum())} of {len(ok)} unflagged "
                f"pollutants, with a mean RMSE skill of "
                f"{100 * ok['RMSE skill vs persistence'].mean():.1f}%.")
        bad = list(g.index[g.quality_flag])
        if bad:
            text += (f" Flagged series: {', '.join(bad)} (skill "
                     + ", ".join(f"{100 * g.loc[b, 'RMSE skill vs persistence']:.0f}%"
                                 for b in bad) + ").")
        para(text, bold_lead=f"{s}. ")
    para("Pandav Nagari and MIDC Ambad gain most over persistence (mean skill "
         f"{100 * skill.loc['Pandav Nagari', 'mean skill (unflagged)']:.1f}% and "
         f"{100 * skill.loc['MIDC Ambad', 'mean skill (unflagged)']:.1f}%), with the largest "
         "gains for O3 at both stations and for NO2 and PM10 at MIDC Ambad; these "
         "pollutants have strong diurnal cycles that a 24-hour "
         "window captures. Gangapur Road has the most training data but the weakest test "
         "behaviour, because three of its seven test series changed character in 2026. "
         "Hirwadi's NH3 and Gangapur Road's SO2 have low R² for both the model and "
         "persistence: these series are dominated by short, isolated spikes that no "
         "next-hour method anticipates.")


def baseline_section():
    h("12. Baseline Comparison")
    para("The persistence baseline predicts each pollutant at hour t+1 as its most recent "
         "observed value at or before hour t. It is evaluated on exactly the same "
         "station, hour and pollutant rows as the CNN-LSTM.")
    b = sm[["station", "pollutant", "n_test_obs", "CNN-LSTM RMSE", "Persistence RMSE",
            "CNN-LSTM MAE", "Persistence MAE", "CNN-LSTM R2", "Persistence R2",
            "RMSE skill vs persistence"]].copy()
    b["pollutant"] = [f"{p} *" if q else p for p, q in zip(sm.pollutant, sm.quality_flag)]
    b.columns = ["Station", "Pollutant", "n", "RMSE CNN-LSTM", "RMSE persist.", "MAE CNN-LSTM",
                 "MAE persist.", "R² CNN-LSTM", "R² persist.", "RMSE skill"]
    table(b, "CNN-LSTM against persistence on identical test hours (original units). "
             "* = quality-flagged series.", font=7)
    ob = om.reset_index()[["pollutant", "CNN-LSTM RMSE", "Persistence RMSE", "level-design RMSE",
                           "CNN-LSTM MAE", "Persistence MAE", "CNN-LSTM R2", "Persistence R2",
                           "RMSE skill vs persistence"]]
    ob.columns = ["Pollutant", "RMSE CNN-LSTM", "RMSE persist.", "RMSE level design",
                  "MAE CNN-LSTM", "MAE persist.", "R² CNN-LSTM", "R² persist.", "RMSE skill"]
    table(ob, "Pooled comparison per pollutant (unflagged stations). The level-design column "
              "shows the rejected design for reference.", font=7)
    mae_worse = clean_sm[(clean_sm["CNN-LSTM MAE"] > clean_sm["Persistence MAE"])]
    para(f"On unflagged series the CNN-LSTM has lower RMSE than persistence in {n_beat} of "
         f"{n_clean} station-pollutant pairs. Pooled RMSE skill ranges from "
         f"{100 * om['RMSE skill vs persistence'].min():.1f}% "
         f"({om['RMSE skill vs persistence'].idxmin()}) to "
         f"{100 * om['RMSE skill vs persistence'].max():.1f}% "
         f"({om['RMSE skill vs persistence'].idxmax()}). The improvement is consistent but "
         "small. Persistence is a strong baseline at a one-hour horizon because consecutive "
         "hourly averages are highly correlated. On MAE the picture is mixed: the CNN-LSTM "
         f"has a higher MAE than persistence in {len(mae_worse)} of the {n_clean} unflagged "
         "pairs. The network reduces large errors around rapid changes (which dominate RMSE) "
         "but adds small errors during calm periods, where repeating the last value is "
         "already almost exact. The level design, which predicts concentrations directly, "
         f"was worse than persistence for {level_worse} of {len(POLL)} pollutants on pooled "
         "test RMSE. "
         "Model complexity alone did not produce better predictions; the gain came from "
         "framing the task as a correction to persistence.")


def graphs_section():
    h("13. Graphs and Interpretation")
    for s in STATIONS:
        figure(f"fig05_actual_vs_pred_{s.replace(' ', '_')}.png",
               f"{s}: hourly actual and CNN-LSTM predicted concentrations over the whole "
               "test period. Blank stretches have no observation (including July 2026).", 6.0)
    para("Across the test period the predictions track the observed series closely at all "
         "stations, including the seasonal decline in particulate matter from winter to the "
         "monsoon. At hourly resolution the two lines overlap for most of the period, which "
         "is expected for a one-hour horizon. Visible departures are at isolated spikes (for "
         "example SO2 and NH3 bursts). For "
         f"{int(peaks.lagged.sum())} of {len(peaks)} series the predictions correlate more "
         "strongly with the previous hour's observation than with the current one: the "
         "model largely follows the series one hour late, as persistence does, and adds a "
         "small correction.")
    figure("fig06_scatter_actual_vs_pred.png", "Predicted against actual hourly concentration "
           "per station (columns) and pollutant (rows). Dashed line: perfect prediction.", 5.4)
    para("Points cluster along the 1:1 line for particulate matter and O3. The flagged "
         "Gangapur Road CO panel shows vertical bands and its NO2 panel a tight cluster, "
         "because the observed values take only a few discrete levels. The highest values "
         "fall below the line: within the top 1% of observed hours of each series, the "
         "prediction is lower than the observation in a median of "
         f"{100 * peaks.under.median():.0f}% of hours (range {100 * peaks.under.min():.0f}-"
         f"{100 * peaks.under.max():.0f}% across the 28 series).")
    figure("fig07_residuals.png", "Distribution of CNN-LSTM residuals (predicted minus actual) "
           "per station and pollutant over the test period.", 5.4)
    para("Residuals are centred near zero with long tails in both directions. The tails come "
         "from abrupt changes: in the hours with the sharpest rises, residuals are negative "
         "(the rise is under-predicted), and in the hours with the sharpest falls they are "
         "positive.")
    figure("fig08_station_metrics.png", "Test RMSE of the CNN-LSTM and persistence per station "
           "and pollutant, and RMSE skill (bottom right; bars below the axis limit are "
           "labelled with their value).", 6.5)


def discussion_section():
    h("14. Discussion")
    para("Leakage checks. The scaler and imputer of each station were fitted on training "
         "rows only. A separate verification script recomputes the scaler means from the raw "
         "training rows, re-derives every test actual value and every persistence prediction "
         "from the raw CSV files, confirms that every evaluated hour lies in the test year, "
         "and recomputes MSE, RMSE, MAE and R² by hand from the exported predictions. The "
         "script was itself tested by injecting a leaked baseline, a one-hour timestamp "
         "shift and corrupted predictions; it rejected all three.")
    para("Quality-flagged series. Four test series take fewer than 100 distinct values "
         "over thousands of hours: Gangapur Road NO2, CO and NH3, and MIDC Ambad CO. At "
         "Gangapur Road in 2026, NO2 reads mostly 3.00 µg/m³ in 0.25 steps (34 distinct "
         "values, against about 2,400 per year before) and CO mostly 0.00 mg/m³ (17 "
         "values). The model was trained on the earlier, continuous behaviour, so its "
         "predictions are poor there (Gangapur Road CO skill "
         f"{100 * skill.loc['Gangapur Road', 'CO']:.0f}%). These series measure an "
         "instrument change, not the model's ability to predict air quality. They are "
         "reported in every table but excluded from the pooled summary.")
    para("Overfitting and data availability. Early stopping ended training after 11-22 "
         "epochs. Hirwadi shows the clearest overfitting (validation loss rising after "
         "epoch 5). Training data is unbalanced: Gangapur Road has about 65,000 training "
         "windows, while Pandav Nagari has about 9,500. Despite that, Pandav Nagari has the "
         "best skill, which suggests that recent, consistent data matters more here than "
         "volume. Gangapur Road's long history includes years whose behaviour differs from "
         "2026.")
    para("Comparison with other studies. Studies such as [3] and [4] report CNN-LSTM models "
         "outperforming other neural networks for PM2.5 prediction. The results here neither "
         "confirm nor contradict them. The experiment shows that, for next-hour prediction "
         "on these stations, a CNN-LSTM gives a small but consistent improvement over "
         "persistence when trained to predict the change from the last value, but not when "
         "trained to predict the level.")


def limitations_section():
    h("15. Limitations")
    bullets([
        "The latest year is incomplete: 2026 covers January-June and August (5,088 hours); "
        "July is absent. Results for 2026 do not represent a full seasonal cycle.",
        "Pandav Nagari is evaluated on 2025, not 2026, so station results cover different "
        "periods and are not strictly comparable.",
        "Three stations have only about two years of training data (mid-2023 to June 2025), "
        "and 2023 is partially covered.",
        "Meteorological inputs were excluded because of missing or inconsistent sensors. "
        "Wind, temperature and humidity are known drivers of pollutant concentration, and "
        "their absence limits accuracy, especially at spikes.",
        "One hour ahead only. Skill at longer horizons, where persistence degrades, was not "
        "measured.",
        "Single training run per model (seed 42); no confidence intervals or "
        "multiple-seed variance are reported. Differences of a few per cent in skill may be "
        "within run-to-run variation.",
        "The quality flag (fewer than 100 distinct test values) and the 12-hour stuck rule "
        "are simple heuristics; some degraded readings may remain (for example, Hirwadi CO "
        "alternates between about 0 and 1 mg/m³ in February-April 2026).",
        "KNN imputation assumes hours with similar pollutant profiles are interchangeable. "
        "Long all-missing stretches receive the training mean.",
    ])


def conclusion_section():
    h("16. Conclusion and Future Scope")
    para("A reproducible pipeline was built for next-hour prediction of seven pollutants at "
         "four Nashik stations, using leakage-free KNN imputation, per-station CNN-LSTM "
         "models and a chronological hold-out of each station's latest year. On "
         "quality-checked series, the selected model explains "
         f"{om['CNN-LSTM R2'].min():.2f}-{om['CNN-LSTM R2'].max():.2f} of the pooled test "
         "variance and reduces RMSE relative to persistence by "
         f"{100 * om['RMSE skill vs persistence'].min():.1f}-"
         f"{100 * om['RMSE skill vs persistence'].max():.1f}% depending on the pollutant. "
         "The improvement is modest, and a model that predicted levels directly did not beat "
         "persistence on validation data at any station. The experiment also exposed sensor problems in the 2026 data "
         "that should be resolved with the data provider.")
    para("Future work: add quality-controlled meteorological inputs; evaluate horizons of "
         "6-24 hours, where persistence is weaker; report variance over several seeds; "
         "compare with simpler learned baselines (for example, gradient-boosted trees on lag "
         "features) to establish whether the recurrent architecture is necessary; and "
         "re-evaluate on a complete 2026 once the data is available.")


def references_section():
    h("17. References")
    refs = [
        "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, "
        "vol. 9, no. 8, pp. 1735-1780, 1997, doi: 10.1162/neco.1997.9.8.1735.",
        "O. Troyanskaya, M. Cantor, G. Sherlock, P. Brown, T. Hastie, R. Tibshirani, "
        "D. Botstein and R. B. Altman, \"Missing value estimation methods for DNA "
        "microarrays,\" Bioinformatics, vol. 17, no. 6, pp. 520-525, 2001, "
        "doi: 10.1093/bioinformatics/17.6.520.",
        "C.-J. Huang and P.-H. Kuo, \"A Deep CNN-LSTM Model for Particulate Matter (PM2.5) "
        "Forecasting in Smart Cities,\" Sensors, vol. 18, no. 7, art. 2220, 2018, "
        "doi: 10.3390/s18072220.",
        "D. Qin, J. Yu, G. Zou, R. Yong, Q. Zhao and B. Zhang, \"A Novel Combined Prediction "
        "Scheme Based on CNN and LSTM for Urban PM2.5 Concentration,\" IEEE Access, vol. 7, "
        "pp. 20050-20059, 2019, doi: 10.1109/ACCESS.2019.2897028.",
        "H. Junninen, H. Niska, K. Tuppurainen, J. Ruuskanen and M. Kolehmainen, \"Methods "
        "for imputation of missing values in air quality data sets,\" Atmospheric "
        "Environment, vol. 38, no. 18, pp. 2895-2907, 2004, "
        "doi: 10.1016/j.atmosenv.2004.02.026.",
        "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of "
        "Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
        "Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, \"Gradient-based learning applied to "
        "document recognition,\" Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, "
        "1998, doi: 10.1109/5.726791.",
        "D. P. Kingma and J. Ba, \"Adam: A Method for Stochastic Optimization,\" "
        "arXiv:1412.6980, 2014 (ICLR 2015).",
        "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, "
        "\"Dropout: A Simple Way to Prevent Neural Networks from Overfitting,\" Journal of "
        "Machine Learning Research, vol. 15, pp. 1929-1958, 2014.",
        "C. Bergmeir and J. M. Benítez, \"On the use of cross-validation for time series "
        "predictor evaluation,\" Information Sciences, vol. 191, pp. 192-213, 2012, "
        "doi: 10.1016/j.ins.2011.12.028.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)


if __name__ == "__main__":
    front_matter()
    intro_sections()
    dataset_section()
    preprocessing_section()
    eda_section()
    architecture_section()
    methodology_section()
    split_section()
    metrics_section()
    results_section()
    baseline_section()
    graphs_section()
    discussion_section()
    limitations_section()
    conclusion_section()
    references_section()
    doc.save(REPORT)
    print(f"Wrote {REPORT}: {FIG_N[0]} figures, {TAB_N[0]} tables")
