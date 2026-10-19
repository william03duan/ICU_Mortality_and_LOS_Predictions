from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use the script's location, not PyCharm's working directory.  This keeps the
# code portable across teammates' computers as long as they share the project
# structure shown in the docstring.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "raw_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eda_results"

# One clinically interpretable summary per time-varying variable.
# Urine is a volume, so use its 48-hour total.  All other values below represent
# a patient's most abnormal value, except HR and RespRate, for which mean level
# is used as a simple EDA summary.
FEATURE_RULES = {
    "GCS": ("min", "gcs_min"),
    "Creatinine": ("max", "creatinine_max"),
    "Urine": ("sum", "urine_sum_48h"),
    "NIMAP": ("min", "nimap_min"),
    "BUN": ("max", "bun_max"),
    "HR": ("mean", "hr_mean"),
    "RespRate": ("mean", "resprate_mean"),
}

PREDICTORS = ["SAPS_I", "SOFA", *[new_name for _, new_name in FEATURE_RULES.values()]]
KEY_OUTCOMES = ["Length_of_stay", "In_hospital_death"]

DISPLAY_NAMES = {
    "SAPS_I": "SAPS-I",
    "SOFA": "SOFA",
    "gcs_min": "GCS (minimum)",
    "creatinine_max": "Creatinine (maximum, mg/dL)",
    "urine_sum_48h": "Urine (48-hour total, mL)",
    "nimap_min": "NIMAP (minimum, mmHg)",
    "bun_max": "BUN (maximum, mg/dL)",
    "hr_mean": "HR (mean, bpm)",
    "resprate_mean": "RespRate (mean, bpm)",
    "Length_of_stay": "Length of stay (days)",
    "In_hospital_death": "In-hospital death",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EDA tables and figures for in-hospital death and length of stay."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset folder containing outcomes.csv and icu_records/ (default: ./raw_dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for EDA tables and figures (default: ./eda_results).",
    )
    return parser.parse_args()


def patient_features(record_file: Path) -> dict[str, float | int]:
    """Turn one patient's irregular observations into seven EDA features."""
    record_id = int(record_file.stem)
    raw = pd.read_csv(record_file)

    # Dataset documentation states that -1 in a patient record is missing,
    # not a real physiological measurement.
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce").replace(-1, np.nan)
    raw["Parameter"] = raw["Parameter"].astype(str).str.strip()

    result: dict[str, float | int] = {"RecordID": record_id}
    for variable, (method, feature_name) in FEATURE_RULES.items():
        values = raw.loc[raw["Parameter"] == variable, "Value"].dropna()
        if values.empty:
            result[feature_name] = np.nan
        elif method == "min":
            result[feature_name] = values.min()
        elif method == "max":
            result[feature_name] = values.max()
        elif method == "sum":
            result[feature_name] = values.sum()
        else:
            result[feature_name] = values.mean()
    return result


def load_outcomes(data_root: Path) -> pd.DataFrame:
    outcomes_path = data_root / "outcomes.csv"
    if not outcomes_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {outcomes_path}. Put this script next to the 'raw dataset' folder, "
            "or provide --data-root with the correct folder."
        )
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes.rename(
        columns={"SAPS-I": "SAPS_I", "In-hospital_death": "In_hospital_death"}
    )
    required = {"RecordID", "SAPS_I", "SOFA", *KEY_OUTCOMES}
    if missing := required.difference(outcomes.columns):
        raise ValueError(f"outcomes.csv is missing: {sorted(missing)}")
    if outcomes["RecordID"].duplicated().any():
        raise ValueError("outcomes.csv has duplicated RecordID values.")
    return outcomes


def create_patient_level_table(data_root: Path, outcomes: pd.DataFrame) -> pd.DataFrame:
    records_dir = data_root / "icu_records"
    record_files = sorted(records_dir.glob("*.csv"))
    if not record_files:
        raise FileNotFoundError(f"No patient records found in {records_dir}")

    rows = []
    for number, file in enumerate(record_files, start=1):
        rows.append(patient_features(file))
        if number % 1000 == 0 or number == len(record_files):
            print(f"Processed {number:,}/{len(record_files):,} records")

    features = pd.DataFrame(rows)
    return outcomes.merge(features, on="RecordID", how="left", validate="one_to_one")


def create_eda_plots(data: pd.DataFrame, missingness: pd.DataFrame, output_dir: Path) -> None:
    """Creat visualization charts from exploratory data analysis."""

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    # 1. Missingness bar chart: restricted to variables used in this project.
    missing_for_plot = missingness[missingness["variable"].isin(PREDICTORS)].copy()
    missing_for_plot["label"] = missing_for_plot["variable"].map(DISPLAY_NAMES)
    missing_for_plot = missing_for_plot.sort_values("missing_percent")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing_for_plot["label"], missing_for_plot["missing_percent"], color="#4C78A8")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missingness in EDA predictors")
    ax.set_xlim(left=0)
    for y_position, value in enumerate(missing_for_plot["missing_percent"]):
        ax.text(value, y_position, f" {value:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "01_missingness.png", dpi=300)
    plt.close(fig)

    # 2. Histograms: one panel per predictor.  These show distribution shape
    # and help identify implausible values or extreme outliers to discuss.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for axis, column in zip(axes.flat, PREDICTORS):
        values = data[column].dropna()
        axis.hist(values, bins=30, color="#4C78A8", edgecolor="white")
        axis.set_title(DISPLAY_NAMES[column])
        axis.set_xlabel("Value")
        axis.set_ylabel("Patients")
    fig.suptitle("Distribution of EDA predictors", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "02_predictor_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Length-of-stay distribution.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data["Length_of_stay"].dropna(), bins=30, color="#72B7B2", edgecolor="white")
    ax.set_title("Length of stay distribution")
    ax.set_xlabel("Days")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(output_dir / "03_length_of_stay_distribution.png", dpi=300)
    plt.close(fig)

    # 4. Pie chart for the binary in-hospital-death outcome.  The labels show
    # both original coding (0/1) and its clinical meaning.
    death_counts = data["In_hospital_death"].value_counts().reindex([0, 1], fill_value=0)
    pie_labels = [
        f"0 = Survived\n{death_counts.loc[0]:,} patients",
        f"1 = Died in hospital\n{death_counts.loc[1]:,} patients",
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        death_counts,
        labels=pie_labels,
        autopct="%.1f%%",
        startangle=90,
        colors=["#72B7B2", "#E45756"],
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title("In-hospital death proportion")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "04_in_hospital_death_proportion.png", dpi=300)
    plt.close(fig)

    # 5. Compact heatmap: association of each predictor with the two key
    # prediction outcomes.  This is descriptive, not causal evidence.
    correlations = data[[*PREDICTORS, "Length_of_stay", "In_hospital_death"]].corr(method="spearman")
    heatmap_data = correlations.loc[PREDICTORS, ["Length_of_stay", "In_hospital_death"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(heatmap_data, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(heatmap_data.columns)), [DISPLAY_NAMES[column] for column in heatmap_data.columns])
    ax.set_yticks(range(len(heatmap_data.index)), [DISPLAY_NAMES[column] for column in heatmap_data.index])
    for row in range(heatmap_data.shape[0]):
        for column in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[row, column]
            text_colour = "white" if abs(value) > 0.5 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_colour)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Association with key prediction outcomes")
    fig.tight_layout()
    fig.savefig(output_dir / "05_spearman_heatmap.png", dpi=300)
    plt.close(fig)

    # 6. Input-variable correlation heatmap.  This helps identify highly
    # correlated predictors, which should be discussed before later modelling.
    input_correlations = data[PREDICTORS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(input_correlations, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    input_labels = [DISPLAY_NAMES[column] for column in PREDICTORS]
    ax.set_xticks(range(len(PREDICTORS)), input_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(PREDICTORS)), input_labels)
    for row in range(input_correlations.shape[0]):
        for column in range(input_correlations.shape[1]):
            value = input_correlations.iloc[row, column]
            text_colour = "white" if abs(value) > 0.5 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_colour)
    fig.colorbar(image, ax=ax, label="Spearman correlation coefficient")
    ax.set_title("Correlation between input variables")
    fig.tight_layout()
    fig.savefig(output_dir / "06_input_variable_correlation_heatmap.png", dpi=300)
    plt.close(fig)


def save_eda_outputs(data: pd.DataFrame, output_dir: Path) -> None:
    """Save the data visualization outputs"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Missingness is calculated for the chart and for the preprocessing flags,
    # but no separate missingness CSV is saved.
    missingness = pd.DataFrame(
        {
            "variable": data.columns,
            "n_missing": data.isna().sum().to_numpy(),
            "missing_percent": (100 * data.isna().mean()).to_numpy(),
        }
    ).sort_values("missing_percent", ascending=False)
    # Final patient-level dataset for later modelling.  It contains only the
    # nine selected input variables, their missing-value indicators, RecordID,
    # and the two key outcomes.  Outcomes are never imputed.
    preprocessed_data = data[["RecordID", *PREDICTORS, *KEY_OUTCOMES]].copy()
    for column in PREDICTORS:
        preprocessed_data[f"{column}_was_missing"] = preprocessed_data[column].isna().astype("int8")
        preprocessed_data[column] = preprocessed_data[column].fillna(preprocessed_data[column].median())
    preprocessed_data.to_csv(output_dir / "icu_eda_preprocessed_dataset.csv", index=False)

    create_eda_plots(data, missingness, output_dir)


def main() -> None:
    args = arguments()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    outcomes = load_outcomes(data_root)
    data = create_patient_level_table(data_root, outcomes)
    save_eda_outputs(data, output_dir)

    print(f"\nDone. EDA figures and the preprocessed dataset are saved to: {output_dir}")
    print("Key prediction outcomes: In_hospital_death and Length_of_stay (days).")


if __name__ == "__main__":
    main()
