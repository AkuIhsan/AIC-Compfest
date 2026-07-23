from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_ = ROOT / "SECOM"
OUTPUT_ = ROOT / "data" / "secom" / "processed"
DATA_ = RAW_ / "secom.data"
LABEL_ = RAW_ / "secom_labels.data"
MISS_THRESHOLD = 0.50


def main() -> None:
    OUTPUT_.mkdir(parents=True, exist_ok=True)

    X_RAW = pd.read_csv(
        DATA_,
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )
    X_RAW.columns = [f"sensor_{index}" for index in range(X_RAW.shape[1])]
    labels = pd.read_csv(
        LABEL_,
        sep=r"\s+",
        header=None,
        names=["raw_label", "timestamp"],
    )

    if len(X_RAW) != len(labels):
        raise ValueError(
            f"jumlah baris sensor dan label beda: {len(X_RAW)} != {len(labels)}"
        )

    valid_raw = {-1, 1}
    found_raw = set(labels["raw_label"].unique())
    if not found_raw.issubset(valid_raw):
        raise ValueError(f"Label ga dikenal: {sorted(found_raw)}")

    # -1 = pass
    # 1 = fail
    y = (
        labels["raw_label"]
        .map(
            {
                -1: 0,
                1: 1,
            }
        )
        .astype(int)
    )

    timestamp = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )

    invalid_timestamp = int(timestamp.isna().sum())

    # laporan tiap fitur
    if invalid_timestamp:
        raise ValueError(f"ketemu {invalid_timestamp} timestamp ga valid")

    missing_count = X_RAW.isna().sum()
    missing_ratio = X_RAW.isna().mean()
    unique_non_null = X_RAW.nunique(dropna=True)
    feature_report = pd.DataFrame(
        {
            "feature": X_RAW.columns,
            "missing_count": missing_count.values,
            "missing_ratio": missing_ratio.values,
            "unique_non_null": unique_non_null.values,
        }
    )
    feature_report["drop_reason"] = ""
    mask_high_missing = feature_report["missing_ratio"] > MISS_THRESHOLD
    feature_report.loc[
        mask_high_missing,
        "drop_reason",
    ] = "missing_above_50_percent"
    mask_constant = ~mask_high_missing & (feature_report["unique_non_null"] <= 1)
    feature_report.loc[
        mask_constant,
        "drop_reason",
    ] = "constant"
    feature_report.loc[
        feature_report["drop_reason"] == "",
        "drop_reason",
    ] = "kept"
    columns_high_missing = feature_report.loc[
        mask_high_missing,
        "feature",
    ].tolist()
    columns_constant = feature_report.loc[
        mask_constant,
        "feature",
    ].tolist()
    # missing values sengaja ada.
    # imputasi harus setelah train-test split.
    X_prepared = X_RAW.drop(columns=columns_high_missing + columns_constant)
    prepared = X_prepared.copy()
    prepared["qc_result"] = y
    metadata = pd.DataFrame(
        {
            "row_id": range(len(X_RAW)),
            "timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    label_counts = y.value_counts().sort_index()
    report = {
        "status": "lolos",
        "target_mapping": {
            "-1": "pass/0",
            "1": "fail/1",
        },
        "raw": {
            "rows": int(X_RAW.shape[0]),
            "sensor_columns": int(X_RAW.shape[1]),
            "missing_cells": int(X_RAW.isna().sum().sum()),
            "dupe_rows": int(X_RAW.duplicated().sum()),
        },
        "cleaning": {
            "missing_threshold": MISS_THRESHOLD,
            "dropped_high_missing_columns": len(columns_high_missing),
            "dropped_constant_columns": len(columns_constant),
            "remaining_sensor_columns": int(X_prepared.shape[1]),
            "remaining_missing_cells": int(X_prepared.isna().sum().sum()),
            "remaining_columns_with_missing": int(X_prepared.isna().any().sum()),
        },
        "target": {
            "pass_count": int(label_counts.get(0, 0)),
            "fail_count": int(label_counts.get(1, 0)),
            "fail_ratio": float(y.mean()),
        },
        "metadata": {
            "invalid_timestamps": invalid_timestamp,
            "duplicate_timestamp": int(timestamp.duplicated().sum()),
        },
    }

    prepared.to_csv(
        OUTPUT_ / "secom_prepared.csv",
        index=False,
    )
    feature_report.to_csv(
        OUTPUT_ / "secom_feature_report.csv",
        index=False,
    )
    metadata.to_csv(
        OUTPUT_ / "secom_metadata.csv",
        index=False)
    
    with (OUTPUT_ / "secom_feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(list(X_prepared.columns), file, indent=2)
    with (OUTPUT_ / "secom_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Status                  : LOLOS")
    print("Baris                   :", len(prepared))
    print("Sensor mentah           :", X_RAW.shape[1])
    print("Buang >50% kosong       :", len(columns_high_missing))
    print("Buang sensor konstan    :", len(columns_constant))
    print("Sensor akhir            :", X_prepared.shape[1])
    print("Missing masih tersisa   :", X_prepared.isna().sum().sum())
    print("Pass                    :", int((y == 0).sum()))
    print("Fail                    :", int((y == 1).sum()))
    print("Rasio fail              :", round(float(y.mean()), 4))
    print("Output                  :", OUTPUT_.resolve())
    print()
    print(
        "Catatan: missing values sengaja gak diisi. "
        "Imputer harus di-fit di training set."
    )


if __name__ == "__main__":
    main()
