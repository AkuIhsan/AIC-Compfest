from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESS_LOG_ = BASE_DIR / "process_log.csv"
SOP_ = BASE_DIR / "sop_standard.csv"
REPORT_ = BASE_DIR / "data" / "processed" / "data_quality_report.json"

REQUIRED_COLUMNS = {
    "batch_id",
    "timestamp",
    "machine_id",
    "shift",
    "operator_id",
    "material_batch",
    "curing_time_actual",
    "temp_actual",
    "pressure_actual",
    "manual_override",
    "visual_check_done",
    "qc_result",
}

NUMERIC_COLUMNS = [
    "curing_time_actual",
    "temp_actual",
    "pressure_actual",
]

BINARY_COLUMNS = [
    "manual_override",
    "visual_check_done",
]

CATEGORICAL_COLUMNS = [
    "machine_id",
    "shift",
    "operator_id",
    "material_batch",
]


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    statistic: dict[str, object] = {}

    if not PROCESS_LOG_.exists():
        raise FileNotFoundError(f"process_log.csv gada: {PROCESS_LOG_}")

    if not SOP_.exists():
        raise FileNotFoundError(f"sop_standard.csv gada {SOP_}")
    df = pd.read_csv(PROCESS_LOG_)
    statistic["jumlah_baris"] = int(len(df))
    statistic["jumlah_kolom"] = int(df.shape[1])
    missing_column = sorted(REQUIRED_COLUMNS - set(df.columns))
    extra_column = sorted(set(df.columns) - REQUIRED_COLUMNS)

    if missing_column:
        errors.append(f"kolom wajib gada: {missing_column}")
    if extra_column:
        warnings.append(f"ada kolom tambahan: {extra_column}")
    if not missing_column:
        dupe_row = int(df.duplicated().sum())
        dupe_batch_ids = int(df["batch_id"].duplicated().sum())

        statistic["baris_dupe"] = dupe_row
        statistic["batch_id_dupe"] = dupe_batch_ids

        if dupe_row:
            warnings.append(f"ada {dupe_row} baris dupe penuh")
        if dupe_batch_ids:
            warnings.append(f"ada {dupe_batch_ids} batch_id dupe")

        parsed_timestamp = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )
        invalid_timestamp = int(parsed_timestamp.isna().sum())
        statistic["timestamp_invalid"] = invalid_timestamp

        if invalid_timestamp:
            errors.append(f"ada {invalid_timestamp} timestamp ga valid")

        for column in NUMERIC_COLUMNS:
            converted = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int(converted.isna().sum())

            statistic[f"{column}_invalid"] = invalid_count

            if invalid_count:
                errors.append(f"{column}: {invalid_count} nilai kosong/bukan nomor")

            negative_count = int((converted < 0).sum())

            if negative_count:
                errors.append(f"{column}: {negative_count} nilai negatif")

        for column in BINARY_COLUMNS:
            converted = pd.to_numeric(df[column], errors="coerce")
            invalid_value = sorted(
                {
                    str(value)
                    for value in converted.dropna().unique()
                    if value not in {0, 1}
                }
            )

            if converted.isna().any():
                errors.append(f"{column} ada nilai kosong/bukan nomor")
            if invalid_value:
                errors.append(f"{column} harus 0 atau 1, ketemu: {invalid_value}")

        normalized_label = df["qc_result"].astype("string").str.strip().str.lower()
        invalid_labels = sorted(
            set(normalized_label.dropna().unique()) - {"pass", "fail"}
        )
        statistic["qc_result"] = {
            str(key): int(value)
            for key, value in normalized_label.value_counts(dropna=False).items()
        }

        if normalized_label.isna().any():
            errors.append("qc_result ada nilai kosong")
        if invalid_labels:
            errors.append(f"qc_result ga valid: {invalid_labels}")

        statistic["kategori"] = {}

        for column in CATEGORICAL_COLUMNS:
            empty_count = int(
                df[column].isna().sum()
                + df[column].astype("string").str.strip().eq("").sum()
            )

            unique_values = sorted(
                df[column].dropna().astype(str).str.strip().unique().tolist()
            )
            statistic["kategori"][column] = unique_values

            if empty_count:
                errors.append(f"{column}: {empty_count} nilai kosong")

    sop = pd.read_csv(SOP_)
    required_sop_col = {"parameter", "nilai"}
    mising_sop_col = sorted(required_sop_col - set(sop.columns))

    if mising_sop_col:
        errors.append(f"kolom sop ga lengkap: {mising_sop_col}")
    else:
        sop["parameter"] = sop["parameter"].astype(str).str.strip()
        sop["nilai"] = pd.to_numeric(sop["nilai"], errors="coerce")

        expected_parameter = {
            "curing_time",
            "temp_min",
            "temp_max",
        }
        missing_parameter = sorted(expected_parameter - set(sop["parameter"]))

        if missing_parameter:
            errors.append(f"parameter sop ga lengkap {missing_parameter}")

        dupe_parameter = sop.loc[
            sop["parameter"].duplicated(),
            "parameter",
        ].tolist()

        if dupe_parameter:
            errors.append(f"parameter sop dupe: {dupe_parameter}")
        if sop["nilai"].isna().any():
            errors.append("sop ada nilai kosong/bukan nomor")
        if not missing_parameter and not sop["nilai"].isna().any():
            sop_value = sop.set_index("parameter")["nilai"]

            statistic["sop"] = {
                key: float(sop_value[key]) for key in expected_parameter
            }

            if sop_value["temp_min"] >= sop_value["temp_max"]:
                errors.append("temp_min harus lebih kecil dari temp_max")

    report = {
        "status": "gagal" if errors else "lolos",
        "errors": errors,
        "warnings": warnings,
        "statistics": statistic,
    }

    REPORT_.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Status: {report['status'].upper()}")
    print(f"Error: {len(errors)}")
    print(f"Warning: {len(warnings)}")
    print(f"Report: {REPORT_}")

    for error in errors:
        print(f"[ERROR] {error}")

    for warning in warnings:
        print(f"[WARNING] {warning}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
