from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from f_engineer import REQUIRED_COLUMNS, load_sop


def check_sop(
    df: pd.DataFrame,
    sop_path: str | Path = "sop_standard.csv",
) -> pd.DataFrame:
    if REQUIRED_COLUMNS - set(df.columns):
        raise ValueError(
            f"Kolom process log kurang: {sorted(REQUIRED_COLUMNS - set(df.columns))}"
        )

    data = df.copy()
    sop = load_sop(sop_path)

    numeric_columns = [
        "curing_time_actual",
        "temp_actual",
        "pressure_actual",
        "manual_override",
        "visual_check_done",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    curing_min = sop["curing_time"] - sop["curing_tolerance"]
    curing_max = sop["curing_time"] + sop["curing_tolerance"]
    result = pd.DataFrame(index=data.index)
    result["curing_status"] = "within_sop"
    result.loc[
        data["curing_time_actual"] < curing_min,
        "curing_status",
    ] = "below_sop"
    result.loc[
        data["curing_time_actual"] > curing_max,
        "curing_status",
    ] = "above_sop"
    result["curing_deviation"] = 0.0
    result.loc[
        data["curing_time_actual"] < curing_min,
        "curing_deviation",
    ] = data["curing_time_actual"] - curing_min
    result.loc[
        data["curing_time_actual"] > curing_max,
        "curing_deviation",
    ] = data["curing_time_actual"] - curing_max
    result["temp_status"] = "within_sop"
    result.loc[
        data["temp_actual"] < sop["temp_min"],
        "temp_status",
    ] = "below_sop"
    result.loc[data["temp_actual"] > sop["temp_max"], "temp_status"] = "above_sop"
    result["temp_deviation"] = 0.0
    result.loc[data["temp_actual"] < sop["temp_min"], "temp_deviation"] = (
        data["temp_actual"] - sop["temp_min"]
    )
    result.loc[data["temp_actual"] > sop["temp_max"], "temp_deviation"] = (
        data["temp_actual"] - sop["temp_max"]
    )
    result["pressure_status"] = "within_sop"
    result.loc[
        data["pressure_actual"] < sop["pressure_min"],
        "pressure_status",
    ] = "below_sop"
    result.loc[
        data["pressure_actual"] > sop["pressure_max"],
        "pressure_status",
    ] = "above_sop"
    result["pressure_deviation"] = 0.0
    result.loc[
        data["pressure_actual"] < sop["pressure_min"],
        "pressure_deviation",
    ] = data["pressure_actual"] - sop["pressure_min"]
    result.loc[
        data["pressure_actual"] > sop["pressure_max"],
        "pressure_deviation",
    ] = data["pressure_actual"] - sop["pressure_max"]
    result["manual_override_violation"] = (
        data["manual_override"] > sop["manual_override_allowed"]
    ).astype(int)
    result["visual_check_violation"] = (
        data["visual_check_done"] < sop["visual_check_required"]
    ).astype(int)
    result["deviation_count"] = (
        result["curing_status"].ne("within_sop").astype(int)
        + result["temp_status"].ne("within_sop").astype(int)
        + result["pressure_status"].ne("within_sop").astype(int)
        + result["manual_override_violation"]
        + result["visual_check_violation"]
    )
    result["has_sop_deviation"] = (result["deviation_count"] > 0).astype(int)
    return result


def check__singele_record(
    record: dict[str, Any],
    sop_path: str | Path = "sop_standard.csv",
) -> dict[str, Any]:
    result = check_sop(
        pd.DataFrame([record]),
        sop_path=sop_path,
    )

    return result.iloc[0].to_dict()
