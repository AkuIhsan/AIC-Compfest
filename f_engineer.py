from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "curing_time_actual",
    "temp_actual",
    "pressure_actual",
    "manual_override",
    "visual_check_done",
    "machine_id",
    "shift",
    "material_batch",
    "operator_id",
}
REQUIRED_SOP_PARAMETERS = {
    "curing_time",
    "temp_min",
    "temp_max",
}


def load_sop(sop_path: str | Path) -> pd.Series:

    if not Path(sop_path).exists():
        raise FileNotFoundError(f"file sop gada: {Path(sop_path).resolve()}")

    sop_df = pd.read_csv(sop_path)

    required_columns = {"parameter", "nilai"}
    missing_columns = required_columns - set(sop_df.columns)

    if missing_columns:
        raise ValueError(f"kolom sop kurang: {sorted(missing_columns)}")
    sop = sop_df.set_index("parameter")["nilai"]

    missing_parameter = REQUIRED_SOP_PARAMETERS - set(sop.index)
    if missing_parameter:
        raise ValueError(f"parameter sop kurang: {sorted(missing_parameter)}")
    return pd.to_numeric(sop, errors="raise")


def build_features(
    df: pd.DataFrame,
    sop_path: str | Path = "sop_standard.csv",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    ngubah proses log mentah jadi fitur model.
    feature_columns:
        Schema kolom dari data training. kalo diberikan, hasil akan
        disesuaikan agar kolom serta urutannya sama persis.
    """
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
    for column in ["manual_override", "visual_check_done"]:
        invalid_values = set(data[column].dropna().unique()) - {0, 1}

        if invalid_values:
            raise ValueError(
                f"{column} harus bernilai 0 atau 1, "
                f"tetapi ditemukan: {sorted(invalid_values)}"
            )
    categorical_columns = [
        "machine_id",
        "shift",
        "material_batch",
        "operator_id",
    ]
    for column in categorical_columns:
        out = pd.DataFrame(index=data.index)
        out["curing_deviation"] = sop["curing_time"] - data["curing_time_actual"]
        out["temp_deviation"] = (sop["temp_min"] - data["temp_actual"]).clip(
            lower=0
        ) + (data["temp_actual"] - sop["temp_max"]).clip(lower=0)
        out["pressure"] = data["pressure_actual"]
        out["manual_override"] = data["manual_override"].astype(int)
        out["v_check_done"] = data["visual_check_done"].astype(int)

    categorical_features = pd.get_dummies(
        data[categorical_columns],
        prefix=["mesin", "shift", "material", "op"],
        dtype=int,
    )
    X = pd.concat([out, categorical_features], axis=1)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X
