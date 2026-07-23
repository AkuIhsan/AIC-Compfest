import json
from pathlib import Path
import pandas as pd
from f_engineer import build_features


BASE_DIR = Path(__file__).resolve().parent
INPUT_ = BASE_DIR / "process_log.csv"
SOP_ = BASE_DIR / "sop_standard.csv"
OUTPUT_ = BASE_DIR / "data" / "processed"


def main() -> None:
    OUTPUT_.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_)

    if "qc_result" not in df.columns:
        raise ValueError("Kolom qc_result ga ada")

    valid_label = {"pass", "fail"}
    labels = set(df["qc_result"].dropna().astype(str).str.lower().unique())

    if labels - valid_label:
        raise ValueError(f"NIlai qc_result aneh: {sorted(labels - valid_label)}")

    X = build_features(df, SOP_)
    y = df["qc_result"].astype(str).str.lower().eq("fail").astype(int)
    metadata_col = [
        column for column in ["batch_id", "timestamp"] if column in df.columns
    ]
    metadata = df[metadata_col].copy()

    X.to_csv(OUTPUT_ / "features.csv", index=False)
    y.rename("target").to_csv(OUTPUT_ / "target.csv", index=False)
    metadata.to_csv(OUTPUT_ / "metadata.csv", index=False)

    with open(
        OUTPUT_ / "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(list(X.columns), file, indent=2)

    print("Jumlah baris:", len(df))
    print("Jumlah fitur:", X.shape[1])
    print("Rasio fail:", round(float(y.mean()), 3))
    print("Output:", OUTPUT_.resolve())


if __name__ == "__main__":
    main()
