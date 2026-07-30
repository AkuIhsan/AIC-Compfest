import json
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
import shap
from sop_checker import check__singele_record
from f_engineer import build_features, load_sop

ROOT = Path(__file__).resolve().parent
try:
    PIPELINE = joblib.load(ROOT / "aic_qc_pipeline.pkl")
    with open(ROOT / "aic_qc_config.json", "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    with open(ROOT / "feature_columns.json", "r", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)

    XGB_MODEL = PIPELINE.named_steps["clf"]
    EXPLAINER = shap.TreeExplainer(XGB_MODEL)

    # Semua step SEBELUM classifier (None kalau pipeline cuma berisi clf)
    PRE_STEPS = PIPELINE[:-1] if len(PIPELINE.steps) > 1 else None
    SOP = load_sop(ROOT / "sop_standard.csv")

except FileNotFoundError as e:
    raise RuntimeError(f"Missing required artifact file: {e.filename}")


def _make_label(feature_name: str) -> str:
    """Utility function to generate human-readable labels from feature names."""
    if feature_name == "curing_shortfall":
        return "Curing time below SOP minimum"
    if feature_name == "manual_override":
        return "Manual override used"
    if feature_name.startswith("mesin_"):
        return f"Machine {feature_name.split('_')[1]}"
    if feature_name.startswith("shift_"):
        return f"{'Night' if 'malam' in feature_name else 'Morning' if 'pagi' in feature_name else 'Afternoon'} shift"
    if feature_name.startswith("material_"):
        return f"Material batch {feature_name.split('_')[1]}"

    # Default formatting
    return feature_name.replace("_", " ").title()


def _build_sop_check(
    record: Dict[str, Any], sop_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Structures the SOP check output to strictly match the API Contract."""
    deviations = []

    if sop_result["curing_status"] != "within_sop":
        deviations.append(
            {
                "parameter": "curing_time",
                "status": sop_result["curing_status"],
                "actual_value": record["curing_time_actual"],
                "sop_min": SOP["curing_time"] - SOP["curing_tolerance"],
                "sop_max": SOP["curing_time"] + SOP["curing_tolerance"],
                "deviation": sop_result["curing_deviation"],
                "unit": "minute",
            }
        )
    if sop_result["temp_status"] != "within_sop":
        deviations.append(
            {
                "parameter": "temperature",
                "status": sop_result["temp_status"],
                "actual_value": record["temp_actual"],
                "sop_min": SOP["temp_min"],
                "sop_max": SOP["temp_max"],
                "deviation": sop_result["temp_deviation"],
                "unit": "celsius",
            }
        )
    if sop_result["pressure_status"] != "within_sop":
        deviations.append(
            {
                "parameter": "pressure",
                "status": sop_result["pressure_status"],
                "actual_value": record["pressure_actual"],
                "sop_min": SOP["pressure_min"],
                "sop_max": SOP["pressure_max"],
                "deviation": sop_result["pressure_deviation"],
                "unit": "kPa",
            }
        )
    if sop_result["manual_override_violation"]:
        deviations.append(
            {
                "parameter": "manual_override",
                "status": "violation",
                "actual_value": bool(record["manual_override"]),
                "expected_value": False,
            }
        )
    if sop_result["visual_check_violation"]:
        deviations.append(
            {
                "parameter": "visual_check",
                "status": "skipped",
                "actual_value": bool(record["visual_check_done"]),
                "expected_value": True,
            }
        )

    return {
        "has_deviation": bool(sop_result["has_sop_deviation"]),
        "deviation_count": int(sop_result["deviation_count"]),
        "deviations": deviations,
    }


def predict(record: dict) -> dict:
    """
    Encapsulates the entire inference flow from raw record to API contract response.
    """
    # 1. Parse raw record & SOP check
    sop_result = check__singele_record(record)
    sop_check_response = _build_sop_check(record, sop_result)

    # 2. Feature Engineering
    # build_features expects a DataFrame and handles reindexing if feature_columns is provided
    df_raw = pd.DataFrame([record])
    X_eng = build_features(df_raw, feature_columns=FEATURE_COLUMNS)

    # Strict Reindexing: Ensure exactly 28 features in correct order, fill missing with 0
    X_eng = X_eng.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # 3. Risk Prediction
    risk_score = float(PIPELINE.predict_proba(X_eng)[0, 1])

    threshold_review = CONFIG["threshold_config"]["threshold_review"]
    threshold_alarm = CONFIG["threshold_config"]["threshold_alarm"]

    if risk_score < threshold_review:
        zone = "monitor"
    elif threshold_review <= risk_score < threshold_alarm:
        zone = "review"
    else:
        zone = "alarm"

    sop_risk_contributors = []
    context_contributors = []

    # 4. SHAP Calculation
    if PRE_STEPS is not None:
        X_for_shap = PRE_STEPS.transform(X_eng)
    else:
        X_for_shap = X_eng.values

    assert X_for_shap.shape[1] == len(FEATURE_COLUMNS), (
        "Pipeline changes number of columns -> SHAP mapping to feature name is invalid"
    )

    shap_values = EXPLAINER.shap_values(X_for_shap)[0]
    feature_values = X_eng.iloc[0].values  # nilai ASLI, untuk cek fitur aktif

    # 5. Categorization & Filtering
    for i in range(len(shap_values)):
        sv = float(shap_values[i])
        fv = float(feature_values[i])
        feat_name = X_eng.columns[i]

        # Filter: shap_value > 0, actual_value > 0, and not starting with "op_"
        if sv > 0 and fv > 0 and not feat_name.startswith("op_"):
            contributor = {
                "id": feat_name,
                "label": _make_label(feat_name),
                "contribution": sv,
            }
            if (
                feat_name.startswith("shift_")
                or feat_name.startswith("mesin_")
                or feat_name.startswith("material_")
            ):
                context_contributors.append(contributor)
            else:
                sop_risk_contributors.append(contributor)

    # Sort independently by absolute SHAP value (descending)
    sop_risk_contributors.sort(key=lambda x: x["contribution"], reverse=True)
    context_contributors.sort(key=lambda x: x["contribution"], reverse=True)

    # 6. Explanation Status & Override
    if not sop_risk_contributors and not context_contributors:
        explanation_status = "no_positive_driver"
    else:
        explanation_status = "success"

    decision = zone
    # CRITICAL OVERRIDE: If zone == "alarm" AND both contributor lists are empty
    if zone == "alarm" and not sop_risk_contributors and not context_contributors:
        explanation_status = "unexplained_high_risk"
        decision = "review"

    risk_analysis = {
        "risk_score": round(risk_score, 4),
        "zone": zone,
        "decision": decision,
        "explanation_status": explanation_status,
        "sop_risk_contributors": sop_risk_contributors,
        "context_contributors": context_contributors,
        "model_version": CONFIG.get("model_metadata", {}).get(
            "model_version", "pending"
        ),
    }

    # 7. Final Response
    final_response = {
        "batch_id": record.get("batch_id"),
        "timestamp": record.get("timestamp"),
        "sop_check": sop_check_response,
        "risk_analysis": risk_analysis,
    }

    return final_response
