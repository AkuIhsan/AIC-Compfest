from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# CONFIG

ROOT = Path(__file__).resolve().parent
PROCESS_LOG_ = ROOT / "process_log.csv"
SOP_ = ROOT / "sop_standard.csv"
GROUND_TRUTH_ = ROOT / "scenario_ground_truth.csv"
SEED = 42
N = 5000
TIMESTAMP_START = "2026-01-01"
TIMESTAMP_FREQ = "30min"
SHIFT_PAGI_START_HOUR = 6
SHIFT_SIANG_START_HOUR = 14
SHIFT_MALAM_START_HOUR = 22
MACHINE_IDS = ["M1", "M2", "M3"]
OPERATOR_IDS = [f"OP{i:02d}" for i in range(1, 9)]
MATERIAL_BATCHES = ["A", "B", "C"]
HIGH_RISK_MATERIAL = "B"
HIGH_RISK_MACHINE = "M2"
SOP_CURING = 30.0
CURING_TOLERANCE = 2.0
SOP_TEMP_MIN = 80.0
SOP_TEMP_MAX = 85.0
SOP_PRESSURE_MIN = 100.0
SOP_PRESSURE_MAX = 110.0
SOP_MANUAL_OVERRIDE_ALLOWED = 0
SOP_VISUAL_CHECK_REQUIRED = 1
SHORT_CURING_PROB_NIGHT = 0.40
SHORT_CURING_PROB_OTHER = 0.25
SHORT_CURING_MEAN = 24.0
SHORT_CURING_STD = 1.5
NORMAL_CURING_MEAN = 30.0
NORMAL_CURING_STD = 1.2
TEMP_MEAN = 82.0
TEMP_STD = 3.0
PRESSURE_MEAN = 105.0
PRESSURE_STD = 4.0
MANUAL_OVERRIDE_PROB_NIGHT = 0.35
MANUAL_OVERRIDE_PROB_OTHER = 0.12
VISUAL_CHECK_DONE_PROB = 0.80
TEMP_CRITICAL_LOW_THRESHOLD = 78.0
BASE_FAIL_PROBABILITY = 0.05
TEMP_CRITICAL_LOW_FAIL_BUMP = 0.10
OVERRIDE_MATERIAL_B_FAIL_BUMP = 0.15
DANGER_COMBO_FAIL_BUMP = 0.45
DANGER_COMBO_MATERIAL_B_FAIL_BUMP = 0.15
FAIL_PROBABILITY_MIN = 0.02
FAIL_PROBABILITY_MAX = 0.95


def main() -> None:
    """Generate data QC sintetis.

    Output-nya tiga file: process_log.csv untuk fitur training, sop_standard.csv
    untuk referensi SOP, dan scenario_ground_truth.csv untuk audit model.
    Invariant penting: deterministik dengan SEED, dan ground truth dipisah dari
    fitur training supaya model tidak "dibocorin" jawaban audit.
    """
    rng = np.random.default_rng(SEED)

    # Setup waktu & shift
    timestamp = pd.date_range(TIMESTAMP_START, periods=N, freq=TIMESTAMP_FREQ)
    hour = timestamp.hour
    shift = np.select(
        [
            (hour >= SHIFT_PAGI_START_HOUR) & (hour < SHIFT_SIANG_START_HOUR),
            (hour >= SHIFT_SIANG_START_HOUR) & (hour < SHIFT_MALAM_START_HOUR),
        ],
        ["pagi", "siang"],
        default="malam",
    )

    # Atribut batch acak
    machine = rng.choice(MACHINE_IDS, size=N)
    operator = rng.choice(OPERATOR_IDS, size=N)
    material = rng.choice(MATERIAL_BATCHES, size=N)

    # Besaran proses
    short_curing_probability = np.where(
        shift == "malam",
        SHORT_CURING_PROB_NIGHT,
        SHORT_CURING_PROB_OTHER,
    )
    is_short_curing = rng.random(N) < short_curing_probability

    curing = np.where(
        is_short_curing,
        rng.normal(SHORT_CURING_MEAN, SHORT_CURING_STD, N),
        rng.normal(NORMAL_CURING_MEAN, NORMAL_CURING_STD, N),
    ).round(1)

    temp = rng.normal(TEMP_MEAN, TEMP_STD, N).round(1)
    pressure = rng.normal(PRESSURE_MEAN, PRESSURE_STD, N).round(1)
    override_probability = np.where(
        shift == "malam",
        MANUAL_OVERRIDE_PROB_NIGHT,
        MANUAL_OVERRIDE_PROB_OTHER,
    )
    manual_override = rng.random(N) < override_probability
    visual_check_done = rng.random(N) < VISUAL_CHECK_DONE_PROB

    # Pengecekan SOP
    curing_too_short = curing < SOP_CURING - CURING_TOLERANCE
    curing_too_long = curing > SOP_CURING + CURING_TOLERANCE
    curing_outside_sop = curing_too_short | curing_too_long
    temp_outside_sop = (temp < SOP_TEMP_MIN) | (temp > SOP_TEMP_MAX)
    pressure_outside_sop = (pressure < SOP_PRESSURE_MIN) | (pressure > SOP_PRESSURE_MAX)
    visual_check_skipped = ~visual_check_done

    # Rule kualitas yang beneran ngaruh ke QC
    temp_critical_low = temp < TEMP_CRITICAL_LOW_THRESHOLD
    override_material_b = manual_override & (material == HIGH_RISK_MATERIAL)
    danger_combo = (
        curing_too_short & (shift == "malam") & (machine == HIGH_RISK_MACHINE)
    )
    danger_combo_material_b = danger_combo & (material == HIGH_RISK_MATERIAL)

    # Probabilitas fail & label QC
    fail_probability = np.full(N, BASE_FAIL_PROBABILITY)
    fail_probability += np.where(temp_critical_low, TEMP_CRITICAL_LOW_FAIL_BUMP, 0.0)
    fail_probability += np.where(
        override_material_b, OVERRIDE_MATERIAL_B_FAIL_BUMP, 0.0
    )
    fail_probability += np.where(danger_combo, DANGER_COMBO_FAIL_BUMP, 0.0)
    fail_probability += np.where(
        danger_combo_material_b,
        DANGER_COMBO_MATERIAL_B_FAIL_BUMP,
        0.0,
    )
    fail_probability = np.clip(
        fail_probability, FAIL_PROBABILITY_MIN, FAIL_PROBABILITY_MAX
    )
    qc_result = np.where(rng.random(N) < fail_probability, "fail", "pass")

    # Ground truth audit
    has_any_sop_deviation = (
        curing_outside_sop
        | temp_outside_sop
        | pressure_outside_sop
        | manual_override
        | visual_check_skipped
    )
    active_rule_count = (
        temp_critical_low.astype(int)
        + override_material_b.astype(int)
        + danger_combo.astype(int)
        + danger_combo_material_b.astype(int)
    )
    has_observed_driver = active_rule_count > 0
    audit_case = np.select(
        [
            (qc_result == "fail") & has_observed_driver,
            (qc_result == "fail") & ~has_observed_driver,
            (qc_result == "pass") & has_observed_driver,
        ],
        [
            "fail_with_observed_driver",
            "fail_without_observed_driver",
            "pass_despite_observed_driver",
        ],
        default="pass_without_observed_driver",
    )

    batch_id = [f"B{i:05d}" for i in range(N)]
    process_log = pd.DataFrame(
        {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "machine_id": machine,
            "shift": shift,
            "operator_id": operator,
            "material_batch": material,
            "curing_time_actual": curing,
            "temp_actual": temp,
            "pressure_actual": pressure,
            "manual_override": manual_override,
            "visual_check_done": visual_check_done,
            "qc_result": qc_result,
        }
    )
    process_log.to_csv(
        PROCESS_LOG_,
        index=False,
    )
    sop = pd.DataFrame(
        {
            "parameter": [
                "curing_time",
                "curing_tolerance",
                "temp_min",
                "temp_max",
                "pressure_min",
                "pressure_max",
                "manual_override_allowed",
                "visual_check_required",
            ],
            "nilai": [
                SOP_CURING,
                CURING_TOLERANCE,
                SOP_TEMP_MIN,
                SOP_TEMP_MAX,
                SOP_PRESSURE_MIN,
                SOP_PRESSURE_MAX,
                SOP_MANUAL_OVERRIDE_ALLOWED,
                SOP_VISUAL_CHECK_REQUIRED,
            ],
        }
    )
    sop.to_csv(
        SOP_,
        index=False,
    )

    # cuman audit model, jangan dijadiin fitur train
    ground_truth = pd.DataFrame(
        {
            "batch_id": batch_id,
            "curing_outside_sop": curing_outside_sop,
            "curing_too_short": curing_too_short,
            "curing_too_long": curing_too_long,
            "temp_outside_sop": temp_outside_sop,
            "pressure_outside_sop": pressure_outside_sop,
            "manual_override": manual_override,
            "visual_check_skipped": visual_check_skipped,
            "has_any_sop_deviation": has_any_sop_deviation,
            "has_observed_driver": has_observed_driver,
            "active_rule_count": active_rule_count,
            "audit_case": audit_case,
            "rule_temp_critical_low": temp_critical_low,
            "rule_override_material_b": override_material_b,
            "rule_danger_combo": danger_combo,
            "rule_danger_combo_material_b": (danger_combo_material_b),
            "true_fail_probability": (fail_probability.round(3)),
            "qc_result": qc_result,
        }
    )

    ground_truth.to_csv(
        GROUND_TRUTH_,
        index=False,
    )

    print("Total batch              :", N)
    print(
        "Rasio fail               :",
        round(float((qc_result == "fail").mean()), 4),
    )
    print(
        "Danger combo             :",
        int(danger_combo.sum()),
    )
    print(
        "Fail di danger combo     :",
        round(float((qc_result[danger_combo] == "fail").mean()), 4),
    )
    print(
        "Fail di luar danger combo:",
        round(float((qc_result[~danger_combo] == "fail").mean()), 4),
    )
    print(
        "Fail dengan driver      :",
        int(((qc_result == "fail") & has_observed_driver).sum()),
    )
    print(
        "Fail tanpa driver       :",
        int(((qc_result == "fail") & ~has_observed_driver).sum()),
    )
    print(
        "Pass meski ada driver   :",
        int(((qc_result == "pass") & has_observed_driver).sum()),
    )
    print("Process log              :", PROCESS_LOG_)
    print("Ground truth audit       :", GROUND_TRUTH_)


if __name__ == "__main__":
    main()
