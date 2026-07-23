import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N = 5000

SOP_CURING = 30
SOP_TEMP_MIN = 80
SOP_TEMP_MAX = 85

machine = rng.choice(["M1", "M2", "M3"], size=N)
shift = rng.choice(["pagi", "siang", "malam"], size=N)
operator = rng.choice([f"OP{i:02d}" for i in range(1, 9)], size=N)
material = rng.choice(["A", "B", "C"], size=N)
is_short_draw = rng.random(N) < 0.30
curing = np.where(is_short_draw, rng.normal(24, 1.5, N), rng.normal(30, 1.5, N))
temp = rng.normal(82, 3.0, N)
pressure = rng.normal(105, 4.0, N)
p_override = np.where(shift == "malam", 0.35, 0.12)
manual_override = rng.random(N) < p_override
v_check_done = rng.random(N) < 0.80
p_fail = np.full(N, 0.05)
p_fail += np.where(temp < 78, 0.10, 0.0)
p_fail += np.where(manual_override & (material == "B"), 0.15, 0.0)
bahaya = (curing < 26) & (shift == "malam") & (machine == "M2")
p_fail += np.where(bahaya, 0.45, 0.0)
p_fail += np.where(bahaya & (material == "B"), 0.15, 0.0)
p_fail = np.clip(p_fail, 0.02, 0.95)
qc_result = np.where(rng.random(N) < p_fail, "fail", "pass")
df = pd.DataFrame(
    {
        "batch_id": [f"B{i:05d}" for i in range(N)],
        "timestamp": pd.date_range("2026-01-01", periods=N, freq="30min"),
        "machine_id": machine,
        "shift": shift,
        "operator_id": operator,
        "material_batch": material,
        "curing_time_actual": curing.round(1),
        "temp_actual": temp.round(1),
        "pressure_actual": pressure.round(1),
        "manual_override": manual_override,
        "visual_check_done": v_check_done,
        "qc_result": qc_result,
    }
)

df.to_csv("process_log.csv", index=False)
pd.DataFrame(
    {
        "parameter": ["curing_time", "temp_min", "temp_max"],
        "nilai": [SOP_CURING, SOP_TEMP_MIN, SOP_TEMP_MAX],
    }
).to_csv("sop_standard.csv", index=False)

print("Total batch:", len(df))
print("rasio gagal:", (df.qc_result == "fail").mean().round(3))
print("gagal di dalam kombinasi:", (qc_result[bahaya] == "fail").mean().round(3))
print("gagal di luar kombinasi:", (qc_result[~bahaya] == "fail").mean().round(3))
