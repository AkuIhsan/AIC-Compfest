import json
from pathlib import Path
import inference

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"


def run_smoke_test():
    print("Starting Smoke Test...")

    # 1. Load sample input
    with open(SAMPLES / "sample_input.json", "r", encoding="utf-8") as f:
        record = json.load(f)

    # 2. Assert exactly 28 features are configured
    assert len(inference.FEATURE_COLUMNS) == 28, "Feature count is not 28"
    print("[OK] Asserted 28 features configured in schema.")

    # 3. Call inference predict
    result = inference.predict(record)

    # 4. Assert output contains all required keys
    risk_analysis = result.get("risk_analysis", {})
    required_keys = [
        "risk_score",
        "zone",
        "decision",
        "explanation_status",
        "sop_risk_contributors",
        "context_contributors",
        "model_version",
    ]
    for key in required_keys:
        assert key in risk_analysis, f"Missing required key in risk_analysis: {key}"
    print("[OK] All required keys present.")

    # 5. Assert no operator feature leaked into the output
    all_contributors = (
        risk_analysis["sop_risk_contributors"] + risk_analysis["context_contributors"]
    )
    for contrib in all_contributors:
        assert not contrib["id"].startswith("op_"), (
            f"Operator feature leaked into output: {contrib['id']}"
        )
    print("[OK]  No operator features leaked.")

    # 6. Check override logic structure
    assert risk_analysis["zone"] in ["monitor", "review", "alarm"], "Invalid zone"
    assert risk_analysis["decision"] in ["monitor", "review", "alarm"], (
        "Invalid decision value"
    )
    assert risk_analysis["explanation_status"] in [
        "success",
        "no_positive_driver",
        "unexplained_high_risk",
    ], "Invalid explanation_status"
    print("[OK] Valid enums for zone, decision, explanation_status.")

    assert risk_analysis["model_version"] != "pending", (
        "model_version still 'pending; -> config not read correctly "
    )
    print("[OK] model_version read from config.")

    expected_path = SAMPLES / "expected_output.json"
    if expected_path.exists():
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)
        exp_risk = expected.get("risk_analysis", {})

        if exp_risk.get("model_version") == "pending":
            print(
                "[SKIP] expected_output.json is still a placeholder, not yet compared."
            )
        else:
            assert risk_analysis["zone"] == exp_risk["zone"], (
                f"Zone beda: {risk_analysis['zone']} vs {exp_risk['zone']}"
            )
            assert abs(risk_analysis["risk_score"] - exp_risk["risk_score"]) < 1e-4, (
                f"risk_score beda: {risk_analysis['risk_score']} vs {exp_risk['risk_score']}"
            )
            print("[OK] output matches with expected_output.json.")

    print("\nSmoke test passed successfully!")


if __name__ == "__main__":
    run_smoke_test()
