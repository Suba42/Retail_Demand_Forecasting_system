import os

from app import build_reports_powerbi_context


def test_build_reports_powerbi_context_uses_existing_shared_dataset_and_report_files():
    context = build_reports_powerbi_context(
        shared_dataset_path=os.path.join("data", "cleaned", "cleaned_shared_dataset.csv"),
        forecast_report_path=os.path.join("data", "forecasts", "forecast_report.json"),
        inventory_results_path=os.path.join("data", "reports", "inventory_optimization_results.csv"),
    )

    assert context["enabled"] in (True, False)
    assert context["dataset_rows"] > 0
    assert context["dataset_columns"] > 0
    assert context["forecast_metrics_available"] is True
    assert context["inventory_available"] is True
    assert context["report_load_mode"] in {"iframe", "async", "fallback"}
    assert context["report_url"] or context["fallback_message"]
