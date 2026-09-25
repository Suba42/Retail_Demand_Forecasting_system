import pandas as pd
import numpy as np


# ==========================================
# MODULE 2.7 - MODEL COMPARISON
# ==========================================

PROPHET_FILE = "module2_prophet_validation_forecast.csv"
XGBOOST_FILE = "module2_xgboost_validation_forecast.csv"

print("Loading model validation results...")

prophet = pd.read_csv(PROPHET_FILE)
xgboost = pd.read_csv(XGBOOST_FILE)

prophet["ds"] = pd.to_datetime(prophet["ds"])
xgboost["date"] = pd.to_datetime(xgboost["date"])


print("\n==========================================")
print("MODULE 2.7 - MODEL COMPARISON")
print("==========================================")


# ------------------------------------------
# 1. Prophet validation period
# ------------------------------------------

prophet_start = prophet["ds"].min()
prophet_end = prophet["ds"].max()

print("\n--- COMMON VALIDATION PERIOD ---")

print(f"Start Date : {prophet_start}")
print(f"End Date   : {prophet_end}")


# ------------------------------------------
# 2. Filter XGBoost to same period
# ------------------------------------------

xgb_common = xgboost[
    (xgboost["date"] >= prophet_start) &
    (xgboost["date"] <= prophet_end)
].copy()


# ------------------------------------------
# 3. Aggregate XGBoost to daily level
# ------------------------------------------

xgb_daily = (
    xgb_common
    .groupby("date", as_index=False)
    .agg({
        "y_actual": "sum",
        "y_pred": "sum"
    })
)


# ------------------------------------------
# 4. Prepare Prophet daily results
# ------------------------------------------

prophet_daily = (
    prophet
    .groupby("ds", as_index=False)
    .agg({
        "y_actual": "sum",
        "y_pred": "sum"
    })
)


# ------------------------------------------
# 5. Rename columns
# ------------------------------------------

prophet_daily = prophet_daily.rename(
    columns={
        "ds": "date",
        "y_actual": "prophet_actual",
        "y_pred": "prophet_pred"
    }
)

xgb_daily = xgb_daily.rename(
    columns={
        "y_actual": "xgb_actual",
        "y_pred": "xgb_pred"
    }
)


# ------------------------------------------
# 6. Merge models
# ------------------------------------------

comparison = pd.merge(
    prophet_daily,
    xgb_daily,
    on="date",
    how="inner"
)


print("\n--- COMPARISON DATA ---")

print(f"Common Days : {len(comparison)}")


# ------------------------------------------
# 7. Calculate metrics
# ------------------------------------------

actual = comparison["prophet_actual"]

prophet_pred = comparison["prophet_pred"]

xgb_pred = comparison["xgb_pred"]


# Prophet metrics

prophet_mae = np.mean(
    np.abs(actual - prophet_pred)
)

prophet_rmse = np.sqrt(
    np.mean(
        (actual - prophet_pred) ** 2
    )
)

non_zero = actual != 0

prophet_mape = np.mean(
    np.abs(
        (
            actual[non_zero]
            - prophet_pred[non_zero]
        )
        / actual[non_zero]
    )
) * 100


# XGBoost metrics

xgb_mae = np.mean(
    np.abs(actual - xgb_pred)
)

xgb_rmse = np.sqrt(
    np.mean(
        (actual - xgb_pred) ** 2
    )
)

xgb_mape = np.mean(
    np.abs(
        (
            actual[non_zero]
            - xgb_pred[non_zero]
        )
        / actual[non_zero]
    )
) * 100


# ------------------------------------------
# 8. Comparison table
# ------------------------------------------

metrics = pd.DataFrame({
    "Model": [
        "Prophet",
        "XGBoost"
    ],
    "MAE": [
        prophet_mae,
        xgb_mae
    ],
    "RMSE": [
        prophet_rmse,
        xgb_rmse
    ],
    "MAPE": [
        prophet_mape,
        xgb_mape
    ]
})


print("\n--- MODEL PERFORMANCE COMPARISON ---")

print(
    metrics.to_string(index=False)
)


# ------------------------------------------
# 9. Determine best model
# ------------------------------------------

best_mae_model = metrics.loc[
    metrics["MAE"].idxmin(),
    "Model"
]

best_rmse_model = metrics.loc[
    metrics["RMSE"].idxmin(),
    "Model"
]

best_mape_model = metrics.loc[
    metrics["MAPE"].idxmin(),
    "Model"
]


print("\n--- BEST MODEL BY METRIC ---")

print(f"Best MAE  : {best_mae_model}")
print(f"Best RMSE : {best_rmse_model}")
print(f"Best MAPE : {best_mape_model}")


# ------------------------------------------
# 10. Final model selection
# ------------------------------------------

xgb_wins = (
    (xgb_mae < prophet_mae) +
    (xgb_rmse < prophet_rmse) +
    (xgb_mape < prophet_mape)
)

prophet_wins = (
    (prophet_mae < xgb_mae) +
    (prophet_rmse < xgb_rmse) +
    (prophet_mape < xgb_mape)
)


if xgb_wins > prophet_wins:
    selected_model = "XGBoost"
elif prophet_wins > xgb_wins:
    selected_model = "Prophet"
else:
    selected_model = "XGBoost"


print("\n--- FINAL MODEL SELECTION ---")

print(
    f"Selected Forecasting Model : {selected_model}"
)


# ------------------------------------------
# 11. Save comparison
# ------------------------------------------

metrics.to_csv(
    "module2_model_comparison.csv",
    index=False
)

comparison.to_csv(
    "module2_common_validation_comparison.csv",
    index=False
)


print("\n==========================================")
print("MODULE 2.7 COMPLETED")
print("==========================================")

print("Created files:")
print("✓ module2_model_comparison.csv")
print("✓ module2_common_validation_comparison.csv")

print(
    f"\nFinal Selected Model : {selected_model}"
)