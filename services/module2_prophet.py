import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# MODULE 2.4 - PROPHET FORECASTING MODEL
# ==========================================

TRAIN_FILE = "module2_train.csv"
VALIDATION_FILE = "module2_validation.csv"

FORECAST_FILE = "module2_prophet_validation_forecast.csv"

print("Loading training and validation datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])

print(f"Training rows   : {len(train)}")
print(f"Validation rows : {len(validation)}")

print("\n==========================================")
print("MODULE 2.4 - PROPHET FORECASTING")
print("==========================================")


# ------------------------------------------
# 1. Prepare Prophet format
# ------------------------------------------

prophet_train = train[
    ["date", "sales"]
].rename(
    columns={
        "date": "ds",
        "sales": "y"
    }
)

prophet_validation = validation[
    ["date", "sales"]
].rename(
    columns={
        "date": "ds",
        "sales": "y_actual"
    }
)

print("\n--- PROPHET DATA FORMAT ---")

print(prophet_train.head())

print(
    f"\nTraining period : "
    f"{prophet_train['ds'].min()} "
    f"to "
    f"{prophet_train['ds'].max()}"
)

print(
    f"Validation period : "
    f"{prophet_validation['ds'].min()} "
    f"to "
    f"{prophet_validation['ds'].max()}"
)


# ------------------------------------------
# 2. Create Prophet model
# ------------------------------------------

print("\nCreating Prophet model...")

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)


# ------------------------------------------
# 3. Train model
# ------------------------------------------

print("\nTraining Prophet model...")

model.fit(prophet_train)

print("✓ Prophet model training completed.")


# ------------------------------------------
# 4. Generate validation forecast
# ------------------------------------------

print("\nGenerating validation forecast...")

future = prophet_validation[
    ["ds"]
].copy()

forecast = model.predict(future)


# ------------------------------------------
# 5. Combine actual + prediction
# ------------------------------------------

results = prophet_validation.copy()

results["y_pred"] = forecast["yhat"].values

results["y_lower"] = forecast[
    "yhat_lower"
].values

results["y_upper"] = forecast[
    "yhat_upper"
].values


# ------------------------------------------
# 6. Prevent negative predictions
# ------------------------------------------

results["y_pred"] = results[
    "y_pred"
].clip(lower=0)


# ------------------------------------------
# 7. Evaluation metrics
# ------------------------------------------

y_actual = results["y_actual"]
y_pred = results["y_pred"]

mae = mean_absolute_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

# MAPE excluding zero actual values
non_zero = y_actual != 0

if non_zero.sum() > 0:
    mape = np.mean(
        np.abs(
            (
                y_actual[non_zero]
                - y_pred[non_zero]
            )
            / y_actual[non_zero]
        )
    ) * 100
else:
    mape = np.nan


# ------------------------------------------
# 8. Display metrics
# ------------------------------------------

print("\n--- PROPHET VALIDATION PERFORMANCE ---")

print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")


# ------------------------------------------
# 9. Display sample predictions
# ------------------------------------------

print("\n--- SAMPLE FORECAST RESULTS ---")

print(
    results[
        [
            "ds",
            "y_actual",
            "y_pred",
            "y_lower",
            "y_upper"
        ]
    ].head(10)
)


# ------------------------------------------
# 10. Save validation forecast
# ------------------------------------------

results.to_csv(
    FORECAST_FILE,
    index=False
)


# ------------------------------------------
# 11. Completion
# ------------------------------------------

print("\n==========================================")
print("MODULE 2.4 COMPLETED")
print("==========================================")

print("Prophet validation forecast saved:")
print(f"✓ {FORECAST_FILE}")

print("\nModel metrics:")
print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")