import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# MODULE 2.8 - FINAL XGBOOST FORECAST
# ==========================================

INPUT_FILE = "module2_xgboost_ready.csv"

print("Loading XGBoost-ready dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


print("\n==========================================")
print("MODULE 2.8 - FINAL XGBOOST FORECAST")
print("==========================================")


# ------------------------------------------
# 1. Sort data
# ------------------------------------------

df = df.sort_values(
    ["date", "store_nbr"]
).reset_index(drop=True)


# ------------------------------------------
# 2. Features and target
# ------------------------------------------

TARGET = "sales"

FEATURES = [
    "store_nbr",
    "cluster",
    "oil_price",
    "holiday_count",
    "Festival_Flag",
    "Promotion_Quantity",
    "Transactions",
    "day_of_month",
    "month_num",
    "year_num",
    "day_of_week_num",
    "week_of_year_num",
    "is_weekend_xgb"
]


# ------------------------------------------
# 3. Time-based split
# ------------------------------------------

unique_dates = sorted(
    df["date"].unique()
)

total_days = len(unique_dates)

train_days = int(total_days * 0.80)

train_end_date = unique_dates[train_days - 1]

train_df = df[
    df["date"] <= train_end_date
].copy()

test_df = df[
    df["date"] > train_end_date
].copy()


print("\n--- FINAL TRAIN / TEST SPLIT ---")

print(f"Total Days : {total_days}")

print(
    f"Training Period : "
    f"{train_df['date'].min()} to "
    f"{train_df['date'].max()}"
)

print(
    f"Test Period     : "
    f"{test_df['date'].min()} to "
    f"{test_df['date'].max()}"
)

print(f"Training Rows : {len(train_df)}")
print(f"Test Rows     : {len(test_df)}")


# ------------------------------------------
# 4. Prepare training data
# ------------------------------------------

X_train = train_df[FEATURES]

y_train = train_df[TARGET]

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


# ------------------------------------------
# 5. Create final XGBoost model
# ------------------------------------------

print("\nCreating final XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------
# 6. Train final model
# ------------------------------------------

print("\nTraining final XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("✓ Final XGBoost model training completed.")


# ------------------------------------------
# 7. Test prediction
# ------------------------------------------

print("\nGenerating test forecasts...")

y_pred = model.predict(X_test)


# ------------------------------------------
# 8. Test evaluation
# ------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

non_zero = y_test != 0

if non_zero.sum() > 0:

    mape = np.mean(
        np.abs(
            (
                y_test[non_zero]
                - y_pred[non_zero]
            )
            / y_test[non_zero]
        )
    ) * 100

else:
    mape = np.nan


print("\n--- FINAL XGBOOST TEST PERFORMANCE ---")

print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")


# ------------------------------------------
# 9. Forecast results
# ------------------------------------------

forecast = test_df[
    ["date", "store_nbr", "sales"]
].copy()

forecast["actual_sales"] = forecast["sales"]

forecast["predicted_sales"] = y_pred

forecast["forecast_error"] = (
    forecast["actual_sales"]
    - forecast["predicted_sales"]
)

forecast["absolute_error"] = np.abs(
    forecast["forecast_error"]
)

forecast = forecast.drop(
    columns=["sales"]
)


# ------------------------------------------
# 10. Daily total forecast
# ------------------------------------------

daily_forecast = (
    forecast
    .groupby("date", as_index=False)
    .agg({
        "actual_sales": "sum",
        "predicted_sales": "sum"
    })
)

daily_forecast["forecast_error"] = (
    daily_forecast["actual_sales"]
    - daily_forecast["predicted_sales"]
)

daily_forecast["absolute_error"] = np.abs(
    daily_forecast["forecast_error"]
)


# ------------------------------------------
# 11. Store-wise forecast
# ------------------------------------------

store_forecast = (
    forecast
    .groupby("store_nbr", as_index=False)
    .agg({
        "actual_sales": "sum",
        "predicted_sales": "sum"
    })
)

store_forecast["forecast_error"] = (
    store_forecast["actual_sales"]
    - store_forecast["predicted_sales"]
)


# ------------------------------------------
# 12. Feature importance
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


print("\n--- FINAL FEATURE IMPORTANCE ---")

print(importance)


# ------------------------------------------
# 13. Sample forecasts
# ------------------------------------------

print("\n--- SAMPLE FINAL FORECASTS ---")

print(
    forecast.head(10)
)


# ------------------------------------------
# 14. Save files
# ------------------------------------------

forecast.to_csv(
    "module2_final_xgboost_forecast.csv",
    index=False
)

daily_forecast.to_csv(
    "module2_daily_final_forecast.csv",
    index=False
)

store_forecast.to_csv(
    "module2_store_final_forecast.csv",
    index=False
)

importance.to_csv(
    "module2_final_feature_importance.csv",
    index=False
)


# ------------------------------------------
# 15. Save metrics
# ------------------------------------------

metrics = pd.DataFrame({
    "Model": ["Final XGBoost"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape]
})

metrics.to_csv(
    "module2_final_model_metrics.csv",
    index=False
)


print("\n==========================================")
print("MODULE 2.8 COMPLETED")
print("==========================================")

print("Created files:")

print("✓ module2_final_xgboost_forecast.csv")
print("✓ module2_daily_final_forecast.csv")
print("✓ module2_store_final_forecast.csv")
print("✓ module2_final_feature_importance.csv")
print("✓ module2_final_model_metrics.csv")

print("\nFinal Model: XGBoost")

print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")