import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# MODULE 2.6 - XGBOOST FORECASTING
# ==========================================

INPUT_FILE = "module2_xgboost_ready.csv"

print("Loading XGBoost-ready dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


print("\n==========================================")
print("MODULE 2.6 - XGBOOST FORECASTING")
print("==========================================")


# ------------------------------------------
# 1. Sort data chronologically
# ------------------------------------------

df = df.sort_values(
    ["date", "store_nbr"]
).reset_index(drop=True)


# ------------------------------------------
# 2. Define target
# ------------------------------------------

TARGET = "sales"


# ------------------------------------------
# 3. Define features
# ------------------------------------------

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


print("\n--- FEATURES ---")

for feature in FEATURES:
    print(f"✓ {feature}")


# ------------------------------------------
# 4. Time-based split
# ------------------------------------------

unique_dates = sorted(
    df["date"].unique()
)

total_days = len(unique_dates)

train_days = int(total_days * 0.80)
validation_days = total_days - train_days

train_end_date = unique_dates[train_days - 1]

validation_start_date = unique_dates[train_days]


train_df = df[
    df["date"] <= train_end_date
].copy()

validation_df = df[
    df["date"] >= validation_start_date
].copy()


print("\n--- TRAIN / VALIDATION SPLIT ---")

print(f"Total Days      : {total_days}")
print(f"Training Days   : {len(train_df['date'].unique())}")
print(f"Validation Days : {len(validation_df['date'].unique())}")

print(
    f"Training Period   : "
    f"{train_df['date'].min()} to {train_df['date'].max()}"
)

print(
    f"Validation Period : "
    f"{validation_df['date'].min()} to "
    f"{validation_df['date'].max()}"
)

print(f"Training Rows     : {len(train_df)}")
print(f"Validation Rows   : {len(validation_df)}")


# ------------------------------------------
# 5. Prepare X and y
# ------------------------------------------

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]


# ------------------------------------------
# 6. Create XGBoost model
# ------------------------------------------

print("\nCreating XGBoost model...")

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
# 7. Train model
# ------------------------------------------

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("✓ XGBoost model training completed.")


# ------------------------------------------
# 8. Validation prediction
# ------------------------------------------

print("\nGenerating validation predictions...")

y_pred = model.predict(
    X_validation
)


# ------------------------------------------
# 9. Evaluation
# ------------------------------------------

mae = mean_absolute_error(
    y_validation,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        y_pred
    )
)


# MAPE excluding zero actual values
non_zero = y_validation != 0

if non_zero.sum() > 0:

    mape = np.mean(
        np.abs(
            (
                y_validation[non_zero]
                - y_pred[non_zero]
            )
            / y_validation[non_zero]
        )
    ) * 100

else:
    mape = np.nan


print("\n--- XGBOOST VALIDATION PERFORMANCE ---")

print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")


# ------------------------------------------
# 10. Sample predictions
# ------------------------------------------

results = validation_df[
    ["date", "store_nbr", "sales"]
].copy()

results["y_actual"] = results["sales"]

results["y_pred"] = y_pred

results["error"] = (
    results["y_actual"]
    - results["y_pred"]
)

results["absolute_error"] = (
    np.abs(results["error"])
)

results = results.drop(
    columns=["sales"]
)


print("\n--- SAMPLE XGBOOST RESULTS ---")

print(
    results.head(10)
)


# ------------------------------------------
# 11. Feature importance
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


print("\n--- FEATURE IMPORTANCE ---")

print(
    importance
)


# ------------------------------------------
# 12. Save predictions
# ------------------------------------------

OUTPUT_FILE = (
    "module2_xgboost_validation_forecast.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------
# 13. Save feature importance
# ------------------------------------------

IMPORTANCE_FILE = (
    "module2_xgboost_feature_importance.csv"
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


print("\n==========================================")
print("MODULE 2.6 COMPLETED")
print("==========================================")

print(
    f"Validation Forecast : {OUTPUT_FILE}"
)

print(
    f"Feature Importance   : {IMPORTANCE_FILE}"
)

print("\nModel Metrics:")
print(f"MAE  : {mae:,.4f}")
print(f"RMSE : {rmse:,.4f}")
print(f"MAPE : {mape:.2f}%")