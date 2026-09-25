import pandas as pd
import numpy as np

# ==========================================
# MODULE 2.5 - XGBOOST FEATURE PREPARATION
# ==========================================

INPUT_FILE = "module5_model_ready.csv"
OUTPUT_FILE = "module2_xgboost_ready.csv"

print("Loading model-ready dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 2.5 - XGBOOST FEATURE PREPARATION")
print("==========================================")

# ------------------------------------------
# 1. Convert date
# ------------------------------------------

df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------
# 2. Create calendar features
# ------------------------------------------

df["day_of_month"] = df["date"].dt.day
df["month_num"] = df["date"].dt.month
df["year_num"] = df["date"].dt.year
df["day_of_week_num"] = df["date"].dt.dayofweek
df["week_of_year_num"] = df["date"].dt.isocalendar().week.astype(int)

df["is_weekend_xgb"] = (
    df["date"].dt.dayofweek >= 5
).astype(int)

# ------------------------------------------
# 3. Safe feature list
# ------------------------------------------

safe_columns = [
    "date",
    "sales",
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

print("\n--- SELECTED FEATURES ---")

for column in safe_columns:
    print(f"✓ {column}")

xgb_df = df[safe_columns].copy()

# ------------------------------------------
# 4. Aggregate to daily store level
# ------------------------------------------

print("\nAggregating data to daily store level...")

daily_store = (
    xgb_df
    .groupby(
        ["date", "store_nbr"],
        as_index=False
    )
    .agg({
        "sales": "sum",
        "cluster": "first",
        "oil_price": "mean",
        "holiday_count": "max",
        "Festival_Flag": "max",
        "Promotion_Quantity": "sum",
        "Transactions": "sum",
        "day_of_month": "first",
        "month_num": "first",
        "year_num": "first",
        "day_of_week_num": "first",
        "week_of_year_num": "first",
        "is_weekend_xgb": "first"
    })
)

# ------------------------------------------
# 5. Sort
# ------------------------------------------

daily_store = (
    daily_store
    .sort_values(["date", "store_nbr"])
    .reset_index(drop=True)
)

# ------------------------------------------
# 6. Check output
# ------------------------------------------

print("\n--- XGBOOST DATASET ---")

print(f"Rows    : {len(daily_store)}")
print(f"Columns : {len(daily_store.columns)}")

print("\nColumns:")
print(list(daily_store.columns))

# ------------------------------------------
# 7. Missing values
# ------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing = daily_store.isnull().sum()

print(
    missing[missing > 0]
    if missing.sum() > 0
    else "No missing values found."
)

# ------------------------------------------
# 8. Duplicate check
# ------------------------------------------

print("\n--- DUPLICATE CHECK ---")

duplicates = daily_store.duplicated(
    subset=["date", "store_nbr"]
).sum()

print(
    f"Duplicate date-store records: {duplicates}"
)

# ------------------------------------------
# 9. Target statistics
# ------------------------------------------

print("\n--- TARGET STATISTICS ---")

print(
    daily_store["sales"].describe()
)

# ------------------------------------------
# 10. Leakage check
# ------------------------------------------

print("\n--- TARGET LEAKAGE CHECK ---")

leakage_features = [
    "Sales_per_Transaction",
    "sales"
]

for feature in leakage_features:

    if feature in daily_store.columns:
        print(f"⚠ {feature} present")
    else:
        print(f"✓ {feature} excluded")

# ------------------------------------------
# 11. Save
# ------------------------------------------

daily_store.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("MODULE 2.5 FEATURE PREPARATION COMPLETED")
print("==========================================")

print(f"Output File : {OUTPUT_FILE}")
print(f"Rows        : {len(daily_store)}")
print(f"Columns     : {len(daily_store.columns)}")