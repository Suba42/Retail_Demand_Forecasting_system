import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.1 - INVENTORY DATA PREPARATION
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.1 - INVENTORY DATA PREPARATION")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/01_Demand_Forecasting/"
    "demand_forecasting_train_features.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_demand_prepared.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODULE 5 FEATURE-ENGINEERED DATASET
# ------------------------------------------------------------

print("Loading Module 5 feature-engineered dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "date",
    "store_nbr",
    "family",
    "sales"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified ✓")


# ------------------------------------------------------------
# 4. DATE CONVERSION
# ------------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

invalid_dates = df["date"].isna().sum()

print(f"Invalid dates : {invalid_dates:,}")

if invalid_dates > 0:
    df = df.dropna(subset=["date"])


# ------------------------------------------------------------
# 5. CLEAN KEY FIELDS
# ------------------------------------------------------------

df["store_nbr"] = pd.to_numeric(
    df["store_nbr"],
    errors="coerce"
)

df["sales"] = pd.to_numeric(
    df["sales"],
    errors="coerce"
)

df["family"] = df["family"].astype(str).str.strip()


# Remove rows with invalid essential values
df = df.dropna(
    subset=["store_nbr", "family", "sales"]
)


# Sales cannot be negative
negative_sales = (df["sales"] < 0).sum()

print(f"Negative sales records : {negative_sales:,}")

if negative_sales > 0:
    df.loc[df["sales"] < 0, "sales"] = 0


# ------------------------------------------------------------
# 6. INVENTORY-RELEVANT DATA
# ------------------------------------------------------------

inventory_columns = [
    "date",
    "store_nbr",
    "family",
    "sales"
]

# Include useful demand-driver columns if available
optional_columns = [
    "Transactions",
    "Promotion_Quantity",
    "Promotion_Status",
    "Holiday_Flag",
    "Festival_Flag",
    "oil_price"
]

for col in optional_columns:
    if col in df.columns:
        inventory_columns.append(col)

inventory_df = df[inventory_columns].copy()


# ------------------------------------------------------------
# 7. DAILY STORE × PRODUCT DEMAND
# ------------------------------------------------------------

print("\nPreparing daily store-product demand...")

daily_demand = (
    inventory_df
    .groupby(
        ["date", "store_nbr", "family"],
        as_index=False
    )
    .agg(
        daily_sales=("sales", "sum")
    )
)

print(
    f"Daily Store × Product records : "
    f"{len(daily_demand):,}"
)


# ------------------------------------------------------------
# 8. STORE × PRODUCT DEMAND SUMMARY
# ------------------------------------------------------------

print("\nCalculating historical demand statistics...")

inventory_summary = (
    daily_demand
    .groupby(
        ["store_nbr", "family"],
        as_index=False
    )
    .agg(
        total_sales=("daily_sales", "sum"),
        average_daily_demand=("daily_sales", "mean"),
        maximum_daily_demand=("daily_sales", "max"),
        demand_std=("daily_sales", "std"),
        active_days=("daily_sales", "count")
    )
)


# ------------------------------------------------------------
# 9. HANDLE ZERO / MISSING STANDARD DEVIATION
# ------------------------------------------------------------

inventory_summary["demand_std"] = (
    inventory_summary["demand_std"]
    .fillna(0)
)


# ------------------------------------------------------------
# 10. DEMAND VARIABILITY
# ------------------------------------------------------------

inventory_summary["demand_variability"] = np.where(
    inventory_summary["average_daily_demand"] > 0,
    inventory_summary["demand_std"]
    / inventory_summary["average_daily_demand"],
    0
)


# ------------------------------------------------------------
# 11. ROUND NUMERICAL VALUES
# ------------------------------------------------------------

numeric_columns = [
    "total_sales",
    "average_daily_demand",
    "maximum_daily_demand",
    "demand_std",
    "demand_variability"
]

inventory_summary[numeric_columns] = (
    inventory_summary[numeric_columns]
    .round(4)
)


# ------------------------------------------------------------
# 12. SORT OUTPUT
# ------------------------------------------------------------

inventory_summary = inventory_summary.sort_values(
    by=["store_nbr", "family"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 13. DISPLAY SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("INVENTORY DATA PREPARATION SUMMARY")
print("==========================================")

print(
    f"Unique Stores       : "
    f"{inventory_summary['store_nbr'].nunique()}"
)

print(
    f"Unique Product Types: "
    f"{inventory_summary['family'].nunique()}"
)

print(
    f"Store × Product combinations : "
    f"{len(inventory_summary):,}"
)

print(
    f"Average Daily Demand "
    f"(overall mean) : "
    f"{inventory_summary['average_daily_demand'].mean():.2f}"
)

print(
    f"Average Demand Std Dev : "
    f"{inventory_summary['demand_std'].mean():.2f}"
)


# ------------------------------------------------------------
# 14. SAVE OUTPUT
# ------------------------------------------------------------

inventory_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput saved successfully ✓")
print(f"File : {OUTPUT_FILE}")


# ------------------------------------------------------------
# 15. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample inventory demand data:")

print(
    inventory_summary.head(10).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 16. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.1 COMPLETED")
print("==========================================")