import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.4 - REORDER POINT CALCULATION
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.4 - REORDER POINT CALCULATION")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_safety_stock.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_reorder_point.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODULE 4.3 OUTPUT
# ------------------------------------------------------------

print("Loading Module 4.3 safety stock dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ------------------------------------------------------------
# 3. REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "store_nbr",
    "family",
    "average_daily_demand",
    "demand_std",
    "lead_time_days",
    "safety_stock_units"
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
# 4. NUMERIC CLEANING
# ------------------------------------------------------------

numeric_columns = [
    "average_daily_demand",
    "demand_std",
    "lead_time_days",
    "safety_stock_units"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[numeric_columns] = (
    df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ------------------------------------------------------------
# 5. LEAD TIME DEMAND
# ------------------------------------------------------------
#
# Lead Time Demand =
# Average Daily Demand × Lead Time
#
# This represents expected demand during the time
# required for replenishment.
# ------------------------------------------------------------

df["lead_time_demand"] = (
    df["average_daily_demand"]
    * df["lead_time_days"]
)


# ------------------------------------------------------------
# 6. REORDER POINT
# ------------------------------------------------------------
#
# Reorder Point =
# Lead Time Demand + Safety Stock
#
# ROP =
# (Average Daily Demand × Lead Time)
# + Safety Stock
# ------------------------------------------------------------

df["reorder_point"] = (
    df["lead_time_demand"]
    + df["safety_stock_units"]
)


# ------------------------------------------------------------
# 7. ROUND UP TO WHOLE UNITS
# ------------------------------------------------------------

df["lead_time_demand_units"] = (
    np.ceil(df["lead_time_demand"])
    .astype(int)
)

df["reorder_point_units"] = (
    np.ceil(df["reorder_point"])
    .astype(int)
)


# ------------------------------------------------------------
# 8. COVERAGE DAYS
# ------------------------------------------------------------
#
# Reorder Point Coverage =
# Reorder Point / Average Daily Demand
#
# Shows approximately how many days of demand are
# represented by the reorder point.
# ------------------------------------------------------------

df["reorder_point_days"] = np.where(
    df["average_daily_demand"] > 0,
    df["reorder_point_units"]
    / df["average_daily_demand"],
    0
)

df["reorder_point_days"] = (
    df["reorder_point_days"]
    .round(2)
)


# ------------------------------------------------------------
# 9. REORDER LEVEL CLASSIFICATION
# ------------------------------------------------------------

def classify_reorder_level(row):

    avg_demand = row["average_daily_demand"]
    reorder_point = row["reorder_point_units"]

    if avg_demand <= 0:
        return "No Demand"

    coverage = (
        reorder_point / avg_demand
    )

    if coverage <= 10:
        return "Standard Reorder Level"

    elif coverage <= 20:
        return "High Reorder Level"

    else:
        return "Very High Reorder Level"


df["reorder_level"] = (
    df.apply(
        classify_reorder_level,
        axis=1
    )
)


# ------------------------------------------------------------
# 10. ROUND DECIMAL VALUES
# ------------------------------------------------------------

decimal_columns = [
    "average_daily_demand",
    "demand_std",
    "lead_time_demand",
    "reorder_point"
]

df[decimal_columns] = (
    df[decimal_columns]
    .round(2)
)


# ------------------------------------------------------------
# 11. SORT BY REORDER POINT
# ------------------------------------------------------------

df = df.sort_values(
    by=[
        "reorder_point_units",
        "store_nbr",
        "family"
    ],
    ascending=[
        False,
        True,
        True
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 12. CONFIGURATION SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("REORDER POINT CONFIGURATION")
print("==========================================")

print(
    f"Service Level : "
    f"{df['service_level'].iloc[0]}"
    if "service_level" in df.columns
    else "Service Level : 95%"
)

print(
    f"Average Lead Time : "
    f"{df['lead_time_days'].mean():.2f} days"
)


# ------------------------------------------------------------
# 13. REORDER POINT STATISTICS
# ------------------------------------------------------------

print("\n==========================================")
print("REORDER POINT SUMMARY")
print("==========================================")

print(
    f"Average Lead-Time Demand : "
    f"{df['lead_time_demand'].mean():.2f}"
)

print(
    f"Average Safety Stock : "
    f"{df['safety_stock_units'].mean():.2f}"
)

print(
    f"Average Reorder Point : "
    f"{df['reorder_point_units'].mean():.2f}"
)

print(
    f"Maximum Reorder Point : "
    f"{df['reorder_point_units'].max():,}"
)

print(
    f"Total Reorder Point : "
    f"{df['reorder_point_units'].sum():,}"
)


# ------------------------------------------------------------
# 14. REORDER LEVEL DISTRIBUTION
# ------------------------------------------------------------

print("\nReorder Level Distribution:")

print(
    df["reorder_level"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 15. TOP REORDER POINTS
# ------------------------------------------------------------

print("\n==========================================")
print("TOP REORDER POINT REQUIREMENTS")
print("==========================================")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "lead_time_days",
            "lead_time_demand_units",
            "safety_stock_units",
            "reorder_point_units",
            "reorder_level"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 16. SAVE OUTPUT
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("OUTPUT SAVED")
print("==========================================")

print(
    f"File : {OUTPUT_FILE}"
)


# ------------------------------------------------------------
# 17. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample reorder point data:")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "lead_time_demand_units",
            "safety_stock_units",
            "reorder_point_units",
            "reorder_point_days"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 18. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.4 COMPLETED")
print("==========================================")