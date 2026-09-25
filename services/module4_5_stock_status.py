import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.5 - STOCK STATUS ANALYSIS
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.5 - STOCK STATUS ANALYSIS")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_reorder_point.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_stock_status.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODULE 4.4 OUTPUT
# ------------------------------------------------------------

print("Loading Module 4.4 reorder point dataset...")

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
    "reorder_point_units"
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
# 4. CURRENT STOCK INPUT
# ------------------------------------------------------------
#
# Actual current-stock data is not available in the
# historical forecasting dataset.
#
# Therefore, we DO NOT invent stock quantities.
#
# NaN means:
# "Current inventory data not supplied yet."
#
# This allows the same dataset to accept real inventory
# values later from the web application / inventory upload.
# ------------------------------------------------------------

df["current_stock"] = np.nan


# ------------------------------------------------------------
# 5. STOCK DATA AVAILABILITY
# ------------------------------------------------------------

df["stock_data_status"] = np.where(
    df["current_stock"].notna(),
    "Available",
    "Not Available"
)


# ------------------------------------------------------------
# 6. STOCK STATUS
# ------------------------------------------------------------

def determine_stock_status(row):

    if pd.isna(row["current_stock"]):
        return "Stock Data Required"

    current_stock = row["current_stock"]
    reorder_point = row["reorder_point_units"]

    if current_stock <= 0:
        return "Critical - Out of Stock"

    elif current_stock < reorder_point:
        return "Reorder Required"

    elif current_stock <= reorder_point * 1.25:
        return "Low Stock"

    elif current_stock <= reorder_point * 2:
        return "Healthy Stock"

    else:
        return "Potential Overstock"


df["stock_status"] = (
    df.apply(
        determine_stock_status,
        axis=1
    )
)


# ------------------------------------------------------------
# 7. STOCK GAP
# ------------------------------------------------------------
#
# Stock Gap =
# Current Stock - Reorder Point
#
# Positive -> above reorder point
# Negative -> below reorder point
#
# Since current stock is currently unavailable,
# the value remains NaN.
# ------------------------------------------------------------

df["stock_gap"] = (
    df["current_stock"]
    - df["reorder_point_units"]
)


# ------------------------------------------------------------
# 8. STOCK COVERAGE DAYS
# ------------------------------------------------------------
#
# Stock Coverage Days =
# Current Stock / Average Daily Demand
#
# Indicates approximately how many days current inventory
# can support expected daily demand.
# ------------------------------------------------------------

df["stock_coverage_days"] = np.where(
    (
        df["current_stock"].notna()
        & (df["average_daily_demand"] > 0)
    ),
    df["current_stock"]
    / df["average_daily_demand"],
    np.nan
)


# ------------------------------------------------------------
# 9. REORDER GAP
# ------------------------------------------------------------
#
# Reorder Gap =
# Reorder Point - Current Stock
#
# Positive -> quantity below reorder point
# Negative -> stock is above reorder point
# ------------------------------------------------------------

df["reorder_gap"] = (
    df["reorder_point_units"]
    - df["current_stock"]
)


# ------------------------------------------------------------
# 10. ROUND NUMERICAL VALUES
# ------------------------------------------------------------

decimal_columns = [
    "average_daily_demand",
    "stock_gap",
    "stock_coverage_days",
    "reorder_gap"
]

for col in decimal_columns:
    df[col] = df[col].round(2)


# ------------------------------------------------------------
# 11. STOCK STATUS PRIORITY
# ------------------------------------------------------------

status_priority = {
    "Critical - Out of Stock": 1,
    "Reorder Required": 2,
    "Low Stock": 3,
    "Healthy Stock": 4,
    "Potential Overstock": 5,
    "Stock Data Required": 6
}

df["status_priority"] = (
    df["stock_status"]
    .map(status_priority)
    .fillna(99)
    .astype(int)
)


# ------------------------------------------------------------
# 12. SORT DATA
# ------------------------------------------------------------

df = df.sort_values(
    by=[
        "status_priority",
        "store_nbr",
        "family"
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 13. SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("STOCK DATA SUMMARY")
print("==========================================")

available_stock = (
    df["current_stock"]
    .notna()
    .sum()
)

missing_stock = (
    df["current_stock"]
    .isna()
    .sum()
)

print(
    f"Stock records available : "
    f"{available_stock:,}"
)

print(
    f"Stock records missing   : "
    f"{missing_stock:,}"
)


# ------------------------------------------------------------
# 14. STOCK STATUS DISTRIBUTION
# ------------------------------------------------------------

print("\n==========================================")
print("STOCK STATUS DISTRIBUTION")
print("==========================================")

print(
    df["stock_status"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 15. DATA REQUIRED MESSAGE
# ------------------------------------------------------------

if missing_stock > 0:

    print("\n------------------------------------------")
    print("IMPORTANT")
    print("------------------------------------------")

    print(
        "Actual current-stock data is not available "
        "in the existing forecasting dataset."
    )

    print(
        "Therefore, stock status cannot yet be classified "
        "as Reorder / Healthy / Overstock."
    )

    print(
        "Current stock values should be supplied through "
        "inventory records or the web application."
    )


# ------------------------------------------------------------
# 16. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\n==========================================")
print("SAMPLE STOCK STATUS DATA")
print("==========================================")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "reorder_point_units",
            "current_stock",
            "stock_status"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 17. SAVE OUTPUT
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
# 18. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.5 COMPLETED")
print("==========================================")