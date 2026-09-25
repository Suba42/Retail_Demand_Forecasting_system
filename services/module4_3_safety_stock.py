import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.3 - SAFETY STOCK CALCULATION
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.3 - SAFETY STOCK CALCULATION")
print("==========================================\n")


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

SERVICE_LEVEL = 0.95
Z_VALUE = 1.645

# Temporary configurable assumption.
# Replace with actual supplier lead time when available.
DEFAULT_LEAD_TIME_DAYS = 7


# ------------------------------------------------------------
# 2. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_demand_analysis.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_safety_stock.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD MODULE 4.2 OUTPUT
# ------------------------------------------------------------

print("Loading Module 4.2 demand analysis dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ------------------------------------------------------------
# 4. REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "store_nbr",
    "family",
    "average_daily_demand",
    "demand_std",
    "demand_variability"
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
# 5. NUMERIC CONVERSION
# ------------------------------------------------------------

numeric_columns = [
    "average_daily_demand",
    "demand_std",
    "demand_variability"
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
# 6. LEAD TIME
# ------------------------------------------------------------

df["lead_time_days"] = DEFAULT_LEAD_TIME_DAYS


# ------------------------------------------------------------
# 7. SAFETY STOCK CALCULATION
# ------------------------------------------------------------
#
# Formula:
#
# Safety Stock =
# Z × Demand Standard Deviation × √Lead Time
#
# Z = 1.645 for approximately 95% service level
# ------------------------------------------------------------

df["safety_stock"] = (
    Z_VALUE
    * df["demand_std"]
    * np.sqrt(df["lead_time_days"])
)


# ------------------------------------------------------------
# 8. ROUND SAFETY STOCK
# ------------------------------------------------------------

df["safety_stock"] = (
    df["safety_stock"]
    .round(2)
)


# ------------------------------------------------------------
# 9. SAFETY STOCK INTEGER VALUE
# ------------------------------------------------------------
#
# Inventory quantities should normally be whole units.
# We round upward so that protection stock is not
# accidentally reduced.
# ------------------------------------------------------------

df["safety_stock_units"] = (
    np.ceil(df["safety_stock"])
    .astype(int)
)


# ------------------------------------------------------------
# 10. SAFETY STOCK AS DAYS OF DEMAND
# ------------------------------------------------------------

df["safety_stock_days"] = np.where(
    df["average_daily_demand"] > 0,
    df["safety_stock"]
    / df["average_daily_demand"],
    0
)

df["safety_stock_days"] = (
    df["safety_stock_days"]
    .round(2)
)


# ------------------------------------------------------------
# 11. SERVICE LEVEL LABEL
# ------------------------------------------------------------

df["service_level"] = (
    SERVICE_LEVEL * 100
)

df["service_level"] = (
    df["service_level"]
    .astype(int)
    .astype(str)
    + "%"
)


# ------------------------------------------------------------
# 12. SAFETY STOCK CATEGORY
# ------------------------------------------------------------

def classify_safety_stock(row):

    if row["average_daily_demand"] <= 0:
        return "No Demand"

    ratio = (
        row["safety_stock"]
        / row["average_daily_demand"]
    )

    if ratio <= 0.25:
        return "Low Protection"

    elif ratio <= 0.75:
        return "Moderate Protection"

    else:
        return "High Protection"


df["safety_stock_category"] = (
    df.apply(
        classify_safety_stock,
        axis=1
    )
)


# ------------------------------------------------------------
# 13. SORT OUTPUT
# ------------------------------------------------------------

df = df.sort_values(
    by=[
        "safety_stock_units",
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
# 14. SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("SAFETY STOCK CONFIGURATION")
print("==========================================")

print(
    f"Service Level       : "
    f"{SERVICE_LEVEL * 100:.0f}%"
)

print(
    f"Z Value             : "
    f"{Z_VALUE}"
)

print(
    f"Default Lead Time   : "
    f"{DEFAULT_LEAD_TIME_DAYS} days"
)


# ------------------------------------------------------------
# 15. SAFETY STOCK STATISTICS
# ------------------------------------------------------------

print("\n==========================================")
print("SAFETY STOCK SUMMARY")
print("==========================================")

print(
    f"Average Safety Stock : "
    f"{df['safety_stock'].mean():.2f}"
)

print(
    f"Maximum Safety Stock : "
    f"{df['safety_stock'].max():.2f}"
)

print(
    f"Total Safety Stock   : "
    f"{df['safety_stock'].sum():.2f}"
)


# ------------------------------------------------------------
# 16. CATEGORY DISTRIBUTION
# ------------------------------------------------------------

print("\nSafety Stock Category Distribution:")

print(
    df["safety_stock_category"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 17. HIGH SAFETY STOCK ITEMS
# ------------------------------------------------------------

print("\n==========================================")
print("TOP SAFETY STOCK REQUIREMENTS")
print("==========================================")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "demand_std",
            "lead_time_days",
            "safety_stock_units",
            "safety_stock_category"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 18. SAVE OUTPUT
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
# 19. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample safety stock data:")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "demand_std",
            "lead_time_days",
            "safety_stock_units",
            "safety_stock_category"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 20. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.3 COMPLETED")
print("==========================================")