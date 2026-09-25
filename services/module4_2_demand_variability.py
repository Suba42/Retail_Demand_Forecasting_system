import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.2 - DEMAND & VARIABILITY ANALYSIS
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.2 - DEMAND & VARIABILITY ANALYSIS")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_demand_prepared.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_demand_analysis.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODULE 4.1 OUTPUT
# ------------------------------------------------------------

print("Loading Module 4.1 inventory dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ------------------------------------------------------------
# 3. REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "store_nbr",
    "family",
    "total_sales",
    "average_daily_demand",
    "maximum_daily_demand",
    "demand_std",
    "active_days",
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
# 4. NUMERIC CLEANING
# ------------------------------------------------------------

numeric_columns = [
    "total_sales",
    "average_daily_demand",
    "maximum_daily_demand",
    "demand_std",
    "active_days",
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
# 5. DEMAND CLASSIFICATION
# ------------------------------------------------------------
#
# Coefficient of Variation (CV):
#
# CV = Standard Deviation / Average Demand
#
# Lower CV  -> More stable demand
# Higher CV -> More variable demand
#
# We use practical thresholds:
#
# CV <= 0.50       -> Low Variability
# 0.50 < CV <= 1.00 -> Medium Variability
# CV > 1.00        -> High Variability
#
# For products with zero demand, classify separately.
# ------------------------------------------------------------

def classify_variability(row):

    avg_demand = row["average_daily_demand"]
    cv = row["demand_variability"]

    if avg_demand <= 0:
        return "No Demand"

    elif cv <= 0.50:
        return "Low Variability"

    elif cv <= 1.00:
        return "Medium Variability"

    else:
        return "High Variability"


df["demand_variability_class"] = (
    df.apply(
        classify_variability,
        axis=1
    )
)


# ------------------------------------------------------------
# 6. DEMAND LEVEL CLASSIFICATION
# ------------------------------------------------------------
#
# We classify demand using the overall distribution
# of average daily demand.
#
# This is relative to our dataset and is NOT a business
# assumption about specific products.
# ------------------------------------------------------------

demand_25 = df["average_daily_demand"].quantile(0.25)
demand_75 = df["average_daily_demand"].quantile(0.75)


def classify_demand_level(value):

    if value <= 0:
        return "No Demand"

    elif value <= demand_25:
        return "Low Demand"

    elif value <= demand_75:
        return "Medium Demand"

    else:
        return "High Demand"


df["demand_level"] = (
    df["average_daily_demand"]
    .apply(classify_demand_level)
)


# ------------------------------------------------------------
# 7. INVENTORY DEMAND PRIORITY
# ------------------------------------------------------------
#
# High demand + high variability products need
# greater inventory attention.
#
# This is an analytical priority indicator.
# It will NOT replace the actual reorder calculation.
# ------------------------------------------------------------

def calculate_priority(row):

    demand_level = row["demand_level"]
    variability = row["demand_variability_class"]

    if (
        demand_level == "High Demand"
        and variability == "High Variability"
    ):
        return "Critical Attention"

    elif (
        demand_level == "High Demand"
        or variability == "High Variability"
    ):
        return "High Attention"

    elif (
        demand_level == "Medium Demand"
        or variability == "Medium Variability"
    ):
        return "Medium Attention"

    elif demand_level == "No Demand":
        return "No Demand"

    else:
        return "Low Attention"


df["inventory_priority"] = (
    df.apply(
        calculate_priority,
        axis=1
    )
)


# ------------------------------------------------------------
# 8. DEMAND RANGE
# ------------------------------------------------------------

df["demand_range"] = (
    df["maximum_daily_demand"]
    - df["average_daily_demand"]
)


# ------------------------------------------------------------
# 9. PEAK DEMAND RATIO
# ------------------------------------------------------------
#
# Shows how much higher the maximum demand is compared
# with the average demand.
# ------------------------------------------------------------

df["peak_demand_ratio"] = np.where(
    df["average_daily_demand"] > 0,
    df["maximum_daily_demand"]
    / df["average_daily_demand"],
    0
)


# ------------------------------------------------------------
# 10. ROUND VALUES
# ------------------------------------------------------------

round_columns = [
    "total_sales",
    "average_daily_demand",
    "maximum_daily_demand",
    "demand_std",
    "demand_variability",
    "demand_range",
    "peak_demand_ratio"
]

df[round_columns] = (
    df[round_columns]
    .round(4)
)


# ------------------------------------------------------------
# 11. SORT DATA
# ------------------------------------------------------------

df = df.sort_values(
    by=[
        "inventory_priority",
        "store_nbr",
        "family"
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 12. SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("DEMAND ANALYSIS SUMMARY")
print("==========================================")

print("\nDemand Level Distribution:")

print(
    df["demand_level"]
    .value_counts()
    .to_string()
)


print("\nDemand Variability Distribution:")

print(
    df["demand_variability_class"]
    .value_counts()
    .to_string()
)


print("\nInventory Priority Distribution:")

print(
    df["inventory_priority"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 13. DATASET STATISTICS
# ------------------------------------------------------------

print("\n==========================================")
print("DEMAND STATISTICS")
print("==========================================")

print(
    f"Average Daily Demand : "
    f"{df['average_daily_demand'].mean():.2f}"
)

print(
    f"Maximum Daily Demand : "
    f"{df['maximum_daily_demand'].mean():.2f}"
)

print(
    f"Average Demand Std   : "
    f"{df['demand_std'].mean():.2f}"
)

print(
    f"Average CV           : "
    f"{df['demand_variability'].mean():.2f}"
)

print(
    f"25th Percentile Demand : "
    f"{demand_25:.2f}"
)

print(
    f"75th Percentile Demand : "
    f"{demand_75:.2f}"
)


# ------------------------------------------------------------
# 14. HIGH PRIORITY PRODUCTS
# ------------------------------------------------------------

print("\n==========================================")
print("HIGH PRIORITY INVENTORY ITEMS")
print("==========================================")

high_priority = df[
    df["inventory_priority"].isin(
        [
            "Critical Attention",
            "High Attention"
        ]
    )
].copy()

if len(high_priority) > 0:

    print(
        high_priority[
            [
                "store_nbr",
                "family",
                "average_daily_demand",
                "demand_std",
                "demand_variability",
                "demand_level",
                "demand_variability_class",
                "inventory_priority"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

else:

    print("No high-priority items identified.")


# ------------------------------------------------------------
# 15. SAVE OUTPUT
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
# 16. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample analyzed inventory data:")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "demand_std",
            "demand_variability",
            "demand_variability_class",
            "demand_level",
            "inventory_priority"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 17. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.2 COMPLETED")
print("==========================================")