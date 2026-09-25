import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.6 - INVENTORY ORDER RECOMMENDATION
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.6 - INVENTORY ORDER RECOMMENDATION")
print("==========================================\n")


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

PLANNING_HORIZON_DAYS = 7


# ------------------------------------------------------------
# 2. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_reorder_point.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_order_recommendations.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD MODULE 4.4 DATA
# ------------------------------------------------------------

print("Loading Module 4.4 reorder point dataset...")

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
    "lead_time_days",
    "safety_stock_units",
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
# 5. NUMERIC CLEANING
# ------------------------------------------------------------

numeric_columns = [
    "average_daily_demand",
    "demand_std",
    "lead_time_days",
    "safety_stock_units",
    "reorder_point_units"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[numeric_columns] = (
    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------------------------
# 6. PLANNING HORIZON DEMAND
# ------------------------------------------------------------
#
# Planning Horizon Demand =
# Average Daily Demand × Planning Horizon
#
# Current stock is unavailable, so this represents the
# expected demand requirement for the planning period.
# ------------------------------------------------------------

df["planning_horizon_days"] = (
    PLANNING_HORIZON_DAYS
)

df["planning_horizon_demand"] = (
    df["average_daily_demand"]
    * PLANNING_HORIZON_DAYS
)


# ------------------------------------------------------------
# 7. TARGET INVENTORY REQUIREMENT
# ------------------------------------------------------------
#
# Target Requirement =
# Planning Horizon Demand + Safety Stock
#
# This provides a conservative inventory requirement
# while actual current stock is unavailable.
# ------------------------------------------------------------

df["target_inventory_requirement"] = (
    df["planning_horizon_demand"]
    + df["safety_stock_units"]
)


# ------------------------------------------------------------
# 8. RECOMMENDED ORDER QUANTITY
# ------------------------------------------------------------
#
# Since current inventory is NOT available:
#
# Recommended Order Quantity =
# Target Inventory Requirement
#
# This is explicitly a GROSS planning recommendation.
#
# Once current stock becomes available:
#
# Net Order Quantity =
# Target Inventory Requirement - Current Stock
# ------------------------------------------------------------

df["recommended_order_quantity"] = (
    df["target_inventory_requirement"]
)


# ------------------------------------------------------------
# 9. ROUND UP TO WHOLE UNITS
# ------------------------------------------------------------

df["planning_demand_units"] = (
    np.ceil(
        df["planning_horizon_demand"]
    ).astype(int)
)

df["target_inventory_units"] = (
    np.ceil(
        df["target_inventory_requirement"]
    ).astype(int)
)

df["recommended_order_units"] = (
    np.ceil(
        df["recommended_order_quantity"]
    ).astype(int)
)


# ------------------------------------------------------------
# 10. ORDER PRIORITY
# ------------------------------------------------------------

def determine_order_priority(row):

    if row["average_daily_demand"] <= 0:
        return "No Demand"

    if row["recommended_order_units"] <= 100:
        return "Low Priority"

    elif row["recommended_order_units"] <= 1000:
        return "Medium Priority"

    elif row["recommended_order_units"] <= 5000:
        return "High Priority"

    else:
        return "Critical Priority"


df["order_priority"] = (
    df.apply(
        determine_order_priority,
        axis=1
    )
)


# ------------------------------------------------------------
# 11. INVENTORY RECOMMENDATION
# ------------------------------------------------------------

def generate_recommendation(row):

    if row["average_daily_demand"] <= 0:
        return "No replenishment required"

    if row["recommended_order_units"] <= 100:
        return "Monitor and replenish as needed"

    elif row["recommended_order_units"] <= 1000:
        return "Plan regular replenishment"

    elif row["recommended_order_units"] <= 5000:
        return "Prepare high-volume replenishment"

    else:
        return "Prioritize replenishment planning"


df["inventory_recommendation"] = (
    df.apply(
        generate_recommendation,
        axis=1
    )
)


# ------------------------------------------------------------
# 12. DEMAND RISK
# ------------------------------------------------------------

def determine_demand_risk(row):

    variability = row["demand_std"]

    average = row["average_daily_demand"]

    if average <= 0:
        return "No Demand"

    cv = variability / average

    if cv >= 1:
        return "High Demand Risk"

    elif cv >= 0.5:
        return "Medium Demand Risk"

    else:
        return "Low Demand Risk"


df["demand_risk"] = (
    df.apply(
        determine_demand_risk,
        axis=1
    )
)


# ------------------------------------------------------------
# 13. ROUND DECIMAL COLUMNS
# ------------------------------------------------------------

decimal_columns = [
    "average_daily_demand",
    "planning_horizon_demand",
    "target_inventory_requirement"
]

for col in decimal_columns:

    df[col] = (
        df[col]
        .round(2)
    )


# ------------------------------------------------------------
# 14. SORT BY PRIORITY
# ------------------------------------------------------------

priority_order = {
    "Critical Priority": 1,
    "High Priority": 2,
    "Medium Priority": 3,
    "Low Priority": 4,
    "No Demand": 5
}

df["priority_rank"] = (
    df["order_priority"]
    .map(priority_order)
    .fillna(99)
)

df = df.sort_values(
    by=[
        "priority_rank",
        "recommended_order_units",
        "store_nbr",
        "family"
    ],
    ascending=[
        True,
        False,
        True,
        True
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 15. SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("ORDER PLANNING CONFIGURATION")
print("==========================================")

print(
    f"Planning Horizon : "
    f"{PLANNING_HORIZON_DAYS} days"
)

print(
    "Current Stock    : "
    "Not Available"
)

print(
    "Recommendation   : "
    "Gross Planning Requirement"
)


# ------------------------------------------------------------
# 16. DEMAND SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("PLANNING DEMAND SUMMARY")
print("==========================================")

print(
    f"Average Daily Demand : "
    f"{df['average_daily_demand'].mean():.2f}"
)

print(
    f"Average Planning Demand : "
    f"{df['planning_horizon_demand'].mean():.2f}"
)

print(
    f"Average Safety Stock : "
    f"{df['safety_stock_units'].mean():.2f}"
)


# ------------------------------------------------------------
# 17. ORDER SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("ORDER RECOMMENDATION SUMMARY")
print("==========================================")

print(
    f"Average Recommended Order : "
    f"{df['recommended_order_units'].mean():.2f}"
)

print(
    f"Maximum Recommended Order : "
    f"{df['recommended_order_units'].max():,}"
)

print(
    f"Total Recommended Order : "
    f"{df['recommended_order_units'].sum():,}"
)


# ------------------------------------------------------------
# 18. PRIORITY DISTRIBUTION
# ------------------------------------------------------------

print("\nOrder Priority Distribution:")

print(
    df["order_priority"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 19. DEMAND RISK DISTRIBUTION
# ------------------------------------------------------------

print("\nDemand Risk Distribution:")

print(
    df["demand_risk"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 20. TOP RECOMMENDATIONS
# ------------------------------------------------------------

print("\n==========================================")
print("TOP INVENTORY RECOMMENDATIONS")
print("==========================================")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "planning_demand_units",
            "safety_stock_units",
            "reorder_point_units",
            "recommended_order_units",
            "order_priority",
            "demand_risk"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 21. SAVE OUTPUT
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
# 22. SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample inventory recommendation data:")

print(
    df[
        [
            "store_nbr",
            "family",
            "planning_demand_units",
            "safety_stock_units",
            "recommended_order_units",
            "order_priority",
            "inventory_recommendation"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 23. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n==========================================")
print("INVENTORY DATA NOTE")
print("==========================================")

print(
    "Current stock data is not available."
)

print(
    "Therefore, recommended quantity represents "
    "a GROSS inventory planning requirement."
)

print(
    "Once actual current stock is uploaded, "
    "NET ORDER QUANTITY can be calculated."
)


# ------------------------------------------------------------
# 24. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.6 COMPLETED")
print("==========================================")