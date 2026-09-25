import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 4.7 - INVENTORY OPTIMIZATION SUMMARY
# Retail Demand Forecasting & Inventory Optimization System
# ============================================================

print("\n==========================================")
print("MODULE 4.7 - INVENTORY OPTIMIZATION SUMMARY")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "Dataset/04_Inventory_Optimization/"
    "inventory_order_recommendations.csv"
)

OUTPUT_DIR = "Dataset/04_Inventory_Optimization"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inventory_optimization_final.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODULE 4.6 OUTPUT
# ------------------------------------------------------------

print("Loading Module 4.6 inventory recommendations...")

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
    "safety_stock_units",
    "reorder_point_units",
    "planning_horizon_days",
    "planning_horizon_demand",
    "recommended_order_units",
    "order_priority",
    "demand_risk",
    "inventory_recommendation"
]

missing_columns = [
    col
    for col in required_columns
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
    "safety_stock_units",
    "reorder_point_units",
    "planning_horizon_days",
    "planning_horizon_demand",
    "recommended_order_units"
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
# 5. INVENTORY COVERAGE
# ------------------------------------------------------------
#
# Inventory Coverage Days =
# Reorder Point / Average Daily Demand
#
# This shows approximately how many days of demand
# the reorder point represents.
# ------------------------------------------------------------

df["inventory_coverage_days"] = np.where(
    df["average_daily_demand"] > 0,
    df["reorder_point_units"]
    / df["average_daily_demand"],
    0
)


# ------------------------------------------------------------
# 6. SAFETY STOCK RATIO
# ------------------------------------------------------------
#
# Safety Stock Ratio =
# Safety Stock / Reorder Point
# ------------------------------------------------------------

df["safety_stock_ratio"] = np.where(
    df["reorder_point_units"] > 0,
    df["safety_stock_units"]
    / df["reorder_point_units"],
    0
)


# ------------------------------------------------------------
# 7. INVENTORY RISK SCORE
# ------------------------------------------------------------
#
# Higher demand variability + higher safety stock ratio
# indicates greater inventory planning risk.
# ------------------------------------------------------------

df["coefficient_of_variation"] = np.where(
    df["average_daily_demand"] > 0,
    df["demand_std"]
    / df["average_daily_demand"],
    0
)


def calculate_risk_score(row):

    if row["average_daily_demand"] <= 0:
        return 0

    cv = row["coefficient_of_variation"]

    if cv >= 1:
        return 3

    elif cv >= 0.5:
        return 2

    else:
        return 1


df["inventory_risk_score"] = (
    df.apply(
        calculate_risk_score,
        axis=1
    )
)


# ------------------------------------------------------------
# 8. FINAL INVENTORY ACTION
# ------------------------------------------------------------

def determine_final_action(row):

    if row["average_daily_demand"] <= 0:
        return "No Action Required"

    if row["order_priority"] == "Critical Priority":
        return "Immediate Replenishment Planning"

    elif row["order_priority"] == "High Priority":
        return "High Priority Replenishment"

    elif row["order_priority"] == "Medium Priority":
        return "Regular Replenishment"

    else:
        return "Monitor Inventory"


df["final_inventory_action"] = (
    df.apply(
        determine_final_action,
        axis=1
    )
)


# ------------------------------------------------------------
# 9. BUSINESS DECISION CATEGORY
# ------------------------------------------------------------

def determine_business_category(row):

    if row["average_daily_demand"] <= 0:
        return "Inactive Product"

    if (
        row["inventory_risk_score"] == 3
        and row["order_priority"] in [
            "Critical Priority",
            "High Priority"
        ]
    ):
        return "High Risk - High Demand"

    elif row["inventory_risk_score"] == 3:
        return "High Demand Variability"

    elif row["order_priority"] in [
        "Critical Priority",
        "High Priority"
    ]:
        return "High Replenishment Requirement"

    elif row["order_priority"] == "Medium Priority":
        return "Moderate Replenishment Requirement"

    else:
        return "Low Replenishment Requirement"


df["business_decision_category"] = (
    df.apply(
        determine_business_category,
        axis=1
    )
)


# ------------------------------------------------------------
# 10. ROUND DECIMAL VALUES
# ------------------------------------------------------------

decimal_columns = [
    "average_daily_demand",
    "demand_std",
    "planning_horizon_demand",
    "inventory_coverage_days",
    "safety_stock_ratio",
    "coefficient_of_variation"
]

for col in decimal_columns:

    df[col] = (
        df[col]
        .round(2)
    )


# ------------------------------------------------------------
# 11. SORT FINAL DATA
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
        "recommended_order_units"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 12. FINAL DATASET COLUMNS
# ------------------------------------------------------------

final_columns = [
    "store_nbr",
    "family",
    "average_daily_demand",
    "demand_std",
    "coefficient_of_variation",
    "demand_risk",
    "lead_time_days",
    "planning_horizon_days",
    "planning_horizon_demand",
    "safety_stock_units",
    "safety_stock_ratio",
    "reorder_point_units",
    "inventory_coverage_days",
    "recommended_order_units",
    "order_priority",
    "inventory_risk_score",
    "inventory_recommendation",
    "final_inventory_action",
    "business_decision_category"
]

df_final = df[final_columns].copy()


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("FINAL INVENTORY OPTIMIZATION SUMMARY")
print("==========================================")

print(
    f"Store × Product combinations : "
    f"{len(df_final):,}"
)

print(
    f"Average Daily Demand : "
    f"{df_final['average_daily_demand'].mean():.2f}"
)

print(
    f"Average Safety Stock : "
    f"{df_final['safety_stock_units'].mean():.2f}"
)

print(
    f"Average Reorder Point : "
    f"{df_final['reorder_point_units'].mean():.2f}"
)

print(
    f"Total Recommended Order : "
    f"{df_final['recommended_order_units'].sum():,}"
)


# ------------------------------------------------------------
# 14. FINAL ACTION DISTRIBUTION
# ------------------------------------------------------------

print("\nFinal Inventory Action Distribution:")

print(
    df_final[
        "final_inventory_action"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 15. BUSINESS CATEGORY DISTRIBUTION
# ------------------------------------------------------------

print("\nBusiness Decision Category Distribution:")

print(
    df_final[
        "business_decision_category"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 16. TOP CRITICAL ITEMS
# ------------------------------------------------------------

print("\n==========================================")
print("TOP CRITICAL INVENTORY ITEMS")
print("==========================================")

critical_items = df_final[
    df_final["order_priority"]
    == "Critical Priority"
]

print(
    critical_items[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "safety_stock_units",
            "reorder_point_units",
            "recommended_order_units",
            "demand_risk",
            "final_inventory_action"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 17. SAVE FINAL OUTPUT
# ------------------------------------------------------------

df_final.to_csv(
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
# 18. SAMPLE FINAL DATA
# ------------------------------------------------------------

print("\nSample final inventory optimization data:")

print(
    df_final
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 19. IMPORTANT SYSTEM NOTE
# ------------------------------------------------------------

print("\n==========================================")
print("SYSTEM NOTE")
print("==========================================")

print(
    "Current stock data is not available."
)

print(
    "Therefore, recommended order quantities are "
    "gross planning requirements."
)

print(
    "Actual net order quantity requires current "
    "inventory/stock data."
)


# ------------------------------------------------------------
# 20. COMPLETION
# ------------------------------------------------------------

print("\n==========================================")
print("MODULE 4.7 COMPLETED")
print("==========================================")