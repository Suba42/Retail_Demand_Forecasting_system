import pandas as pd
from pathlib import Path

print("Loading seasonal analysis results...")

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MONTHLY_FILE = BASE_DIR / "module3_monthly_seasonality.csv"
WEEKDAY_FILE = BASE_DIR / "module3_weekday_pattern.csv"
QUARTER_FILE = BASE_DIR / "module3_quarterly_pattern.csv"
HOLIDAY_FILE = BASE_DIR / "module3_holiday_impact.csv"

STORE_FILE = BASE_DIR / "module3_store_seasonality.csv"
PRODUCT_FILE = BASE_DIR / "module3_product_seasonality.csv"

STORE_MONTH_FILE = BASE_DIR / "module3_store_monthly_seasonality.csv"
PRODUCT_MONTH_FILE = BASE_DIR / "module3_product_monthly_seasonality.csv"

PEAK_FILE = BASE_DIR / "module3_store_product_peak_periods.csv"

# Output files
INSIGHTS_FILE = BASE_DIR / "module3_seasonal_insights.csv"
RECOMMENDATIONS_FILE = BASE_DIR / "module3_inventory_recommendations.csv"
SUMMARY_FILE = BASE_DIR / "module3_seasonal_summary.csv"


# ==================================================
# LOAD FILES
# ==================================================
monthly = pd.read_csv(MONTHLY_FILE)
weekday = pd.read_csv(WEEKDAY_FILE)
quarterly = pd.read_csv(QUARTER_FILE)
holiday = pd.read_csv(HOLIDAY_FILE)

store = pd.read_csv(STORE_FILE)
product = pd.read_csv(PRODUCT_FILE)

store_month = pd.read_csv(STORE_MONTH_FILE)
product_month = pd.read_csv(PRODUCT_MONTH_FILE)

peak = pd.read_csv(PEAK_FILE)


print("\n==========================================")
print("MODULE 3.5 - SEASONAL INSIGHTS")
print("==========================================")

# ==================================================
# 1. MONTHLY INSIGHTS
# ==================================================
print("\n--- MONTHLY INSIGHTS ---")

highest_month = monthly.loc[
    monthly["Average_Sales"].idxmax()
]

lowest_month = monthly.loc[
    monthly["Average_Sales"].idxmin()
]

print(
    f"Highest Demand Month : "
    f"{highest_month['Month_Name']} "
    f"({highest_month['Average_Sales']:.2f})"
)

print(
    f"Lowest Demand Month  : "
    f"{lowest_month['Month_Name']} "
    f"({lowest_month['Average_Sales']:.2f})"
)


# ==================================================
# 2. WEEKDAY INSIGHTS
# ==================================================
print("\n--- WEEKDAY INSIGHTS ---")

highest_day = weekday.loc[
    weekday["Average_Sales"].idxmax()
]

lowest_day = weekday.loc[
    weekday["Average_Sales"].idxmin()
]

print(
    f"Highest Demand Day : "
    f"{highest_day['Day_Name']} "
    f"({highest_day['Average_Sales']:.2f})"
)

print(
    f"Lowest Demand Day  : "
    f"{lowest_day['Day_Name']} "
    f"({lowest_day['Average_Sales']:.2f})"
)


# ==================================================
# 3. QUARTER INSIGHTS
# ==================================================
print("\n--- QUARTER INSIGHTS ---")

highest_quarter = quarterly.loc[
    quarterly["Average_Sales"].idxmax()
]

lowest_quarter = quarterly.loc[
    quarterly["Average_Sales"].idxmin()
]

print(
    f"Highest Demand Quarter : "
    f"{highest_quarter['Quarter_Name']} "
    f"({highest_quarter['Average_Sales']:.2f})"
)

print(
    f"Lowest Demand Quarter  : "
    f"{lowest_quarter['Quarter_Name']} "
    f"({lowest_quarter['Average_Sales']:.2f})"
)


# ==================================================
# 4. HOLIDAY INSIGHTS
# ==================================================
print("\n--- HOLIDAY INSIGHTS ---")

holiday_row = holiday[
    holiday["Day_Type"] == "Holiday"
]

normal_row = holiday[
    holiday["Day_Type"] == "Normal Day"
]

holiday_average = (
    holiday_row["Average_Sales"].iloc[0]
)

normal_average = (
    normal_row["Average_Sales"].iloc[0]
)

holiday_change = (
    (holiday_average - normal_average)
    / normal_average
) * 100

print(
    f"Holiday Average Sales : "
    f"{holiday_average:.2f}"
)

print(
    f"Normal Average Sales  : "
    f"{normal_average:.2f}"
)

print(
    f"Holiday Sales Change  : "
    f"{holiday_change:.2f}%"
)


# ==================================================
# 5. TOP STORES
# ==================================================
print("\n--- TOP STORES ---")

top_stores = (
    store
    .sort_values(
        "Average_Sales",
        ascending=False
    )
    .head(10)
)

print(
    top_stores[
        [
            "store_nbr",
            "Average_Sales",
            "Store_Rank"
        ]
    ]
)


# ==================================================
# 6. TOP PRODUCT FAMILIES
# ==================================================
print("\n--- TOP PRODUCT FAMILIES ---")

top_products = (
    product
    .sort_values(
        "Average_Sales",
        ascending=False
    )
    .head(10)
)

print(
    top_products[
        [
            "family",
            "Average_Sales",
            "Product_Rank"
        ]
    ]
)


# ==================================================
# 7. STRONGEST STORE SEASONALITY
# ==================================================
print("\n--- STRONGEST STORE SEASONAL EFFECTS ---")

strong_store = (
    store_month
    .sort_values(
        "Seasonal_Index",
        ascending=False
    )
    .head(10)
)

print(
    strong_store[
        [
            "store_nbr",
            "Month",
            "Average_Sales",
            "Seasonal_Index"
        ]
    ]
)


# ==================================================
# 8. STRONGEST PRODUCT SEASONALITY
# ==================================================
print("\n--- STRONGEST PRODUCT SEASONAL EFFECTS ---")

strong_product = (
    product_month
    .sort_values(
        "Seasonal_Index",
        ascending=False
    )
    .head(10)
)

print(
    strong_product[
        [
            "family",
            "Month",
            "Average_Sales",
            "Seasonal_Index"
        ]
    ]
)


# ==================================================
# 9. CREATE BUSINESS INSIGHTS
# ==================================================
print("\n--- CREATING BUSINESS INSIGHTS ---")

insights = []

# Monthly peak
insights.append({
    "Insight_Type": "Monthly Seasonality",
    "Entity": highest_month["Month_Name"],
    "Metric": "Average Sales",
    "Value": highest_month["Average_Sales"],
    "Interpretation":
        "This month has the highest average demand.",
    "Business_Action":
        "Increase inventory preparation before this month."
})

# Monthly low
insights.append({
    "Insight_Type": "Monthly Seasonality",
    "Entity": lowest_month["Month_Name"],
    "Metric": "Average Sales",
    "Value": lowest_month["Average_Sales"],
    "Interpretation":
        "This month has the lowest average demand.",
    "Business_Action":
        "Avoid excessive inventory accumulation."
})

# Weekday peak
insights.append({
    "Insight_Type": "Weekly Seasonality",
    "Entity": highest_day["Day_Name"],
    "Metric": "Average Sales",
    "Value": highest_day["Average_Sales"],
    "Interpretation":
        "This day has the highest average demand.",
    "Business_Action":
        "Ensure sufficient stock before high-demand days."
})

# Quarter peak
insights.append({
    "Insight_Type": "Quarterly Seasonality",
    "Entity": highest_quarter["Quarter_Name"],
    "Metric": "Average Sales",
    "Value": highest_quarter["Average_Sales"],
    "Interpretation":
        "This quarter has the strongest demand.",
    "Business_Action":
        "Prepare inventory and replenishment capacity in advance."
})

# Holiday
insights.append({
    "Insight_Type": "Holiday Impact",
    "Entity": "Holiday",
    "Metric": "Sales Change %",
    "Value": holiday_change,
    "Interpretation":
        "Holiday periods show higher average demand.",
    "Business_Action":
        "Increase stock availability before important holidays."
})

# Top store
top_store = top_stores.iloc[0]

insights.append({
    "Insight_Type": "Store Demand",
    "Entity": str(top_store["store_nbr"]),
    "Metric": "Average Sales",
    "Value": top_store["Average_Sales"],
    "Interpretation":
        "This store has the highest average sales.",
    "Business_Action":
        "Prioritize inventory availability for this store."
})

# Top product
top_product = top_products.iloc[0]

insights.append({
    "Insight_Type": "Product Demand",
    "Entity": top_product["family"],
    "Metric": "Average Sales",
    "Value": top_product["Average_Sales"],
    "Interpretation":
        "This product family has the highest average demand.",
    "Business_Action":
        "Maintain higher safety stock for this product family."
})


insights_df = pd.DataFrame(insights)


# ==================================================
# 10. INVENTORY RECOMMENDATIONS
# ==================================================
print("\n--- CREATING INVENTORY RECOMMENDATIONS ---")

recommendations = []

# Peak month
recommendations.append({
    "Recommendation_Type": "Seasonal Stock Planning",
    "Target": highest_month["Month_Name"],
    "Priority": "High",
    "Recommendation":
        "Increase inventory preparation before the peak-demand month.",
    "Reason":
        f"Average sales = {highest_month['Average_Sales']:.2f}"
})

# Low month
recommendations.append({
    "Recommendation_Type": "Inventory Reduction",
    "Target": lowest_month["Month_Name"],
    "Priority": "Medium",
    "Recommendation":
        "Reduce excess inventory during the low-demand month.",
    "Reason":
        f"Average sales = {lowest_month['Average_Sales']:.2f}"
})

# Peak day
recommendations.append({
    "Recommendation_Type": "Weekly Replenishment",
    "Target": highest_day["Day_Name"],
    "Priority": "High",
    "Recommendation":
        "Complete replenishment before the highest-demand weekday.",
    "Reason":
        f"Average sales = {highest_day['Average_Sales']:.2f}"
})

# Holiday
recommendations.append({
    "Recommendation_Type": "Holiday Preparation",
    "Target": "Holiday Periods",
    "Priority": "High",
    "Recommendation":
        "Increase stock levels before major holiday periods.",
    "Reason":
        f"Average sales increase = {holiday_change:.2f}%"
})

# Top store
recommendations.append({
    "Recommendation_Type": "Store Allocation",
    "Target": f"Store {top_store['store_nbr']}",
    "Priority": "High",
    "Recommendation":
        "Prioritize inventory allocation for the highest-demand store.",
    "Reason":
        f"Average sales = {top_store['Average_Sales']:.2f}"
})

# Top product
recommendations.append({
    "Recommendation_Type": "Product Stock",
    "Target": top_product["family"],
    "Priority": "High",
    "Recommendation":
        "Maintain stronger stock availability for the highest-demand product family.",
    "Reason":
        f"Average sales = {top_product['Average_Sales']:.2f}"
})

recommendations_df = pd.DataFrame(
    recommendations
)


# ==================================================
# 11. SUMMARY
# ==================================================
summary = pd.DataFrame([
    {
        "Metric": "Peak Month",
        "Value": highest_month["Month_Name"],
        "Sales": highest_month["Average_Sales"]
    },
    {
        "Metric": "Lowest Month",
        "Value": lowest_month["Month_Name"],
        "Sales": lowest_month["Average_Sales"]
    },
    {
        "Metric": "Peak Weekday",
        "Value": highest_day["Day_Name"],
        "Sales": highest_day["Average_Sales"]
    },
    {
        "Metric": "Lowest Weekday",
        "Value": lowest_day["Day_Name"],
        "Sales": lowest_day["Average_Sales"]
    },
    {
        "Metric": "Peak Quarter",
        "Value": highest_quarter["Quarter_Name"],
        "Sales": highest_quarter["Average_Sales"]
    },
    {
        "Metric": "Holiday Sales Change",
        "Value": f"{holiday_change:.2f}%",
        "Sales": holiday_average
    },
    {
        "Metric": "Top Store",
        "Value": top_store["store_nbr"],
        "Sales": top_store["Average_Sales"]
    },
    {
        "Metric": "Top Product Family",
        "Value": top_product["family"],
        "Sales": top_product["Average_Sales"]
    }
])


# ==================================================
# 12. SAVE OUTPUTS
# ==================================================
print("\nSaving seasonal insights...")

insights_df.to_csv(
    INSIGHTS_FILE,
    index=False
)

recommendations_df.to_csv(
    RECOMMENDATIONS_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ==================================================
# FINAL OUTPUT
# ==================================================
print("\n--- FINAL SEASONAL SUMMARY ---")

print(summary.to_string(index=False))

print("\n--- INVENTORY RECOMMENDATIONS ---")

print(
    recommendations_df.to_string(
        index=False
    )
)

print("\n==========================================")
print("MODULE 3.5 COMPLETED")
print("==========================================")

print("Created files:")
print("✓ module3_seasonal_insights.csv")
print("✓ module3_inventory_recommendations.csv")
print("✓ module3_seasonal_summary.csv")

print("\nSeasonal analysis and inventory recommendations completed.")