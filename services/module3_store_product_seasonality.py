import pandas as pd
from pathlib import Path

print("Loading seasonal-ready dataset...")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "module3_seasonal_ready.csv"

STORE_OUTPUT = BASE_DIR / "module3_store_seasonality.csv"
FAMILY_OUTPUT = BASE_DIR / "module3_product_seasonality.csv"
STORE_MONTH_OUTPUT = BASE_DIR / "module3_store_monthly_seasonality.csv"
FAMILY_MONTH_OUTPUT = BASE_DIR / "module3_product_monthly_seasonality.csv"
PEAK_OUTPUT = BASE_DIR / "module3_store_product_peak_periods.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 3.4 - STORE & PRODUCT SEASONALITY")
print("==========================================")

# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------
print("\n--- REQUIRED COLUMN CHECK ---")

required_columns = [
    "date",
    "sales",
    "store_nbr",
    "family",
    "Month",
    "Quarter"
]

for col in required_columns:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        raise ValueError(f"Missing required column: {col}")

# --------------------------------------------------
# 1. STORE-LEVEL SALES ANALYSIS
# --------------------------------------------------
print("\n--- STORE-LEVEL SEASONAL ANALYSIS ---")

store_summary = (
    df.groupby("store_nbr")["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
)

store_summary = store_summary.sort_values(
    "Average_Sales",
    ascending=False
)

store_summary["Store_Rank"] = range(
    1,
    len(store_summary) + 1
)

print("\nTop 10 Stores by Average Sales:")
print(store_summary.head(10))

print("\nBottom 10 Stores by Average Sales:")
print(
    store_summary
    .sort_values("Average_Sales")
    .head(10)
)

# --------------------------------------------------
# 2. PRODUCT FAMILY ANALYSIS
# --------------------------------------------------
print("\n--- PRODUCT FAMILY SEASONAL ANALYSIS ---")

family_summary = (
    df.groupby("family")["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
)

family_summary = family_summary.sort_values(
    "Average_Sales",
    ascending=False
)

family_summary["Product_Rank"] = range(
    1,
    len(family_summary) + 1
)

print("\nTop 10 Product Families:")
print(family_summary.head(10))

print("\nBottom 10 Product Families:")
print(
    family_summary
    .sort_values("Average_Sales")
    .head(10)
)

# --------------------------------------------------
# 3. STORE × MONTH
# --------------------------------------------------
print("\n--- STORE MONTHLY SEASONALITY ---")

store_monthly = (
    df.groupby(
        ["store_nbr", "Month"]
    )["sales"]
    .mean()
    .reset_index(
        name="Average_Sales"
    )
)

# Add each store's overall average
store_overall = (
    df.groupby("store_nbr")["sales"]
    .mean()
    .reset_index(
        name="Store_Average_Sales"
    )
)

store_monthly = store_monthly.merge(
    store_overall,
    on="store_nbr",
    how="left"
)

store_monthly["Seasonal_Index"] = (
    store_monthly["Average_Sales"]
    / store_monthly["Store_Average_Sales"]
)

print("\nSample Store-Month Analysis:")
print(store_monthly.head(15))

# --------------------------------------------------
# 4. PRODUCT × MONTH
# --------------------------------------------------
print("\n--- PRODUCT MONTHLY SEASONALITY ---")

family_monthly = (
    df.groupby(
        ["family", "Month"]
    )["sales"]
    .mean()
    .reset_index(
        name="Average_Sales"
    )
)

family_overall = (
    df.groupby("family")["sales"]
    .mean()
    .reset_index(
        name="Product_Average_Sales"
    )
)

family_monthly = family_monthly.merge(
    family_overall,
    on="family",
    how="left"
)

family_monthly["Seasonal_Index"] = (
    family_monthly["Average_Sales"]
    / family_monthly["Product_Average_Sales"]
)

print("\nSample Product-Month Analysis:")
print(family_monthly.head(15))

# --------------------------------------------------
# 5. STORE × QUARTER
# --------------------------------------------------
print("\n--- STORE QUARTER PATTERN ---")

store_quarter = (
    df.groupby(
        ["store_nbr", "Quarter"]
    )["sales"]
    .mean()
    .reset_index(
        name="Average_Sales"
    )
)

print(store_quarter.head(15))

# --------------------------------------------------
# 6. PRODUCT × QUARTER
# --------------------------------------------------
print("\n--- PRODUCT QUARTER PATTERN ---")

family_quarter = (
    df.groupby(
        ["family", "Quarter"]
    )["sales"]
    .mean()
    .reset_index(
        name="Average_Sales"
    )
)

print(family_quarter.head(15))

# --------------------------------------------------
# 7. PEAK MONTH FOR EACH STORE
# --------------------------------------------------
print("\n--- PEAK MONTH BY STORE ---")

store_peak = (
    store_monthly
    .sort_values(
        ["store_nbr", "Seasonal_Index"],
        ascending=[True, False]
    )
    .groupby("store_nbr")
    .first()
    .reset_index()
)

store_peak = store_peak[
    [
        "store_nbr",
        "Month",
        "Average_Sales",
        "Seasonal_Index"
    ]
]

print(store_peak.head(10))

# --------------------------------------------------
# 8. PEAK MONTH FOR EACH PRODUCT
# --------------------------------------------------
print("\n--- PEAK MONTH BY PRODUCT FAMILY ---")

family_peak = (
    family_monthly
    .sort_values(
        ["family", "Seasonal_Index"],
        ascending=[True, False]
    )
    .groupby("family")
    .first()
    .reset_index()
)

family_peak = family_peak[
    [
        "family",
        "Month",
        "Average_Sales",
        "Seasonal_Index"
    ]
]

print(family_peak.head(10))

# --------------------------------------------------
# 9. STRONGEST STORE SEASONALITY
# --------------------------------------------------
print("\n--- STRONGEST STORE SEASONAL EFFECTS ---")

strong_store_seasonality = (
    store_monthly
    .sort_values(
        "Seasonal_Index",
        ascending=False
    )
    .head(10)
)

print(strong_store_seasonality)

# --------------------------------------------------
# 10. STRONGEST PRODUCT SEASONALITY
# --------------------------------------------------
print("\n--- STRONGEST PRODUCT SEASONAL EFFECTS ---")

strong_family_seasonality = (
    family_monthly
    .sort_values(
        "Seasonal_Index",
        ascending=False
    )
    .head(10)
)

print(strong_family_seasonality)

# --------------------------------------------------
# 11. CREATE PEAK SUMMARY
# --------------------------------------------------
print("\n--- CREATING PEAK PERIOD SUMMARY ---")

store_peak_summary = store_peak.copy()
store_peak_summary["Entity_Type"] = "Store"
store_peak_summary["Entity"] = (
    store_peak_summary["store_nbr"]
    .astype(str)
)

store_peak_summary["Peak_Month"] = (
    store_peak_summary["Month"]
)

store_peak_summary["Peak_Average_Sales"] = (
    store_peak_summary["Average_Sales"]
)

store_peak_summary["Peak_Seasonal_Index"] = (
    store_peak_summary["Seasonal_Index"]
)

store_peak_summary = store_peak_summary[
    [
        "Entity_Type",
        "Entity",
        "Peak_Month",
        "Peak_Average_Sales",
        "Peak_Seasonal_Index"
    ]
]

family_peak_summary = family_peak.copy()
family_peak_summary["Entity_Type"] = "Product_Family"
family_peak_summary["Entity"] = (
    family_peak_summary["family"]
)

family_peak_summary["Peak_Month"] = (
    family_peak_summary["Month"]
)

family_peak_summary["Peak_Average_Sales"] = (
    family_peak_summary["Average_Sales"]
)

family_peak_summary["Peak_Seasonal_Index"] = (
    family_peak_summary["Seasonal_Index"]
)

family_peak_summary = family_peak_summary[
    [
        "Entity_Type",
        "Entity",
        "Peak_Month",
        "Peak_Average_Sales",
        "Peak_Seasonal_Index"
    ]
]

peak_summary = pd.concat(
    [
        store_peak_summary,
        family_peak_summary
    ],
    ignore_index=True
)

# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------
print("\nSaving store and product seasonality results...")

store_summary.to_csv(
    STORE_OUTPUT,
    index=False
)

family_summary.to_csv(
    FAMILY_OUTPUT,
    index=False
)

store_monthly.to_csv(
    STORE_MONTH_OUTPUT,
    index=False
)

family_monthly.to_csv(
    FAMILY_MONTH_OUTPUT,
    index=False
)

peak_summary.to_csv(
    PEAK_OUTPUT,
    index=False
)

# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------
print("\n--- OUTPUT VALIDATION ---")

print(
    f"Store records       : {len(store_summary)}"
)

print(
    f"Product families    : {len(family_summary)}"
)

print(
    f"Store-month records : {len(store_monthly)}"
)

print(
    f"Product-month records : {len(family_monthly)}"
)

print(
    f"Peak summary records: {len(peak_summary)}"
)

print("\n==========================================")
print("MODULE 3.4 COMPLETED")
print("==========================================")

print("Created files:")
print("✓ module3_store_seasonality.csv")
print("✓ module3_product_seasonality.csv")
print("✓ module3_store_monthly_seasonality.csv")
print("✓ module3_product_monthly_seasonality.csv")
print("✓ module3_store_product_peak_periods.csv")