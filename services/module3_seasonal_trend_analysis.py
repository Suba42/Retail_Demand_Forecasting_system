import pandas as pd
from pathlib import Path

print("Loading seasonal-ready dataset...")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "module3_seasonal_ready.csv"

YEARLY_OUTPUT = BASE_DIR / "module3_yearly_trend.csv"
MONTHLY_OUTPUT = BASE_DIR / "module3_monthly_seasonality.csv"
WEEKDAY_OUTPUT = BASE_DIR / "module3_weekday_pattern.csv"
QUARTER_OUTPUT = BASE_DIR / "module3_quarterly_pattern.csv"
PEAK_OUTPUT = BASE_DIR / "module3_peak_low_periods.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 3.2 - SEASONAL TREND ANALYSIS")
print("==========================================")

# --------------------------------------------------
# 1. YEARLY TREND
# --------------------------------------------------
print("\n--- YEARLY SALES TREND ---")

yearly = (
    df.groupby("Year")["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
)

yearly["YoY_Growth_Percent"] = (
    yearly["Total_Sales"].pct_change() * 100
)

print(yearly)

# --------------------------------------------------
# 2. MONTHLY SEASONALITY
# --------------------------------------------------
print("\n--- MONTHLY SEASONAL PATTERN ---")

monthly = (
    df.groupby("Month")["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
)

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

monthly["Month_Name"] = monthly["Month"].map(month_names)

monthly = monthly[
    [
        "Month",
        "Month_Name",
        "Total_Sales",
        "Average_Sales",
        "Records"
    ]
]

monthly["Seasonal_Index"] = (
    monthly["Average_Sales"] /
    monthly["Average_Sales"].mean()
)

print(monthly)

# --------------------------------------------------
# 3. DAY OF WEEK PATTERN
# --------------------------------------------------
print("\n--- DAY OF WEEK PATTERN ---")

weekday = (
    df.groupby(["DayOfWeek", "Day_Name"])["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
    .sort_values("DayOfWeek")
)

weekday["Relative_Index"] = (
    weekday["Average_Sales"] /
    weekday["Average_Sales"].mean()
)

print(weekday)

# --------------------------------------------------
# 4. QUARTERLY PATTERN
# --------------------------------------------------
print("\n--- QUARTERLY PATTERN ---")

quarter = (
    df.groupby("Quarter")["sales"]
    .agg(
        Total_Sales="sum",
        Average_Sales="mean",
        Records="count"
    )
    .reset_index()
)

quarter["Quarter_Name"] = (
    "Q" + quarter["Quarter"].astype(str)
)

quarter["Relative_Index"] = (
    quarter["Average_Sales"] /
    quarter["Average_Sales"].mean()
)

print(quarter)

# --------------------------------------------------
# 5. PEAK / LOW MONTHS
# --------------------------------------------------
print("\n--- PEAK AND LOW SEASONAL PERIODS ---")

highest_month = monthly.loc[
    monthly["Average_Sales"].idxmax()
]

lowest_month = monthly.loc[
    monthly["Average_Sales"].idxmin()
]

print(
    f"Highest Average Sales Month : "
    f"{highest_month['Month_Name']} "
    f"({highest_month['Average_Sales']:.2f})"
)

print(
    f"Lowest Average Sales Month  : "
    f"{lowest_month['Month_Name']} "
    f"({lowest_month['Average_Sales']:.2f})"
)

highest_day = weekday.loc[
    weekday["Average_Sales"].idxmax()
]

lowest_day = weekday.loc[
    weekday["Average_Sales"].idxmin()
]

print(
    f"Highest Sales Day : "
    f"{highest_day['Day_Name']} "
    f"({highest_day['Average_Sales']:.2f})"
)

print(
    f"Lowest Sales Day  : "
    f"{lowest_day['Day_Name']} "
    f"({lowest_day['Average_Sales']:.2f})"
)

highest_quarter = quarter.loc[
    quarter["Average_Sales"].idxmax()
]

lowest_quarter = quarter.loc[
    quarter["Average_Sales"].idxmin()
]

print(
    f"Highest Sales Quarter : "
    f"{highest_quarter['Quarter_Name']} "
    f"({highest_quarter['Average_Sales']:.2f})"
)

print(
    f"Lowest Sales Quarter  : "
    f"{lowest_quarter['Quarter_Name']} "
    f"({lowest_quarter['Average_Sales']:.2f})"
)

# --------------------------------------------------
# 6. TOP 10 HIGH-DEMAND DATES
# --------------------------------------------------
print("\n--- TOP 10 HIGH-DEMAND DAYS ---")

daily = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

daily = daily.sort_values(
    "sales",
    ascending=False
)

top_10 = daily.head(10)

print(top_10)

# --------------------------------------------------
# 7. TOP 10 LOW-DEMAND DATES
# --------------------------------------------------
print("\n--- TOP 10 LOW-DEMAND DAYS ---")

low_10 = daily.sort_values(
    "sales",
    ascending=True
).head(10)

print(low_10)

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------
print("\nSaving seasonal analysis results...")

yearly.to_csv(
    YEARLY_OUTPUT,
    index=False
)

monthly.to_csv(
    MONTHLY_OUTPUT,
    index=False
)

weekday.to_csv(
    WEEKDAY_OUTPUT,
    index=False
)

quarter.to_csv(
    QUARTER_OUTPUT,
    index=False
)

peak_low = pd.DataFrame({
    "Metric": [
        "Highest Average Sales Month",
        "Lowest Average Sales Month",
        "Highest Sales Day",
        "Lowest Sales Day",
        "Highest Sales Quarter",
        "Lowest Sales Quarter"
    ],
    "Period": [
        highest_month["Month_Name"],
        lowest_month["Month_Name"],
        highest_day["Day_Name"],
        lowest_day["Day_Name"],
        highest_quarter["Quarter_Name"],
        lowest_quarter["Quarter_Name"]
    ],
    "Average_Sales": [
        highest_month["Average_Sales"],
        lowest_month["Average_Sales"],
        highest_day["Average_Sales"],
        lowest_day["Average_Sales"],
        highest_quarter["Average_Sales"],
        lowest_quarter["Average_Sales"]
    ]
})

peak_low.to_csv(
    PEAK_OUTPUT,
    index=False
)

print("\n==========================================")
print("MODULE 3.2 COMPLETED")
print("==========================================")

print("Created files:")
print("✓ module3_yearly_trend.csv")
print("✓ module3_monthly_seasonality.csv")
print("✓ module3_weekday_pattern.csv")
print("✓ module3_quarterly_pattern.csv")
print("✓ module3_peak_low_periods.csv")