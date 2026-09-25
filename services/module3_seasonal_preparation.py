import pandas as pd
from pathlib import Path

print("Loading model-ready dataset...")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "Dataset" / "01_Demand_Forecasting" / "demand_forecasting_train_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "module3_seasonal_ready.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 3.1 - SEASONAL ANALYSIS PREPARATION")
print("==========================================")

# --------------------------------------------------
# REQUIRED COLUMN CHECK
# --------------------------------------------------
required_columns = [
    "date",
    "sales",
    "store_nbr",
    "family"
]

print("\n--- REQUIRED COLUMN CHECK ---")

for col in required_columns:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        print(f"✗ {col} - MISSING")

# --------------------------------------------------
# DATE CONVERSION
# --------------------------------------------------
print("\n--- DATE CONVERSION ---")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

invalid_dates = df["date"].isna().sum()

print(f"Invalid dates: {invalid_dates}")

df = df.dropna(subset=["date"])

# --------------------------------------------------
# SORT DATA
# --------------------------------------------------
df = df.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

print("\nData sorted by date, store and product family.")

# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------
print("\n--- CREATING SEASONAL FEATURES ---")

df["Year"] = df["date"].dt.year
df["Month"] = df["date"].dt.month
df["Month_Name"] = df["date"].dt.month_name()

df["DayOfWeek"] = df["date"].dt.dayofweek
df["Day_Name"] = df["date"].dt.day_name()

df["WeekOfYear"] = df["date"].dt.isocalendar().week.astype(int)
df["DayOfMonth"] = df["date"].dt.day
df["Quarter"] = df["date"].dt.quarter

df["Weekend_Flag"] = (
    df["DayOfWeek"] >= 5
).astype(int)

# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------
print("\n--- DATE RANGE ---")

print(f"Start Date : {df['date'].min()}")
print(f"End Date   : {df['date'].max()}")

print(f"Number of Days : {df['date'].nunique()}")

# --------------------------------------------------
# COVERAGE
# --------------------------------------------------
print("\n--- DATA COVERAGE ---")

print(f"Unique Stores   : {df['store_nbr'].nunique()}")
print(f"Unique Families : {df['family'].nunique()}")

if "city" in df.columns:
    print(f"Unique Cities   : {df['city'].nunique()}")

if "state" in df.columns:
    print(f"Unique States   : {df['state'].nunique()}")

# --------------------------------------------------
# SALES VALIDATION
# --------------------------------------------------
print("\n--- SALES VALIDATION ---")

print(f"Minimum Sales : {df['sales'].min()}")
print(f"Maximum Sales : {df['sales'].max()}")
print(f"Average Sales : {df['sales'].mean():.4f}")

negative_sales = (df["sales"] < 0).sum()

print(f"Negative Sales Records : {negative_sales}")

# --------------------------------------------------
# MISSING VALUE CHECK
# --------------------------------------------------
print("\n--- MISSING VALUE CHECK ---")

check_columns = [
    "date",
    "sales",
    "Year",
    "Month",
    "DayOfWeek",
    "WeekOfYear",
    "Quarter"
]

print(df[check_columns].isnull().sum())

# --------------------------------------------------
# YEARLY SUMMARY
# --------------------------------------------------
print("\n--- YEARLY SALES SUMMARY ---")

yearly_summary = (
    df.groupby("Year")["sales"]
    .agg(["sum", "mean", "count"])
    .reset_index()
)

print(yearly_summary)

# --------------------------------------------------
# MONTHLY SUMMARY
# --------------------------------------------------
print("\n--- MONTHLY SALES SUMMARY ---")

monthly_summary = (
    df.groupby(["Year", "Month"])["sales"]
    .agg(["sum", "mean", "count"])
    .reset_index()
)

print(monthly_summary.head(12))

# --------------------------------------------------
# DAY OF WEEK SUMMARY
# --------------------------------------------------
print("\n--- DAY OF WEEK SALES SUMMARY ---")

weekday_summary = (
    df.groupby(["DayOfWeek", "Day_Name"])["sales"]
    .agg(["sum", "mean", "count"])
    .reset_index()
    .sort_values("DayOfWeek")
)

print(weekday_summary)

# --------------------------------------------------
# QUARTER SUMMARY
# --------------------------------------------------
print("\n--- QUARTER SALES SUMMARY ---")

quarter_summary = (
    df.groupby("Quarter")["sales"]
    .agg(["sum", "mean", "count"])
    .reset_index()
)

print(quarter_summary)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
print("\nSaving seasonal-ready dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("MODULE 3.1 COMPLETED")
print("==========================================")

print(f"Output File : {OUTPUT_FILE.name}")
print(f"Rows        : {len(df)}")
print(f"Columns     : {len(df.columns)}")