import pandas as pd
import numpy as np

# ==========================================
# MODULE 2.1 - FORECASTING DATASET PREPARATION
# ==========================================

INPUT_FILE = "module5_model_ready.csv"
OUTPUT_FILE = "module2_forecasting_ready.csv"

print("Loading model-ready dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 2.1 - FORECASTING DATA PREPARATION")
print("==========================================")

# ------------------------------------------
# 1. Check required columns
# ------------------------------------------

required_columns = [
    "date",
    "sales",
    "store_nbr",
    "family"
]

print("\n--- REQUIRED COLUMN CHECK ---")

for column in required_columns:
    if column in df.columns:
        print(f"✓ {column}")
    else:
        print(f"✗ {column} MISSING")

# ------------------------------------------
# 2. Convert date
# ------------------------------------------

print("\n--- DATE CONVERSION ---")

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

invalid_dates = df["date"].isna().sum()

print(f"Invalid dates: {invalid_dates}")

# ------------------------------------------
# 3. Sort data
# ------------------------------------------

df = df.sort_values(
    by=["date", "store_nbr", "family"]
).reset_index(drop=True)

print("Data sorted by date, store and product family.")

# ------------------------------------------
# 4. Date range
# ------------------------------------------

print("\n--- DATE RANGE ---")

print(f"Start Date : {df['date'].min()}")
print(f"End Date   : {df['date'].max()}")

print(
    f"Number of Days : "
    f"{df['date'].dt.normalize().nunique()}"
)

# ------------------------------------------
# 5. Unique stores and products
# ------------------------------------------

print("\n--- DATA COVERAGE ---")

print(f"Unique Stores   : {df['store_nbr'].nunique()}")
print(f"Unique Families : {df['family'].nunique()}")
print(f"Unique Cities   : {df['city'].nunique()}")
print(f"Unique States   : {df['state'].nunique()}")

# ------------------------------------------
# 6. Sales validation
# ------------------------------------------

print("\n--- SALES VALIDATION ---")

print(f"Minimum Sales : {df['sales'].min()}")
print(f"Maximum Sales : {df['sales'].max()}")
print(f"Average Sales : {df['sales'].mean():.4f}")

negative_sales = (df["sales"] < 0).sum()

print(f"Negative Sales Records : {negative_sales}")

# ------------------------------------------
# 7. Missing values
# ------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing_values = df[
    ["date", "sales", "store_nbr", "family"]
].isnull().sum()

print(missing_values)

# ------------------------------------------
# 8. Daily total sales
# ------------------------------------------

daily_sales = (
    df.groupby("date", as_index=False)["sales"]
      .sum()
      .sort_values("date")
)

print("\n--- DAILY SALES SUMMARY ---")

print(f"Number of daily records: {len(daily_sales)}")

print("\nFirst 10 days:")
print(daily_sales.head(10))

print("\nLast 10 days:")
print(daily_sales.tail(10))

# ------------------------------------------
# 9. Missing dates check
# ------------------------------------------

print("\n--- DATE CONTINUITY CHECK ---")

full_date_range = pd.date_range(
    start=daily_sales["date"].min(),
    end=daily_sales["date"].max(),
    freq="D"
)

existing_dates = pd.DatetimeIndex(
    daily_sales["date"]
)

missing_dates = full_date_range.difference(
    existing_dates
)

print(f"Expected calendar days : {len(full_date_range)}")
print(f"Existing dates         : {len(existing_dates)}")
print(f"Missing dates          : {len(missing_dates)}")

if len(missing_dates) > 0:
    print("\nFirst missing dates:")
    print(missing_dates[:20])
else:
    print("✓ No missing calendar dates found.")

# ------------------------------------------
# 10. Product-store combinations
# ------------------------------------------

print("\n--- STORE-PRODUCT COVERAGE ---")

store_family_count = (
    df.groupby(["store_nbr", "family"])
      .size()
)

print(
    f"Store-Family combinations: "
    f"{len(store_family_count)}"
)

print(
    f"Minimum records per combination: "
    f"{store_family_count.min()}"
)

print(
    f"Maximum records per combination: "
    f"{store_family_count.max()}"
)

# ------------------------------------------
# 11. Save prepared dataset
# ------------------------------------------

print("\nSaving forecasting-ready dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print("MODULE 2.1 COMPLETED")
print("==========================================")

print(f"Output File : {OUTPUT_FILE}")
print(f"Rows        : {len(df)}")
print(f"Columns     : {len(df.columns)}")