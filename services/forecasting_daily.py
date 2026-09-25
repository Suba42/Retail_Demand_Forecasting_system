import pandas as pd

# ==========================================
# MODULE 2.2 - TIME-SERIES AGGREGATION
# ==========================================

INPUT_FILE = "module2_forecasting_ready.csv"
OUTPUT_FILE = "module2_daily_forecasting.csv"

print("Loading forecasting-ready dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 2.2 - TIME-SERIES AGGREGATION")
print("==========================================")

# ------------------------------------------
# 1. Daily total sales
# ------------------------------------------

daily_sales = (
    df.groupby("date", as_index=False)
      .agg(
          sales=("sales", "sum"),
          transactions=("Transactions", "sum"),
          promotion_quantity=("Promotion_Quantity", "sum")
      )
      .sort_values("date")
      .reset_index(drop=True)
)

print("\n--- DAILY DATASET ---")

print(f"Rows    : {len(daily_sales)}")
print(f"Columns : {len(daily_sales.columns)}")

print("\nFirst 10 records:")
print(daily_sales.head(10))

print("\nLast 10 records:")
print(daily_sales.tail(10))


# ------------------------------------------
# 2. Weekly aggregation
# ------------------------------------------

weekly_sales = (
    daily_sales
    .set_index("date")
    .resample("W")
    .agg({
        "sales": "sum",
        "transactions": "sum",
        "promotion_quantity": "sum"
    })
    .reset_index()
)

print("\n--- WEEKLY DATASET ---")
print(f"Weekly records: {len(weekly_sales)}")


# ------------------------------------------
# 3. Monthly aggregation
# ------------------------------------------

monthly_sales = (
    daily_sales
    .set_index("date")
    .resample("ME")
    .agg({
        "sales": "sum",
        "transactions": "sum",
        "promotion_quantity": "sum"
    })
    .reset_index()
)

print("\n--- MONTHLY DATASET ---")
print(f"Monthly records: {len(monthly_sales)}")


# ------------------------------------------
# 4. Basic statistics
# ------------------------------------------

print("\n--- DAILY SALES STATISTICS ---")

print(
    daily_sales["sales"].describe()
)


# ------------------------------------------
# 5. Peak demand days
# ------------------------------------------

print("\n--- TOP 10 HIGH-DEMAND DAYS ---")

top_days = (
    daily_sales
    .sort_values("sales", ascending=False)
    .head(10)
)

print(top_days)


# ------------------------------------------
# 6. Lowest demand days
# ------------------------------------------

print("\n--- TOP 10 LOW-DEMAND DAYS ---")

low_days = (
    daily_sales
    .sort_values("sales", ascending=True)
    .head(10)
)

print(low_days)


# ------------------------------------------
# 7. Save datasets
# ------------------------------------------

daily_sales.to_csv(
    OUTPUT_FILE,
    index=False
)

weekly_sales.to_csv(
    "module2_weekly_forecasting.csv",
    index=False
)

monthly_sales.to_csv(
    "module2_monthly_forecasting.csv",
    index=False
)

print("\n==========================================")
print("MODULE 2.2 COMPLETED")
print("==========================================")

print("Created files:")
print(f"✓ {OUTPUT_FILE}")
print("✓ module2_weekly_forecasting.csv")
print("✓ module2_monthly_forecasting.csv")