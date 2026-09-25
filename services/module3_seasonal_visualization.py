import pandas as pd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = BASE_DIR / "module3_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

print("\n==========================================")
print("MODULE 3.6 - SEASONAL VISUALIZATION")
print("==========================================")

# ==================================================
# LOAD DATA
# ==================================================
monthly = pd.read_csv(MONTHLY_FILE)
weekday = pd.read_csv(WEEKDAY_FILE)
quarterly = pd.read_csv(QUARTER_FILE)
holiday = pd.read_csv(HOLIDAY_FILE)
store = pd.read_csv(STORE_FILE)
product = pd.read_csv(PRODUCT_FILE)

# ==================================================
# 1. MONTHLY SEASONALITY
# ==================================================
print("\nCreating monthly seasonality chart...")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

ax.plot(
    monthly["Month_Name"],
    monthly["Average_Sales"],
    marker="o"
)

ax.set_title("Monthly Seasonal Demand Pattern")
ax.set_xlabel("Month")
ax.set_ylabel("Average Sales")
ax.tick_params(axis="x", rotation=45)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "01_monthly_seasonality.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 2. WEEKDAY PATTERN
# ==================================================
print("Creating weekday demand chart...")

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot()

ax.bar(
    weekday["Day_Name"],
    weekday["Average_Sales"]
)

ax.set_title("Average Sales by Day of Week")
ax.set_xlabel("Day")
ax.set_ylabel("Average Sales")
ax.tick_params(axis="x", rotation=30)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "02_weekday_pattern.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 3. QUARTERLY PATTERN
# ==================================================
print("Creating quarterly demand chart...")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.bar(
    quarterly["Quarter_Name"],
    quarterly["Average_Sales"]
)

ax.set_title("Quarterly Seasonal Demand")
ax.set_xlabel("Quarter")
ax.set_ylabel("Average Sales")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "03_quarterly_pattern.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 4. HOLIDAY IMPACT
# ==================================================
print("Creating holiday impact chart...")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.bar(
    holiday["Day_Type"],
    holiday["Average_Sales"]
)

ax.set_title("Holiday vs Normal Day Sales")
ax.set_xlabel("Day Type")
ax.set_ylabel("Average Sales")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "04_holiday_impact.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 5. TOP 10 STORES
# ==================================================
print("Creating top store chart...")

top_stores = (
    store
    .sort_values(
        "Average_Sales",
        ascending=False
    )
    .head(10)
    .sort_values("Average_Sales")
)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

ax.barh(
    top_stores["store_nbr"].astype(str),
    top_stores["Average_Sales"]
)

ax.set_title("Top 10 Stores by Average Sales")
ax.set_xlabel("Average Sales")
ax.set_ylabel("Store Number")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "05_top_stores.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 6. TOP 10 PRODUCT FAMILIES
# ==================================================
print("Creating top product family chart...")

top_products = (
    product
    .sort_values(
        "Average_Sales",
        ascending=False
    )
    .head(10)
    .sort_values("Average_Sales")
)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

ax.barh(
    top_products["family"],
    top_products["Average_Sales"]
)

ax.set_title("Top 10 Product Families by Average Sales")
ax.set_xlabel("Average Sales")
ax.set_ylabel("Product Family")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "06_top_product_families.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 7. SEASONAL INDEX
# ==================================================
print("Creating monthly seasonal index chart...")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

ax.plot(
    monthly["Month_Name"],
    monthly["Seasonal_Index"],
    marker="o"
)

ax.axhline(
    1.0,
    linestyle="--"
)

ax.set_title("Monthly Seasonal Index")
ax.set_xlabel("Month")
ax.set_ylabel("Seasonal Index")
ax.tick_params(axis="x", rotation=45)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "07_seasonal_index.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# ==================================================
# 8. CREATE VISUALIZATION INDEX
# ==================================================
visualization_index = pd.DataFrame([
    {
        "Chart_ID": 1,
        "Chart_Name": "Monthly Seasonality",
        "File": "01_monthly_seasonality.png",
        "Purpose": "Shows monthly demand pattern"
    },
    {
        "Chart_ID": 2,
        "Chart_Name": "Weekday Pattern",
        "File": "02_weekday_pattern.png",
        "Purpose": "Shows demand variation by weekday"
    },
    {
        "Chart_ID": 3,
        "Chart_Name": "Quarterly Pattern",
        "File": "03_quarterly_pattern.png",
        "Purpose": "Shows quarterly demand pattern"
    },
    {
        "Chart_ID": 4,
        "Chart_Name": "Holiday Impact",
        "File": "04_holiday_impact.png",
        "Purpose": "Compares holiday and normal-day demand"
    },
    {
        "Chart_ID": 5,
        "Chart_Name": "Top Stores",
        "File": "05_top_stores.png",
        "Purpose": "Shows highest-demand stores"
    },
    {
        "Chart_ID": 6,
        "Chart_Name": "Top Product Families",
        "File": "06_top_product_families.png",
        "Purpose": "Shows highest-demand product families"
    },
    {
        "Chart_ID": 7,
        "Chart_Name": "Seasonal Index",
        "File": "07_seasonal_index.png",
        "Purpose": "Shows monthly seasonal strength"
    }
])

INDEX_FILE = (
    BASE_DIR /
    "module3_visualization_index.csv"
)

visualization_index.to_csv(
    INDEX_FILE,
    index=False
)

# ==================================================
# FINAL OUTPUT
# ==================================================
print("\n--- VISUALIZATION OUTPUT ---")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"✓ {file.name}")

print("\n==========================================")
print("MODULE 3.6 COMPLETED")
print("==========================================")

print(f"Visualization Folder : {OUTPUT_DIR}")
print("Visualization Index   : module3_visualization_index.csv")