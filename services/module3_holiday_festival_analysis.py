import pandas as pd
from pathlib import Path

print("Loading seasonal-ready dataset...")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "module3_seasonal_ready.csv"

HOLIDAY_OUTPUT = BASE_DIR / "module3_holiday_impact.csv"
FESTIVAL_OUTPUT = BASE_DIR / "module3_festival_impact.csv"
HOLIDAY_TYPE_OUTPUT = BASE_DIR / "module3_holiday_type_analysis.csv"
EVENT_OUTPUT = BASE_DIR / "module3_high_demand_events.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 3.3 - HOLIDAY & FESTIVAL ANALYSIS")
print("==========================================")

# --------------------------------------------------
# CHECK AVAILABLE COLUMNS
# --------------------------------------------------
print("\n--- AVAILABLE EVENT COLUMNS ---")

event_columns = [
    "is_holiday",
    "holiday_count",
    "holiday_types",
    "holiday_locales",
    "holiday_names",
    "Festival_Flag"
]

for col in event_columns:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        print(f"- {col} not available")

# --------------------------------------------------
# 1. HOLIDAY IMPACT
# --------------------------------------------------
print("\n--- HOLIDAY IMPACT ---")

if "is_holiday" in df.columns:

    holiday_summary = (
        df.groupby("is_holiday")["sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Records="count"
        )
        .reset_index()
    )

    holiday_summary["Day_Type"] = holiday_summary[
        "is_holiday"
    ].map({
        0: "Normal Day",
        1: "Holiday"
    })

    overall_average = df["sales"].mean()

    holiday_summary["Sales_Index"] = (
        holiday_summary["Average_Sales"] /
        overall_average
    )

    print(holiday_summary)

else:
    holiday_summary = pd.DataFrame()

# --------------------------------------------------
# 2. FESTIVAL IMPACT
# --------------------------------------------------
print("\n--- FESTIVAL / EVENT IMPACT ---")

if "Festival_Flag" in df.columns:

    festival_summary = (
        df.groupby("Festival_Flag")["sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Records="count"
        )
        .reset_index()
    )

    festival_summary["Event_Type"] = festival_summary[
        "Festival_Flag"
    ].map({
        0: "No Festival/Event",
        1: "Festival/Event"
    })

    festival_average = df["sales"].mean()

    festival_summary["Sales_Index"] = (
        festival_summary["Average_Sales"] /
        festival_average
    )

    print(festival_summary)

else:
    festival_summary = pd.DataFrame()

# --------------------------------------------------
# 3. HOLIDAY TYPE ANALYSIS
# --------------------------------------------------
print("\n--- HOLIDAY TYPE ANALYSIS ---")

if "holiday_types" in df.columns:

    holiday_type_df = df.copy()

    holiday_type_df["holiday_types"] = (
        holiday_type_df["holiday_types"]
        .fillna("No Holiday")
        .replace("", "No Holiday")
    )

    holiday_type_summary = (
        holiday_type_df
        .groupby("holiday_types")["sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Records="count"
        )
        .reset_index()
        .sort_values(
            "Average_Sales",
            ascending=False
        )
    )

    print(holiday_type_summary.head(20))

else:
    holiday_type_summary = pd.DataFrame()

# --------------------------------------------------
# 4. HOLIDAY NAME ANALYSIS
# --------------------------------------------------
print("\n--- TOP HOLIDAY / EVENT NAMES ---")

if "holiday_names" in df.columns:

    holiday_name_df = df.copy()

    holiday_name_df["holiday_names"] = (
        holiday_name_df["holiday_names"]
        .fillna("No Holiday")
        .replace("", "No Holiday")
    )

    holiday_name_summary = (
        holiday_name_df
        .groupby("holiday_names")["sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Records="count"
        )
        .reset_index()
    )

    holiday_name_summary = (
        holiday_name_summary[
            holiday_name_summary["holiday_names"]
            != "No Holiday"
        ]
        .sort_values(
            "Average_Sales",
            ascending=False
        )
    )

    print(holiday_name_summary.head(20))

else:
    holiday_name_summary = pd.DataFrame()

# --------------------------------------------------
# 5. HIGH-DEMAND EVENT DAYS
# --------------------------------------------------
print("\n--- HIGH-DEMAND EVENT DAYS ---")

daily = (
    df.groupby("date")
    .agg(
        sales=("sales", "sum"),
        holiday_flag=(
            "is_holiday",
            "max"
        ) if "is_holiday" in df.columns else (
            "sales",
            lambda x: 0
        ),
        festival_flag=(
            "Festival_Flag",
            "max"
        ) if "Festival_Flag" in df.columns else (
            "sales",
            lambda x: 0
        )
    )
    .reset_index()
)

top_events = (
    daily.sort_values(
        "sales",
        ascending=False
    )
    .head(20)
)

print(top_events)

# --------------------------------------------------
# 6. EVENT vs NON-EVENT DAILY SALES
# --------------------------------------------------
print("\n--- EVENT DAY DAILY IMPACT ---")

if "is_holiday" in df.columns:

    daily_holiday = (
        df.groupby("date")
        .agg(
            sales=("sales", "sum"),
            holiday=("is_holiday", "max")
        )
        .reset_index()
    )

    daily_holiday_summary = (
        daily_holiday
        .groupby("holiday")["sales"]
        .agg(
            Total_Sales="sum",
            Average_Daily_Sales="mean",
            Number_of_Days="count"
        )
        .reset_index()
    )

    daily_holiday_summary["Day_Type"] = (
        daily_holiday_summary["holiday"]
        .map({
            0: "Normal Day",
            1: "Holiday"
        })
    )

    print(daily_holiday_summary)

else:
    daily_holiday_summary = pd.DataFrame()

# --------------------------------------------------
# 7. FIND IMPACT
# --------------------------------------------------
print("\n--- KEY EVENT IMPACT ---")

if not holiday_summary.empty:

    normal_row = holiday_summary[
        holiday_summary["is_holiday"] == 0
    ]

    holiday_row = holiday_summary[
        holiday_summary["is_holiday"] == 1
    ]

    if len(normal_row) > 0 and len(holiday_row) > 0:

        normal_avg = normal_row[
            "Average_Sales"
        ].iloc[0]

        holiday_avg = holiday_row[
            "Average_Sales"
        ].iloc[0]

        holiday_change = (
            (holiday_avg - normal_avg)
            / normal_avg
        ) * 100

        print(
            f"Holiday vs Normal Sales Change : "
            f"{holiday_change:.2f}%"
        )

if not festival_summary.empty:

    no_event = festival_summary[
        festival_summary["Festival_Flag"] == 0
    ]

    event = festival_summary[
        festival_summary["Festival_Flag"] == 1
    ]

    if len(no_event) > 0 and len(event) > 0:

        normal_event_avg = no_event[
            "Average_Sales"
        ].iloc[0]

        festival_avg = event[
            "Average_Sales"
        ].iloc[0]

        festival_change = (
            (festival_avg - normal_event_avg)
            / normal_event_avg
        ) * 100

        print(
            f"Festival vs Normal Sales Change : "
            f"{festival_change:.2f}%"
        )

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------
print("\nSaving holiday and festival analysis...")

if not holiday_summary.empty:
    holiday_summary.to_csv(
        HOLIDAY_OUTPUT,
        index=False
    )

if not festival_summary.empty:
    festival_summary.to_csv(
        FESTIVAL_OUTPUT,
        index=False
    )

if not holiday_type_summary.empty:
    holiday_type_summary.to_csv(
        HOLIDAY_TYPE_OUTPUT,
        index=False
    )

top_events.to_csv(
    EVENT_OUTPUT,
    index=False
)

print("\n==========================================")
print("MODULE 3.3 COMPLETED")
print("==========================================")

print("Created files:")

if not holiday_summary.empty:
    print("✓ module3_holiday_impact.csv")

if not festival_summary.empty:
    print("✓ module3_festival_impact.csv")

if not holiday_type_summary.empty:
    print("✓ module3_holiday_type_analysis.csv")

print("✓ module3_high_demand_events.csv")