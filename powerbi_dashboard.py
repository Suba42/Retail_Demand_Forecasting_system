import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, render_template

# ============================================================
# POWER BI STYLE DASHBOARD BLUEPRINT
# ============================================================

powerbi_bp = Blueprint(
    "powerbi",
    __name__,
    url_prefix="/powerbi"
)


# ============================================================
# FIND PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")

SHARED_DATASET_PATH = os.path.join(
    CLEANED_DIR,
    "cleaned_shared_dataset.csv"
)

FORECAST_RESULT_PATH = os.path.join(
    CLEANED_DIR,
    "forecast_results.csv"
)

FORECAST_REPORT_PATH = os.path.join(
    CLEANED_DIR,
    "forecast_report.json"
)

INVENTORY_RESULT_PATH = os.path.join(
    CLEANED_DIR,
    "inventory_optimization_results.csv"
)

INVENTORY_MODEL_INFO_PATH = os.path.join(
    CLEANED_DIR,
    "inventory_model_info.json"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find a column from a list of possible column names.
    """

    normalized = {
        str(col).lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:

        name = name.lower().strip().replace(" ", "_")

        if name in normalized:
            return normalized[name]

    return None


def safe_number(value, default=0):
    """
    Convert a value safely to a number.
    """

    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def load_csv(path):
    """
    Safely load a CSV file.
    """

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, low_memory=False)

    except Exception:
        return None


def load_json(path):
    """
    Safely load a JSON file.
    """

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {}


# ============================================================
# BUILD DASHBOARD DATA
# ============================================================

def build_powerbi_dashboard():

    df = load_csv(SHARED_DATASET_PATH)

    if df is None or df.empty:

        return {
            "available": False,
            "message": "No cleaned dataset available yet."
        }


    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    total_records = len(df)

    total_columns = len(df.columns)


    # --------------------------------------------------------
    # FIND IMPORTANT COLUMNS
    # --------------------------------------------------------

    date_col = find_column(
        df,
        [
            "date",
            "order_date",
            "sales_date",
            "transaction_date",
            "invoice_date",
            "timestamp"
        ]
    )

    product_col = find_column(
        df,
        [
            "product",
            "product_name",
            "product_id",
            "sku",
            "item",
            "item_name"
        ]
    )

    category_col = find_column(
        df,
        [
            "category",
            "product_category",
            "category_name"
        ]
    )

    demand_col = find_column(
        df,
        [
            "demand",
            "sales",
            "quantity",
            "units_sold",
            "sales_quantity",
            "demand_quantity"
        ]
    )

    revenue_col = find_column(
        df,
        [
            "revenue",
            "sales_amount",
            "total_sales",
            "amount",
            "total_amount"
        ]
    )

    inventory_col = find_column(
        df,
        [
            "inventory",
            "stock",
            "stock_quantity",
            "inventory_quantity",
            "current_inventory",
            "available_stock"
        ]
    )

    price_col = find_column(
        df,
        [
            "price",
            "unit_price",
            "selling_price"
        ]
    )


    # ========================================================
    # DATE PROCESSING
    # ========================================================

    date_start = None
    date_end = None

    if date_col:

        dates = pd.to_datetime(
            df[date_col],
            errors="coerce"
        ).dropna()

        if not dates.empty:

            date_start = dates.min().strftime("%Y-%m-%d")

            date_end = dates.max().strftime("%Y-%m-%d")


    # ========================================================
    # DEMAND / SALES
    # ========================================================

    total_demand = 0

    if demand_col:

        demand_values = pd.to_numeric(
            df[demand_col],
            errors="coerce"
        ).fillna(0)

        total_demand = demand_values.sum()


    # ========================================================
    # REVENUE
    # ========================================================

    total_revenue = 0

    if revenue_col:

        revenue_values = pd.to_numeric(
            df[revenue_col],
            errors="coerce"
        ).fillna(0)

        total_revenue = revenue_values.sum()

    elif demand_col and price_col:

        demand_values = pd.to_numeric(
            df[demand_col],
            errors="coerce"
        ).fillna(0)

        price_values = pd.to_numeric(
            df[price_col],
            errors="coerce"
        ).fillna(0)

        total_revenue = (
            demand_values * price_values
        ).sum()


    # ========================================================
    # INVENTORY
    # ========================================================

    total_inventory = 0

    if inventory_col:

        inventory_values = pd.to_numeric(
            df[inventory_col],
            errors="coerce"
        ).fillna(0)

        total_inventory = inventory_values.sum()


    # ========================================================
    # PRODUCT COUNT
    # ========================================================

    product_count = 0

    if product_col:

        product_count = (
            df[product_col]
            .dropna()
            .astype(str)
            .nunique()
        )


    # ========================================================
    # CATEGORY COUNT
    # ========================================================

    category_count = 0

    if category_col:

        category_count = (
            df[category_col]
            .dropna()
            .astype(str)
            .nunique()
        )


    # ========================================================
    # TOP PRODUCTS
    # ========================================================

    top_products = []

    if product_col and demand_col:

        temp = df[
            [product_col, demand_col]
        ].copy()

        temp[demand_col] = pd.to_numeric(
            temp[demand_col],
            errors="coerce"
        ).fillna(0)

        grouped = (
            temp
            .groupby(product_col)[demand_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        for product, value in grouped.items():

            top_products.append({
                "product": str(product),
                "demand": round(float(value), 2)
            })


    # ========================================================
    # CATEGORY ANALYSIS
    # ========================================================

    category_data = []

    if category_col and demand_col:

        temp = df[
            [category_col, demand_col]
        ].copy()

        temp[demand_col] = pd.to_numeric(
            temp[demand_col],
            errors="coerce"
        ).fillna(0)

        grouped = (
            temp
            .groupby(category_col)[demand_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        for category, value in grouped.items():

            category_data.append({
                "category": str(category),
                "demand": round(float(value), 2)
            })


    # ========================================================
    # DEMAND TREND
    # ========================================================

    demand_trend = []

    if date_col and demand_col:

        temp = df[
            [date_col, demand_col]
        ].copy()

        temp[date_col] = pd.to_datetime(
            temp[date_col],
            errors="coerce"
        )

        temp[demand_col] = pd.to_numeric(
            temp[demand_col],
            errors="coerce"
        ).fillna(0)

        temp = temp.dropna(
            subset=[date_col]
        )

        if not temp.empty:

            temp["period"] = temp[
                date_col
            ].dt.to_period("M").astype(str)

            grouped = (
                temp
                .groupby("period")[demand_col]
                .sum()
                .reset_index()
            )

            grouped = grouped.tail(24)

            for _, row in grouped.iterrows():

                demand_trend.append({
                    "period": str(row["period"]),
                    "demand": round(
                        float(row[demand_col]),
                        2
                    )
                })


    # ========================================================
    # INVENTORY STATUS
    # ========================================================

    low_stock_count = 0
    zero_stock_count = 0

    if inventory_col:

        inventory_values = pd.to_numeric(
            df[inventory_col],
            errors="coerce"
        ).fillna(0)

        zero_stock_count = int(
            (inventory_values <= 0).sum()
        )

        # Uses the dataset's distribution instead
        # of inventing a business threshold.
        low_stock_limit = inventory_values.quantile(0.10)

        low_stock_count = int(
            (
                inventory_values <= low_stock_limit
            ).sum()
        )


    # ========================================================
    # FORECAST INFORMATION
    # ========================================================

    forecast_report = load_json(
        FORECAST_REPORT_PATH
    )

    forecast_mae = None
    forecast_rmse = None
    forecast_model = "Not available"


    if forecast_report:

        forecast_model = (
            forecast_report.get("model")
            or forecast_report.get("best_model")
            or forecast_report.get("selected_model")
            or "Not available"
        )

        forecast_mae = (
            forecast_report.get("mae")
            or forecast_report.get("MAE")
        )

        forecast_rmse = (
            forecast_report.get("rmse")
            or forecast_report.get("RMSE")
        )


    # ========================================================
    # FORECAST RESULTS
    # ========================================================

    forecast_data = []

    forecast_df = load_csv(
        FORECAST_RESULT_PATH
    )

    if forecast_df is not None and not forecast_df.empty:

        forecast_date_col = find_column(
            forecast_df,
            [
                "date",
                "ds",
                "forecast_date",
                "period"
            ]
        )

        actual_col = find_column(
            forecast_df,
            [
                "actual",
                "actual_demand",
                "y"
            ]
        )

        predicted_col = find_column(
            forecast_df,
            [
                "predicted",
                "prediction",
                "forecast",
                "yhat",
                "forecasted_demand"
            ]
        )

        if forecast_date_col and predicted_col:

            temp = forecast_df.copy()

            if forecast_date_col:

                temp[forecast_date_col] = pd.to_datetime(
                    temp[forecast_date_col],
                    errors="coerce"
                )

            temp = temp.dropna(
                subset=[forecast_date_col]
            )

            temp = temp.tail(24)

            for _, row in temp.iterrows():

                item = {
                    "date": str(
                        row[forecast_date_col]
                    ),
                    "forecast": safe_number(
                        row[predicted_col]
                    )
                }

                if actual_col:

                    item["actual"] = safe_number(
                        row[actual_col]
                    )

                forecast_data.append(item)


    # ========================================================
    # FINAL DASHBOARD DATA
    # ========================================================

    return {

        "available": True,

        "kpis": {

            "total_records": int(total_records),

            "total_columns": int(total_columns),

            "products": int(product_count),

            "categories": int(category_count),

            "total_demand": round(
                float(total_demand),
                2
            ),

            "total_revenue": round(
                float(total_revenue),
                2
            ),

            "total_inventory": round(
                float(total_inventory),
                2
            ),

            "low_stock": int(low_stock_count),

            "zero_stock": int(zero_stock_count)

        },

        "date_range": {

            "start": date_start,

            "end": date_end

        },

        "top_products": top_products,

        "categories": category_data,

        "demand_trend": demand_trend,

        "forecast": forecast_data,

        "forecast_info": {

            "model": forecast_model,

            "mae": forecast_mae,

            "rmse": forecast_rmse

        }

    }


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@powerbi_bp.route("/")
def powerbi_dashboard():

    dashboard_data = build_powerbi_dashboard()

    return render_template(
        "powerbi_dashboard.html",
        dashboard=dashboard_data
    )