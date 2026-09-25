# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# DATA CLEANER
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "date": [
        "date",
        "order_date",
        "sales_date",
        "transaction_date",
        "invoice_date",
        "purchase_date",
        "day",
        "datetime",
        "timestamp"
    ],

    "product": [
        "product",
        "product_name",
        "productname",
        "item",
        "item_name",
        "itemname",
        "sku_name",
        "product_description"
    ],

    "product_id": [
        "product_id",
        "productid",
        "item_id",
        "itemid",
        "sku",
        "sku_id",
        "stock_code"
    ],

    "category": [
        "category",
        "product_category",
        "product_type",
        "item_category",
        "department",
        "segment"
    ],

    # ========================================================
    # DEMAND
    # ========================================================
    # quantity-related columns are treated as demand because
    # demand is the forecasting target in this project.

    "demand": [
        "demand",
        "quantity",
        "qty",
        "units",
        "units_sold",
        "unit_sold",
        "sales_quantity",
        "order_quantity",
        "number_of_units",
        "quantity_sold",
        "sold_quantity"
    ],

    "sales": [
        "sales",
        "sales_amount",
        "total_sales",
        "revenue",
        "revenue_amount",
        "turnover",
        "amount",
        "total_amount"
    ],

    "price": [
        "price",
        "unit_price",
        "selling_price",
        "sale_price",
        "product_price",
        "mrp",
        "cost"
    ],

    "discount": [
        "discount",
        "discount_percent",
        "discount_percentage",
        "discount_rate",
        "discount_amount"
    ],

    "promotion": [
        "promotion",
        "promotional",
        "promo",
        "promotion_status",
        "is_promotion",
        "on_promotion"
    ],

    "store": [
        "store",
        "store_id",
        "store_name",
        "shop",
        "shop_id",
        "branch",
        "branch_id"
    ],

    "customer": [
        "customer",
        "customer_id",
        "customer_name",
        "client",
        "client_id"
    ],

    "supplier": [
        "supplier",
        "supplier_id",
        "supplier_name",
        "vendor",
        "vendor_id"
    ],

    # ========================================================
    # INVENTORY
    # ========================================================
    # stock-related columns are treated as inventory.

    "inventory": [
        "inventory",
        "stock",
        "current_stock",
        "inventory_stock",
        "stock_quantity",
        "available_stock",
        "on_hand",
        "quantity_in_stock",
        "stock_level"
    ],

    "reorder_level": [
        "reorder_level",
        "reorder_point",
        "reorder_threshold",
        "minimum_stock",
        "min_stock",
        "safety_level"
    ],

    "lead_time": [
        "lead_time",
        "leadtime",
        "delivery_time",
        "supplier_lead_time",
        "days_to_delivery"
    ],

    "temperature": [
        "temperature",
        "temp",
        "temperature_c",
        "temp_c"
    ],

    "humidity": [
        "humidity",
        "humidity_percent"
    ],

    "holiday": [
        "holiday",
        "holiday_flag",
        "is_holiday",
        "holiday_status"
    ],

    "festival": [
        "festival",
        "festival_flag",
        "is_festival",
        "festival_name",
        "event"
    ]
}


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(column):

    column = str(column).strip().lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    column = re.sub(
        r"_+",
        "_",
        column
    )

    column = column.strip("_")

    return column


# ============================================================
# CLEAN ALL COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df = df.copy()

    original_columns = list(
        df.columns
    )

    new_columns = []

    for column in df.columns:

        new_columns.append(
            normalize_column_name(column)
        )

    # --------------------------------------------------------
    # Handle duplicate column names
    # --------------------------------------------------------

    seen = {}

    final_columns = []

    for column in new_columns:

        if column not in seen:

            seen[column] = 0

            final_columns.append(
                column
            )

        else:

            seen[column] += 1

            final_columns.append(
                f"{column}_{seen[column]}"
            )

    df.columns = final_columns

    return df, original_columns


# ============================================================
# AUTOMATIC COLUMN MAPPING
# ============================================================

def automatic_column_mapping(df):

    mapping = {}

    used_columns = set()

    # --------------------------------------------------------
    # EXACT MATCHING
    # --------------------------------------------------------

    for standard_name, aliases in COLUMN_ALIASES.items():

        for column in df.columns:

            if column in used_columns:
                continue

            if column in aliases:

                mapping[standard_name] = column

                used_columns.add(
                    column
                )

                break

    # --------------------------------------------------------
    # PARTIAL MATCHING
    # --------------------------------------------------------

    for standard_name, aliases in COLUMN_ALIASES.items():

        if standard_name in mapping:
            continue

        for column in df.columns:

            if column in used_columns:
                continue

            for alias in aliases:

                if (
                    alias in column
                    or
                    column in alias
                ):

                    mapping[standard_name] = column

                    used_columns.add(
                        column
                    )

                    break

            if standard_name in mapping:
                break

    return mapping


# ============================================================
# DETECT DATA TYPES
# ============================================================

def detect_data_types(df):

    data_types = {}

    for column in df.columns:

        dtype = str(
            df[column].dtype
        )

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            detected_type = "numeric"

        elif pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            detected_type = "date"

        elif pd.api.types.is_bool_dtype(
            df[column]
        ):

            detected_type = "boolean"

        else:

            detected_type = "text"

        data_types[column] = {

            "pandas_dtype": dtype,

            "detected_type": detected_type

        }

    return data_types


# ============================================================
# AUTOMATIC DATE DETECTION
# ============================================================

def detect_date_columns(df):

    date_columns = []

    date_keywords = [
        "date",
        "time",
        "timestamp"
    ]

    for column in df.columns:

        column_name = column.lower()

        name_match = any(
            keyword in column_name
            for keyword in date_keywords
        )

        if name_match:

            try:

                converted = pd.to_datetime(
                    df[column],
                    format="%Y-%m-%d",
                    errors="coerce"
                )

                valid_ratio = (
                    converted.notna().mean()
                )

                if valid_ratio >= 0.5:

                    date_columns.append(
                        column
                    )

            except Exception:

                pass

    return date_columns


# ============================================================
# CONVERT DATE COLUMNS
# ============================================================

def convert_date_columns(
    df,
    date_columns
):

    df = df.copy()

    for column in date_columns:

        try:

            df[column] = pd.to_datetime(
                df[column],
                format="%Y-%m-%d",
                errors="coerce"
            )

        except Exception:

            pass

    return df


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(df):

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            cleaned = (
                df[column]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False
                )
                .str.replace(
                    "₹",
                    "",
                    regex=False
                )
                .str.replace(
                    "$",
                    "",
                    regex=False
                )
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
                .str.strip()
            )

            numeric_values = pd.to_numeric(
                cleaned,
                errors="coerce"
            )

            valid_ratio = (
                numeric_values.notna().mean()
            )

            if valid_ratio >= 0.8:

                df[column] = numeric_values

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()

    return df, duplicate_count


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    df = df.copy()

    missing_before = int(
        df.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # OIL PRICE
    # --------------------------------------------------------

    if "oil_price" in df.columns:

        df["oil_price_missing"] = (
            df["oil_price"]
            .isna()
            .astype(int)
        )

    if "oil_price" in df.columns:

        if "date" in df.columns:

            df = df.sort_values(
                "date"
            )

        df["oil_price"] = (
            df["oil_price"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

    # --------------------------------------------------------
    # HOLIDAY COLUMNS
    # --------------------------------------------------------

    holiday_columns = [
        "holiday_types",
        "holiday_locales",
        "holiday_names"
    ]

    for column in holiday_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("No Holiday")
            )

    # --------------------------------------------------------
    # REMAINING MISSING VALUES
    # --------------------------------------------------------

    for column in df.columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count == 0:
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            median_value = (
                df[column].median()
            )

            if pd.isna(median_value):

                median_value = 0

            df[column] = (
                df[column]
                .fillna(median_value)
            )

        # Date
        elif pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            df[column] = (
                df[column]
                .ffill()
                .bfill()
            )

        # Text
        else:

            df[column] = (
                df[column]
                .fillna("Unknown")
            )

    missing_after = int(
        df.isnull().sum().sum()
    )

    return (
        df,
        missing_before,
        missing_after
    )


# ============================================================
# OUTLIER DETECTION
# ============================================================

def detect_outliers(df):

    outlier_report = {}

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        series = (
            df[column]
            .dropna()
        )

        if len(series) < 5:

            outlier_report[column] = 0

            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:

            outlier_report[column] = 0

            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outliers = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        outlier_report[column] = int(
            outliers.sum()
        )

    return outlier_report


# ============================================================
# REMOVE EXTREME OUTLIERS
# ============================================================

def remove_extreme_outliers(df):

    df = df.copy()

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    removed_count = 0

    for column in numeric_columns:

        series = df[column]

        if (
            series.dropna()
            .shape[0] < 5
        ):

            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = (
            q1 - 3 * iqr
        )

        upper_bound = (
            q3 + 3 * iqr
        )

        invalid_rows = (
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        )

        removed_count += int(
            invalid_rows.sum()
        )

        df = df[
            ~invalid_rows
        ]

    return (
        df,
        removed_count
    )


# ============================================================
# STANDARDIZE TEXT VALUES
# ============================================================

def standardize_text_columns(df):

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
            )

            df[column] = (
                df[column]
                .str.replace(
                    r"\s+",
                    " ",
                    regex=True
                )
            )

    return df


# ============================================================
# VALIDATE NUMERIC VALUES
# ============================================================

def validate_numeric_values(df):

    warnings = []

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        negative_count = int(
            (df[column] < 0).sum()
        )

        if negative_count > 0:

            warnings.append(
                f"{column}: "
                f"{negative_count} negative values detected."
            )

    return warnings


# ============================================================
# CREATE CLEANING REPORT
# ============================================================

def create_cleaning_report(
    original_df,
    cleaned_df,
    mapping,
    duplicate_count,
    missing_before,
    missing_after,
    outlier_report,
    removed_outliers,
    warnings
):

    report = {

        "original_rows":
            int(len(original_df)),

        "cleaned_rows":
            int(len(cleaned_df)),

        "original_columns":
            int(len(original_df.columns)),

        "cleaned_columns":
            int(len(cleaned_df.columns)),

        "duplicates_removed":
            int(duplicate_count),

        "missing_values_before":
            int(missing_before),

        "missing_values_after":
            int(missing_after),

        "outliers_detected":
            outlier_report,

        "extreme_outliers_removed":
            int(removed_outliers),

        "automatic_column_mapping":
            mapping,

        "warnings":
            warnings

    }

    return report


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean_dataset(
    df,
    remove_outliers=False
):

    # --------------------------------------------------------
    # Preserve original dataset
    # --------------------------------------------------------

    original_df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df, original_columns = (
        clean_column_names(df)
    )

    # --------------------------------------------------------
    # Automatic column mapping
    # --------------------------------------------------------

    mapping = automatic_column_mapping(
        df
    )

    # --------------------------------------------------------
    # Convert numeric values
    # --------------------------------------------------------

    df = convert_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # Detect date columns
    # --------------------------------------------------------

    date_columns = detect_date_columns(
        df
    )

    # --------------------------------------------------------
    # Convert date columns
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        date_columns
    )

    # --------------------------------------------------------
    # Standardize text
    # --------------------------------------------------------

    df = standardize_text_columns(
        df
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    (
        df,
        duplicate_count
    ) = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    (
        df,
        missing_before,
        missing_after
    ) = handle_missing_values(
        df
    )

    # ========================================================
    # OUTLIER DETECTION
    # ========================================================

    outlier_report = detect_outliers(
        df
    )

    # --------------------------------------------------------
    # Optional extreme outlier removal
    # --------------------------------------------------------

    removed_outliers = 0

    if remove_outliers:

        (
            df,
            removed_outliers
        ) = remove_extreme_outliers(
            df
        )

    # --------------------------------------------------------
    # Validate numerical values
    # --------------------------------------------------------

    warnings = validate_numeric_values(
        df
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = create_cleaning_report(

        original_df=original_df,

        cleaned_df=df,

        mapping=mapping,

        duplicate_count=duplicate_count,

        missing_before=missing_before,

        missing_after=missing_after,

        outlier_report=outlier_report,

        removed_outliers=removed_outliers,

        warnings=warnings

    )

    return (
        df,
        report
    )


# ============================================================
# SIMPLE TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    sample_data = {

        "Order Date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-03"
        ],

        "Product Name": [
            "Milk",
            "Bread",
            "Milk",
            "Milk"
        ],

        "Qty Sold": [
            10,
            20,
            np.nan,
            10
        ],

        "Unit Price": [
            50,
            40,
            50,
            50
        ],

        "Current Stock": [
            100,
            80,
            90,
            100
        ]

    }

    sample_df = pd.DataFrame(
        sample_data
    )

    cleaned_df, report = (
        clean_dataset(
            sample_df
        )
    )

    print(
        "\n========================================"
    )

    print(
        "AUTOMATIC COLUMN MAPPING"
    )

    print(
        "========================================"
    )

    print(
        report[
            "automatic_column_mapping"
        ]
    )

    print(
        "\n========================================"
    )

    print(
        "CLEANED DATASET"
    )

    print(
        "========================================"
    )

    print(
        cleaned_df
    )

    print(
        "\n========================================"
    )

    print(
        "CLEANING REPORT"
    )

    print(
        "========================================"
    )

    print(
        report
    )