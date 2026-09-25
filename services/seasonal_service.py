# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# SEASONAL ANALYSIS MACHINE LEARNING SERVICE
# ============================================================

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf


# ============================================================
# COLUMN DETECTION
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_seasonal_column(df, candidates):

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # Exact match
    for candidate in candidates:

        normalized_candidate = (
            normalize_column_name(candidate)
        )

        if normalized_candidate in normalized_columns:

            return normalized_columns[
                normalized_candidate
            ]

    # Partial match
    for normalized_name, original_name in (
        normalized_columns.items()
    ):

        for candidate in candidates:

            normalized_candidate = (
                normalize_column_name(candidate)
            )

            if (
                normalized_candidate in normalized_name
                or
                normalized_name in normalized_candidate
            ):

                return original_name

    return None


def detect_seasonal_columns(df):

    return {

        "date": find_seasonal_column(
            df,
            [
                "date",
                "order_date",
                "sale_date",
                "sales_date",
                "transaction_date"
            ]
        ),

        "demand": find_seasonal_column(
            df,
            [
                "demand",
                "quantity",
                "qty",
                "units",
                "units_sold",
                "sales_quantity",
                "quantity_sold",
                "sold_quantity"
            ]
        ),

        "sales": find_seasonal_column(
            df,
            [
                "sales",
                "sales_amount",
                "revenue",
                "total_sales",
                "amount"
            ]
        ),

        "product": find_seasonal_column(
            df,
            [
                "product",
                "product_name",
                "product_id",
                "sku",
                "item",
                "item_id"
            ]
        ),

        "category": find_seasonal_column(
            df,
            [
                "category",
                "product_category",
                "category_name"
            ]
        ),

        "festival": find_seasonal_column(
            df,
            [
                "festival",
                "festival_name",
                "holiday",
                "holiday_name",
                "special_event",
                "event"
            ]
        )
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_seasonal_data(df):

    if df is None:

        return {
            "success": False,
            "message": "No dataset was provided."
        }

    if df.empty:

        return {
            "success": False,
            "message": "The dataset is empty."
        }

    data = df.copy()

    columns = detect_seasonal_columns(data)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if columns["date"] is None:

        return {
            "success": False,
            "message": (
                "A valid date column is required "
                "for seasonal analysis."
            )
        }

    data["seasonal_date"] = pd.to_datetime(
        data[columns["date"]],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    demand_column = (
        columns["demand"]
        or columns["sales"]
    )

    if demand_column is None:

        return {
            "success": False,
            "message": (
                "No demand or sales column was detected."
            )
        }

    data["seasonal_demand"] = pd.to_numeric(
        data[demand_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "seasonal_date",
            "seasonal_demand"
        ]
    )

    data = data[
        data["seasonal_demand"] >= 0
    ]

    if data.empty:

        return {
            "success": False,
            "message": (
                "No valid date and demand records "
                "were found."
            )
        }

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_values(
        "seasonal_date"
    )

    return {

        "success": True,

        "data": data,

        "columns": columns,

        "demand_column": demand_column,

        "message": (
            "Seasonal data prepared successfully."
        )
    }


# ============================================================
# CREATE TIME SERIES
# ============================================================

def create_daily_time_series(df):

    daily = (
        df
        .groupby("seasonal_date")[
            "seasonal_demand"
        ]
        .sum()
        .sort_index()
    )

    if daily.empty:

        return pd.Series(
            dtype=float
        )

    # --------------------------------------------------------
    # Create continuous daily index
    # --------------------------------------------------------

    full_index = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(
        full_index
    )

    # --------------------------------------------------------
    # Missing days are treated as zero demand
    # --------------------------------------------------------

    daily = daily.fillna(0)

    daily.index.name = "date"

    return daily


# ============================================================
# WEEKLY SEASONALITY
# ============================================================

def analyze_weekly_seasonality(df):

    data = df.copy()

    data["day_of_week"] = (
        data["seasonal_date"]
        .dt.dayofweek
    )

    weekly = (
        data
        .groupby("day_of_week")[
            "seasonal_demand"
        ]
        .mean()
    )

    day_names = [

        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekly_data = []

    for day_number in range(7):

        value = weekly.get(
            day_number,
            0
        )

        weekly_data.append({

            "day": day_names[day_number],

            "demand": round(
                float(value),
                2
            )
        })

    if weekly_data:

        peak_day = max(
            weekly_data,
            key=lambda item: item["demand"]
        )

        lowest_day = min(
            weekly_data,
            key=lambda item: item["demand"]
        )

    else:

        peak_day = None
        lowest_day = None

    return {

        "available": True,

        "weekly_data": weekly_data,

        "peak_day": (
            peak_day["day"]
            if peak_day
            else None
        ),

        "peak_demand": (
            peak_day["demand"]
            if peak_day
            else None
        ),

        "lowest_day": (
            lowest_day["day"]
            if lowest_day
            else None
        ),

        "lowest_demand": (
            lowest_day["demand"]
            if lowest_day
            else None
        )
    }


# ============================================================
# MONTHLY SEASONALITY
# ============================================================

def analyze_monthly_seasonality(df):

    data = df.copy()

    data["month"] = (
        data["seasonal_date"]
        .dt.month
    )

    monthly = (
        data
        .groupby("month")[
            "seasonal_demand"
        ]
        .mean()
    )

    month_names = [

        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    monthly_data = []

    for month_number in range(1, 13):

        value = monthly.get(
            month_number,
            np.nan
        )

        if pd.isna(value):
            continue

        monthly_data.append({

            "month": month_names[
                month_number - 1
            ],

            "month_number": month_number,

            "demand": round(
                float(value),
                2
            )
        })

    if monthly_data:

        peak_month = max(
            monthly_data,
            key=lambda item: item["demand"]
        )

        lowest_month = min(
            monthly_data,
            key=lambda item: item["demand"]
        )

    else:

        peak_month = None
        lowest_month = None

    return {

        "available": True,

        "monthly_data": monthly_data,

        "peak_month": (
            peak_month["month"]
            if peak_month
            else None
        ),

        "peak_demand": (
            peak_month["demand"]
            if peak_month
            else None
        ),

        "lowest_month": (
            lowest_month["month"]
            if lowest_month
            else None
        ),

        "lowest_demand": (
            lowest_month["demand"]
            if lowest_month
            else None
        )
    }


# ============================================================
# QUARTERLY SEASONALITY
# ============================================================

def analyze_quarterly_seasonality(df):

    data = df.copy()

    data["quarter"] = (
        data["seasonal_date"]
        .dt.quarter
    )

    quarterly = (
        data
        .groupby("quarter")[
            "seasonal_demand"
        ]
        .mean()
    )

    quarterly_data = []

    for quarter_number in range(1, 5):

        value = quarterly.get(
            quarter_number,
            np.nan
        )

        if pd.isna(value):
            continue

        quarterly_data.append({

            "quarter": (
                f"Q{quarter_number}"
            ),

            "demand": round(
                float(value),
                2
            )
        })

    if quarterly_data:

        peak_quarter = max(
            quarterly_data,
            key=lambda item: item["demand"]
        )

    else:

        peak_quarter = None

    return {

        "available": True,

        "quarterly_data": quarterly_data,

        "peak_quarter": (
            peak_quarter["quarter"]
            if peak_quarter
            else None
        )
    }


# ============================================================
# SEASONAL INDEX
# ============================================================

def calculate_seasonal_indices(df):

    data = df.copy()

    data["month"] = (
        data["seasonal_date"]
        .dt.month
    )

    monthly_demand = (
        data
        .groupby("month")[
            "seasonal_demand"
        ]
        .mean()
    )

    overall_average = (
        data["seasonal_demand"]
        .mean()
    )

    if overall_average == 0:

        return {
            "available": False,
            "indices": []
        }

    indices = []

    for month_number in range(1, 13):

        if month_number not in monthly_demand.index:
            continue

        average_demand = float(
            monthly_demand.loc[
                month_number
            ]
        )

        seasonal_index = (
            average_demand
            /
            overall_average
        )

        indices.append({

            "month": month_number,

            "seasonal_index": round(
                float(seasonal_index),
                4
            ),

            "percentage": round(
                float(
                    seasonal_index * 100
                ),
                2
            )
        })

    return {

        "available": True,

        "indices": indices,

        "overall_average": round(
            float(overall_average),
            2
        )
    }


# ============================================================
# STL SEASONAL DECOMPOSITION
# ============================================================

def perform_stl_decomposition(
    daily_series
):

    if daily_series is None:

        return {
            "available": False,
            "message": "No time series available."
        }

    if len(daily_series) < 14:

        return {
            "available": False,
            "message": (
                "At least 14 daily observations "
                "are recommended for decomposition."
            )
        }

    # --------------------------------------------------------
    # Choose seasonal period
    # --------------------------------------------------------

    # Weekly seasonality is used for daily retail data.
    period = 7

    if len(daily_series) < (
        period * 2
    ):

        return {
            "available": False,
            "message": (
                "Not enough observations for "
                "STL seasonal decomposition."
            )
        }

    try:

        stl = STL(
            daily_series,
            period=period,
            robust=True
        )

        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        # ----------------------------------------------------
        # Seasonal strength
        # ----------------------------------------------------

        residual_variance = np.var(
            residual
        )

        seasonal_residual_variance = np.var(
            seasonal + residual
        )

        if seasonal_residual_variance == 0:

            seasonal_strength = 0

        else:

            seasonal_strength = max(

                0,

                1 - (
                    residual_variance
                    /
                    seasonal_residual_variance
                )
            )

        # ----------------------------------------------------
        # Trend strength
        # ----------------------------------------------------

        trend_residual_variance = np.var(
            trend + residual
        )

        if trend_residual_variance == 0:

            trend_strength = 0

        else:

            trend_strength = max(

                0,

                1 - (
                    residual_variance
                    /
                    trend_residual_variance
                )
            )

        # ----------------------------------------------------
        # Return sampled decomposition
        # ----------------------------------------------------

        decomposition_data = []

        for index in range(
            len(daily_series)
        ):

            date_value = (
                daily_series.index[index]
            )

            decomposition_data.append({

                "date": pd.to_datetime(
                    date_value
                ).strftime(
                    "%Y-%m-%d"
                ),

                "actual": round(
                    float(
                        daily_series.iloc[index]
                    ),
                    2
                ),

                "trend": round(
                    float(
                        trend.iloc[index]
                    ),
                    2
                ),

                "seasonal": round(
                    float(
                        seasonal.iloc[index]
                    ),
                    2
                ),

                "residual": round(
                    float(
                        residual.iloc[index]
                    ),
                    2
                )
            })

        return {

            "available": True,

            "method": "STL Decomposition",

            "period": period,

            "seasonal_strength": round(
                float(
                    seasonal_strength
                ),
                4
            ),

            "seasonal_strength_percentage": round(
                float(
                    seasonal_strength * 100
                ),
                2
            ),

            "trend_strength": round(
                float(
                    trend_strength
                ),
                4
            ),

            "trend_strength_percentage": round(
                float(
                    trend_strength * 100
                ),
                2
            ),

            "decomposition": decomposition_data
        }

    except Exception as error:

        return {

            "available": False,

            "message": (
                f"STL decomposition failed: {error}"
            )
        }


# ============================================================
# AUTOCORRELATION / SEASONALITY DETECTION
# ============================================================

def detect_seasonal_cycles(
    daily_series
):

    if daily_series is None:

        return {
            "available": False,
            "cycles": []
        }

    if len(daily_series) < 14:

        return {
            "available": False,
            "cycles": []
        }

    try:

        values = daily_series.values.astype(
            float
        )

        # Remove mean
        values = (
            values
            -
            np.mean(values)
        )

        correlation = acf(
            values,
            nlags=min(
                90,
                len(values) // 2
            ),
            fft=True
        )

        cycles = []

        for lag in range(
            2,
            len(correlation)
        ):

            value = float(
                correlation[lag]
            )

            if value >= 0.30:

                cycles.append({

                    "lag_days": lag,

                    "correlation": round(
                        value,
                        4
                    )
                })

        cycles = sorted(
            cycles,
            key=lambda item: item[
                "correlation"
            ],
            reverse=True
        )

        return {

            "available": True,

            "cycles": cycles[:10],

            "strongest_cycle": (
                cycles[0]
                if cycles
                else None
            )
        }

    except Exception as error:

        return {

            "available": False,

            "cycles": [],

            "message": (
                f"Seasonal cycle detection failed: {error}"
            )
        }


# ============================================================
# FESTIVAL / HOLIDAY ANALYSIS
# ============================================================

def analyze_festival_effect(
    df,
    columns
):

    festival_column = columns.get(
        "festival"
    )

    if festival_column is None:

        return {

            "available": False,

            "message": (
                "No festival or holiday column "
                "was detected."
            ),

            "events": []
        }

    data = df.copy()

    data["_festival"] = (
        data[festival_column]
        .astype(str)
        .str.strip()
    )

    data = data[
        data["_festival"].notna()
    ]

    data = data[
        data["_festival"].str.lower()
        .isin([
            "",
            "nan",
            "none",
            "null",
            "no",
            "normal"
        ]) == False
    ]

    if data.empty:

        return {

            "available": False,

            "message": (
                "No festival or holiday records "
                "were found."
            ),

            "events": []
        }

    overall_average = (
        df["seasonal_demand"]
        .mean()
    )

    events = []

    grouped = (
        data
        .groupby("_festival")[
            "seasonal_demand"
        ]
        .agg([
            "mean",
            "count"
        ])
        .reset_index()
    )

    for _, row in grouped.iterrows():

        average_demand = float(
            row["mean"]
        )

        if overall_average != 0:

            lift_percentage = (

                (
                    average_demand
                    -
                    overall_average
                )
                /
                overall_average
            ) * 100

        else:

            lift_percentage = 0

        events.append({

            "event": str(
                row["_festival"]
            ),

            "average_demand": round(
                average_demand,
                2
            ),

            "records": int(
                row["count"]
            ),

            "demand_lift_percentage": round(
                float(lift_percentage),
                2
            )
        })

    events = sorted(
        events,
        key=lambda item: item[
            "demand_lift_percentage"
        ],
        reverse=True
    )

    return {

        "available": True,

        "events": events
    }


# ============================================================
# PEAK PERIOD DETECTION
# ============================================================

def detect_peak_periods(df):

    data = df.copy()

    data["month"] = (
        data["seasonal_date"]
        .dt.month
    )

    data["day_of_week"] = (
        data["seasonal_date"]
        .dt.dayofweek
    )

    monthly = (
        data
        .groupby("month")[
            "seasonal_demand"
        ]
        .mean()
    )

    weekly = (
        data
        .groupby("day_of_week")[
            "seasonal_demand"
        ]
        .mean()
    )

    monthly_peaks = []

    for month, demand in (
        monthly.sort_values(
            ascending=False
        ).head(3).items()
    ):

        monthly_peaks.append({

            "month": int(month),

            "average_demand": round(
                float(demand),
                2
            )
        })

    weekly_peaks = []

    for day, demand in (
        weekly.sort_values(
            ascending=False
        ).head(3).items()
    ):

        weekly_peaks.append({

            "day": int(day),

            "average_demand": round(
                float(demand),
                2
            )
        })

    return {

        "monthly_peaks": monthly_peaks,

        "weekly_peaks": weekly_peaks
    }


# ============================================================
# SEASONAL BUSINESS RECOMMENDATIONS
# ============================================================

def generate_seasonal_recommendations(

    weekly_result,

    monthly_result,

    quarterly_result,

    seasonal_indices,

    decomposition,

    festival_result
):

    recommendations = []

    # --------------------------------------------------------
    # Weekly
    # --------------------------------------------------------

    peak_day = weekly_result.get(
        "peak_day"
    )

    if peak_day:

        recommendations.append(

            f"Prepare additional inventory and "
            f"operational capacity before {peak_day}, "
            f"which shows the highest average weekly demand."
        )

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    peak_month = monthly_result.get(
        "peak_month"
    )

    if peak_month:

        recommendations.append(

            f"Increase stock planning and supplier "
            f"readiness before {peak_month}, "
            f"the strongest seasonal demand month."
        )

    # --------------------------------------------------------
    # Seasonal strength
    # --------------------------------------------------------

    seasonal_strength = decomposition.get(
        "seasonal_strength_percentage"
    )

    if seasonal_strength is not None:

        if seasonal_strength >= 60:

            recommendations.append(

                "The dataset shows strong seasonal "
                "behaviour. Inventory and purchasing "
                "plans should explicitly account for "
                "recurring seasonal demand."
            )

        elif seasonal_strength >= 30:

            recommendations.append(

                "Moderate seasonality was detected. "
                "Seasonal adjustments can improve "
                "inventory and demand planning."
            )

        else:

            recommendations.append(

                "Seasonal variation appears relatively "
                "weak compared with other demand movements."
            )

    # --------------------------------------------------------
    # Seasonal indices
    # --------------------------------------------------------

    indices = seasonal_indices.get(
        "indices",
        []
    )

    high_season_months = [

        item["month"]

        for item in indices

        if item["seasonal_index"] >= 1.20
    ]

    if high_season_months:

        recommendations.append(

            "Several months show demand above the "
            "overall average. Consider increasing "
            "safety stock during these high-season periods."
        )

    # --------------------------------------------------------
    # Festivals
    # --------------------------------------------------------

    if festival_result.get(
        "available"
    ):

        events = festival_result.get(
            "events",
            []
        )

        if events:

            strongest_event = events[0]

            if strongest_event[
                "demand_lift_percentage"
            ] > 0:

                recommendations.append(

                    f"The {strongest_event['event']} "
                    f"period shows approximately "
                    f"{strongest_event['demand_lift_percentage']}% "
                    f"higher demand than the overall average. "
                    f"Consider increasing stock before the event."
                )

    if not recommendations:

        recommendations.append(

            "Continue monitoring demand patterns "
            "as additional historical data becomes available."
        )

    return recommendations


# ============================================================
# COMPLETE SEASONAL ML PIPELINE
# ============================================================

def run_seasonal_ml(dataset):

    result = {

        "success": False,

        "message": (
            "Seasonal analysis could not be completed."
        ),

        "columns": {},

        "weekly": {
            "available": False
        },

        "monthly": {
            "available": False
        },

        "quarterly": {
            "available": False
        },

        "seasonal_indices": {
            "available": False
        },

        "decomposition": {
            "available": False
        },

        "seasonal_cycles": {
            "available": False
        },

        "festival": {
            "available": False
        },

        "peak_periods": {},

        "recommendations": []
    }

    # ========================================================
    # PREPARE DATA
    # ========================================================

    preparation = prepare_seasonal_data(
        dataset
    )

    if not preparation["success"]:

        result["message"] = (
            preparation["message"]
        )

        return result

    df = preparation["data"]

    columns = preparation["columns"]

    result["columns"] = columns

    # ========================================================
    # DAILY TIME SERIES
    # ========================================================

    daily_series = create_daily_time_series(
        df
    )

    # ========================================================
    # WEEKLY
    # ========================================================

    try:

        result["weekly"] = (
            analyze_weekly_seasonality(
                df
            )
        )

    except Exception as error:

        result["weekly"] = {

            "available": False,

            "message": (
                f"Weekly analysis failed: {error}"
            )
        }

    # ========================================================
    # MONTHLY
    # ========================================================

    try:

        result["monthly"] = (
            analyze_monthly_seasonality(
                df
            )
        )

    except Exception as error:

        result["monthly"] = {

            "available": False,

            "message": (
                f"Monthly analysis failed: {error}"
            )
        }

    # ========================================================
    # QUARTERLY
    # ========================================================

    try:

        result["quarterly"] = (
            analyze_quarterly_seasonality(
                df
            )
        )

    except Exception as error:

        result["quarterly"] = {

            "available": False,

            "message": (
                f"Quarterly analysis failed: {error}"
            )
        }

    # ========================================================
    # SEASONAL INDICES
    # ========================================================

    try:

        result["seasonal_indices"] = (
            calculate_seasonal_indices(
                df
            )
        )

    except Exception as error:

        result["seasonal_indices"] = {

            "available": False,

            "message": (
                f"Seasonal index calculation failed: {error}"
            )
        }

    # ========================================================
    # STL DECOMPOSITION
    # ========================================================

    try:

        result["decomposition"] = (
            perform_stl_decomposition(
                daily_series
            )
        )

    except Exception as error:

        result["decomposition"] = {

            "available": False,

            "message": (
                f"STL analysis failed: {error}"
            )
        }

    # ========================================================
    # SEASONAL CYCLES
    # ========================================================

    try:

        result["seasonal_cycles"] = (
            detect_seasonal_cycles(
                daily_series
            )
        )

    except Exception as error:

        result["seasonal_cycles"] = {

            "available": False,

            "message": (
                f"Cycle detection failed: {error}"
            )
        }

    # ========================================================
    # FESTIVAL EFFECT
    # ========================================================

    try:

        result["festival"] = (
            analyze_festival_effect(
                df,
                columns
            )
        )

    except Exception as error:

        result["festival"] = {

            "available": False,

            "events": [],

            "message": (
                f"Festival analysis failed: {error}"
            )
        }

    # ========================================================
    # PEAK PERIODS
    # ========================================================

    try:

        result["peak_periods"] = (
            detect_peak_periods(
                df
            )
        )

    except Exception as error:

        result["peak_periods"] = {}

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    try:

        result["recommendations"] = (
            generate_seasonal_recommendations(

                result["weekly"],

                result["monthly"],

                result["quarterly"],

                result["seasonal_indices"],

                result["decomposition"],

                result["festival"]
            )
        )

    except Exception as error:

        result["recommendations"] = [

            (
                "Seasonal recommendations could "
                f"not be generated: {error}"
            )
        ]

    # ========================================================
    # SUCCESS
    # ========================================================

    result["success"] = True

    result["message"] = (

        "Seasonal machine learning and "
        "time-series analysis completed successfully."
    )

    return result