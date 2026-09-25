# ============================================================
# SMART ALERTS ML ENGINE
# Retail Demand Forecasting System
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# COLUMN UTILITIES
# ============================================================

def normalize_column_name(column):
    """
    Normalize a dataframe column name so that different naming
    styles can be detected consistently.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_alert_column(
    columns,
    candidates
):
    """
    Find the first matching column from a list of candidates.
    """

    normalized = {
        normalize_column_name(col): col
        for col in columns
    }

    # Exact match first
    for candidate in candidates:

        candidate_normalized = (
            normalize_column_name(candidate)
        )

        if candidate_normalized in normalized:

            return normalized[
                candidate_normalized
            ]

    # Partial match
    for column in columns:

        normalized_column = (
            normalize_column_name(column)
        )

        for candidate in candidates:

            normalized_candidate = (
                normalize_column_name(candidate)
            )

            if (
                normalized_candidate in
                normalized_column
            ):

                return column

    return None


# ============================================================
# AUTOMATIC COLUMN DETECTION
# ============================================================

def detect_alert_columns(dataset):

    columns = list(
        dataset.columns
    )

    return {

        "date":
            find_alert_column(
                columns,
                [
                    "date",
                    "ds",
                    "order_date",
                    "sales_date",
                    "transaction_date",
                    "invoice_date",
                    "timestamp"
                ]
            ),

        "product":
            find_alert_column(
                columns,
                [
                    "product",
                    "product_name",
                    "item",
                    "item_name",
                    "sku",
                    "sku_name",
                    "product_id"
                ]
            ),

        "category":
            find_alert_column(
                columns,
                [
                    "category",
                    "product_category",
                    "department",
                    "segment"
                ]
            ),

        "demand":
            find_alert_column(
                columns,
                [
                    "demand",
                    "demand_quantity",
                    "forecast_demand",
                    "predicted_demand",
                    "quantity",
                    "units_sold",
                    "units",
                    "sales_quantity"
                ]
            ),

        "sales":
            find_alert_column(
                columns,
                [
                    "sales",
                    "revenue",
                    "sales_amount",
                    "total_sales",
                    "amount"
                ]
            ),

        "inventory":
            find_alert_column(
                columns,
                [
                    "inventory",
                    "stock",
                    "stock_level",
                    "inventory_level",
                    "current_stock",
                    "available_stock",
                    "quantity_in_stock"
                ]
            ),

        "lead_time":
            find_alert_column(
                columns,
                [
                    "lead_time",
                    "lead_time_days",
                    "supplier_lead_time",
                    "delivery_days"
                ]
            ),

        "price":
            find_alert_column(
                columns,
                [
                    "price",
                    "unit_price",
                    "selling_price"
                ]
            ),

        "promotion":
            find_alert_column(
                columns,
                [
                    "promotion",
                    "discount",
                    "promo",
                    "promotion_flag"
                ]
            )
    }


# ============================================================
# NUMERIC SERIES
# ============================================================

def numeric_series(
    dataset,
    column
):

    if column is None:
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        dataset[column],
        errors="coerce"
    )


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# FORMAT NUMBER
# ============================================================

def format_number(value):

    value = safe_float(
        value
    )

    if abs(value) >= 1000000:

        return f"{value:,.0f}"

    if abs(value) >= 1000:

        return f"{value:,.1f}"

    if value.is_integer():

        return f"{int(value):,}"

    return f"{value:,.2f}"


# ============================================================
# FORMAT PERCENT
# ============================================================

def format_percent(value):

    value = safe_float(
        value
    )

    return f"{value:.1f}%"


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert(
    alert_type,
    severity,
    title,
    message,
    recommendation,
    product=None,
    metric=None,
    value=None,
    reason=None,
    why_triggered=None,
    evidence=None,
    actual_value=None,
    threshold=None,
    baseline=None,
    difference=None,
    date=None,
    business_impact=None
):
    """
    Create one standardized Smart Alert.

    The additional explanation fields are intentionally included
    so the frontend can clearly explain why the alert appeared.
    """

    alert = {

        "alert_type":
            alert_type,

        "type":
            alert_type,

        "severity":
            severity,

        "priority":
            severity,

        "title":
            title,

        "message":
            message,

        "description":
            message,

        "reason":
            reason or message,

        "why":
            why_triggered or reason or message,

        "why_triggered":
            why_triggered or reason or message,

        "explanation":
            why_triggered or reason or message,

        "recommendation":
            recommendation,

        "recommended_action":
            recommendation,

        "business_impact":
            business_impact or
            "This condition may affect inventory efficiency, "
            "sales performance, or customer availability.",

        "product":
            product,

        "metric":
            metric,

        "value":
            value,

        "actual_value":
            actual_value,

        "threshold":
            threshold,

        "baseline":
            baseline,

        "difference":
            difference,

        "date":
            date,

        "created_at":
            pd.Timestamp.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "status":
            "Active",

        "alert_id":
            None,

        "evidence":
            evidence or {}
    }

    return alert


# ============================================================
# DEMAND SPIKES
# ============================================================

def detect_demand_spikes(
    dataset,
    columns
):

    alerts = []

    demand_col = columns.get(
        "demand"
    )

    product_col = columns.get(
        "product"
    )

    date_col = columns.get(
        "date"
    )

    if demand_col is None:
        return alerts

    demand = numeric_series(
        dataset,
        demand_col
    )

    valid = demand.dropna()

    if len(valid) < 5:
        return alerts

    mean_demand = valid.mean()

    std_demand = valid.std()

    if pd.isna(std_demand):
        return alerts

    threshold = (
        mean_demand +
        (2 * std_demand)
    )

    spike_rows = dataset[
        demand >= threshold
    ].copy()

    # Avoid generating hundreds of duplicate alerts
    if len(spike_rows) > 20:

        spike_rows = (
            spike_rows
            .sort_values(
                by=demand_col,
                ascending=False
            )
            .head(20)
        )

    for _, row in spike_rows.iterrows():

        actual = safe_float(
            row[demand_col]
        )

        difference = (
            actual -
            threshold
        )

        percentage_above = (
            difference /
            threshold *
            100
            if threshold != 0
            else 0
        )

        product = (
            str(row[product_col])
            if product_col
            and pd.notna(row[product_col])
            else "Unknown Product"
        )

        date = (
            str(row[date_col])
            if date_col
            and pd.notna(row[date_col])
            else None
        )

        reason = (
            f"Demand of {format_number(actual)} "
            f"units exceeded the spike threshold of "
            f"{format_number(threshold)} units."
        )

        explanation = (
            f"The observed demand is "
            f"{format_percent(percentage_above)} "
            f"above the calculated normal-demand threshold. "
            f"The threshold is based on the dataset mean "
            f"({format_number(mean_demand)} units) plus "
            f"two standard deviations "
            f"({format_number(std_demand)} units)."
        )

        message = (
            f"{product} recorded unusually high demand "
            f"of {format_number(actual)} units."
        )

        recommendation = (
            "Review current demand and increase replenishment "
            "if the spike is expected to continue."
        )

        business_impact = (
            "A sustained demand spike can cause stockouts, "
            "lost sales, and reduced customer satisfaction "
            "if inventory is not replenished."
        )

        evidence = {

            "observed_demand":
                actual,

            "normal_mean":
                mean_demand,

            "standard_deviation":
                std_demand,

            "trigger_threshold":
                threshold,

            "difference_from_threshold":
                difference,

            "percentage_above_threshold":
                percentage_above
        }

        alerts.append(
            create_alert(

                alert_type=
                    "demand_spike",

                severity=
                    "high",

                title=
                    "Unusually High Demand",

                message=
                    message,

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "Demand",

                value=
                    actual,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    actual,

                threshold=
                    threshold,

                baseline=
                    mean_demand,

                difference=
                    difference,

                date=
                    date,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# DEMAND DROPS
# ============================================================

def detect_demand_drops(
    dataset,
    columns
):

    alerts = []

    demand_col = columns.get(
        "demand"
    )

    product_col = columns.get(
        "product"
    )

    date_col = columns.get(
        "date"
    )

    if demand_col is None:
        return alerts

    demand = numeric_series(
        dataset,
        demand_col
    )

    valid = demand.dropna()

    if len(valid) < 5:
        return alerts

    mean_demand = valid.mean()

    std_demand = valid.std()

    if pd.isna(std_demand):
        return alerts

    threshold = max(
        0,
        mean_demand -
        (2 * std_demand)
    )

    drop_rows = dataset[
        demand <= threshold
    ].copy()

    if len(drop_rows) > 20:

        drop_rows = (
            drop_rows
            .sort_values(
                by=demand_col,
                ascending=True
            )
            .head(20)
        )

    for _, row in drop_rows.iterrows():

        actual = safe_float(
            row[demand_col]
        )

        difference = (
            threshold -
            actual
        )

        percentage_below = (
            difference /
            mean_demand *
            100
            if mean_demand != 0
            else 0
        )

        product = (
            str(row[product_col])
            if product_col
            and pd.notna(row[product_col])
            else "Unknown Product"
        )

        date = (
            str(row[date_col])
            if date_col
            and pd.notna(row[date_col])
            else None
        )

        reason = (
            f"Demand fell to {format_number(actual)} units, "
            f"below the low-demand threshold of "
            f"{format_number(threshold)} units."
        )

        explanation = (
            f"The observed demand is significantly below "
            f"the dataset baseline of {format_number(mean_demand)} "
            f"units. The alert threshold is calculated as "
            f"the mean minus two standard deviations."
        )

        message = (
            f"{product} recorded unusually low demand "
            f"of {format_number(actual)} units."
        )

        recommendation = (
            "Review pricing, promotions, seasonality, and "
            "customer demand before placing additional stock."
        )

        business_impact = (
            "A sustained demand decline can result in excess "
            "inventory, slower stock movement, and increased "
            "holding costs."
        )

        evidence = {

            "observed_demand":
                actual,

            "normal_mean":
                mean_demand,

            "standard_deviation":
                std_demand,

            "trigger_threshold":
                threshold,

            "difference_from_threshold":
                difference,

            "percentage_below_mean":
                percentage_below
        }

        alerts.append(
            create_alert(

                alert_type=
                    "demand_drop",

                severity=
                    "medium",

                title=
                    "Unusually Low Demand",

                message=
                    message,

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "Demand",

                value=
                    actual,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    actual,

                threshold=
                    threshold,

                baseline=
                    mean_demand,

                difference=
                    difference,

                date=
                    date,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# STOCKOUT RISK
# ============================================================

def detect_stockout_risk(
    dataset,
    columns
):

    alerts = []

    inventory_col = columns.get(
        "inventory"
    )

    demand_col = columns.get(
        "demand"
    )

    product_col = columns.get(
        "product"
    )

    if (
        inventory_col is None
        or demand_col is None
    ):

        return alerts

    working = dataset.copy()

    working["_inventory_numeric"] = (
        pd.to_numeric(
            working[inventory_col],
            errors="coerce"
        )
    )

    working["_demand_numeric"] = (
        pd.to_numeric(
            working[demand_col],
            errors="coerce"
        )
    )

    working = working[
        working["_inventory_numeric"].notna()
        &
        working["_demand_numeric"].notna()
    ]

    if working.empty:
        return alerts

    if product_col:

        grouped = (
            working
            .groupby(
                product_col,
                dropna=False
            )
            .agg(
                avg_inventory=(
                    "_inventory_numeric",
                    "mean"
                ),
                avg_demand=(
                    "_demand_numeric",
                    "mean"
                )
            )
            .reset_index()
        )

    else:

        grouped = pd.DataFrame({

            "product":
                ["All Products"],

            "avg_inventory":
                [working["_inventory_numeric"].mean()],

            "avg_demand":
                [working["_demand_numeric"].mean()]
        })

        product_col = "product"

    for _, row in grouped.iterrows():

        inventory = safe_float(
            row["avg_inventory"]
        )

        demand = safe_float(
            row["avg_demand"]
        )

        product = (
            str(row[product_col])
            if pd.notna(row[product_col])
            else "Unknown Product"
        )

        if demand <= 0:
            continue

        coverage = (
            inventory /
            demand
        )

        if inventory <= 0:

            severity = "critical"

            threshold = 0

            reason = (
                f"{product} has average inventory of "
                f"{format_number(inventory)} units while "
                f"average demand is {format_number(demand)} "
                f"units."
            )

            explanation = (
                "Inventory has reached zero or below while "
                "positive demand is present. This means the "
                "product currently has no stock available to "
                "serve expected demand."
            )

            title = (
                "Immediate Stockout Risk"
            )

            recommendation = (
                "Replenish stock immediately and review "
                "supplier lead time."
            )

            business_impact = (
                "Immediate stockout can cause lost sales, "
                "unfulfilled customer orders, and poor "
                "customer satisfaction."
            )

        elif coverage < 1:

            severity = "critical"

            threshold = demand

            reason = (
                f"Available inventory of "
                f"{format_number(inventory)} units is lower "
                f"than the average demand of "
                f"{format_number(demand)} units."
            )

            explanation = (
                f"Current inventory covers only "
                f"{coverage:.2f} demand periods. "
                f"Because coverage is below 1, available "
                f"stock may not satisfy the next normal "
                f"demand period."
            )

            title = (
                "Critical Stockout Risk"
            )

            recommendation = (
                "Place an urgent replenishment order and "
                "verify supplier availability."
            )

            business_impact = (
                "The product may run out before the next "
                "replenishment cycle, creating lost-sales risk."
            )

        elif coverage < 2:

            severity = "high"

            threshold = (
                demand * 2
            )

            reason = (
                f"Inventory of {format_number(inventory)} "
                f"units provides only {coverage:.2f} "
                f"demand periods of coverage."
            )

            explanation = (
                f"The system expects approximately "
                f"{format_number(demand)} units of demand "
                f"per period, while only "
                f"{format_number(inventory)} units are "
                f"currently available. Inventory coverage "
                f"is below the 2-period safety level."
            )

            title = (
                "Low Inventory Coverage"
            )

            recommendation = (
                "Plan replenishment soon and monitor demand "
                "closely."
            )

            business_impact = (
                "Low inventory coverage increases the "
                "probability of stockout if demand increases "
                "or supplier delivery is delayed."
            )

        else:

            continue

        evidence = {

            "average_inventory":
                inventory,

            "average_demand":
                demand,

            "inventory_coverage":
                coverage,

            "coverage_threshold":
                threshold
        }

        alerts.append(
            create_alert(

                alert_type=
                    "stockout_risk",

                severity=
                    severity,

                title=
                    title,

                message=
                    f"{product} has "
                    f"{format_number(inventory)} units "
                    f"in inventory against "
                    f"{format_number(demand)} units "
                    f"average demand.",

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "Inventory Coverage",

                value=
                    coverage,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    inventory,

                threshold=
                    threshold,

                baseline=
                    demand,

                difference=
                    inventory - demand,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# OVERSTOCK
# ============================================================

def detect_overstock(
    dataset,
    columns
):

    alerts = []

    inventory_col = columns.get(
        "inventory"
    )

    demand_col = columns.get(
        "demand"
    )

    product_col = columns.get(
        "product"
    )

    if (
        inventory_col is None
        or demand_col is None
    ):

        return alerts

    working = dataset.copy()

    working["_inventory_numeric"] = (
        pd.to_numeric(
            working[inventory_col],
            errors="coerce"
        )
    )

    working["_demand_numeric"] = (
        pd.to_numeric(
            working[demand_col],
            errors="coerce"
        )
    )

    working = working[
        working["_inventory_numeric"].notna()
        &
        working["_demand_numeric"].notna()
    ]

    if working.empty:
        return alerts

    if product_col:

        grouped = (
            working
            .groupby(
                product_col,
                dropna=False
            )
            .agg(
                avg_inventory=(
                    "_inventory_numeric",
                    "mean"
                ),
                avg_demand=(
                    "_demand_numeric",
                    "mean"
                )
            )
            .reset_index()
        )

    else:

        grouped = pd.DataFrame({

            "product":
                ["All Products"],

            "avg_inventory":
                [working["_inventory_numeric"].mean()],

            "avg_demand":
                [working["_demand_numeric"].mean()]
        })

        product_col = "product"

    for _, row in grouped.iterrows():

        inventory = safe_float(
            row["avg_inventory"]
        )

        demand = safe_float(
            row["avg_demand"]
        )

        if demand <= 0:
            continue

        coverage = (
            inventory /
            demand
        )

        if coverage < 6:
            continue

        product = (
            str(row[product_col])
            if pd.notna(row[product_col])
            else "Unknown Product"
        )

        threshold = (
            demand * 6
        )

        excess = (
            inventory -
            threshold
        )

        reason = (
            f"{product} has inventory of "
            f"{format_number(inventory)} units, "
            f"providing approximately {coverage:.1f} "
            f"demand periods of coverage."
        )

        explanation = (
            f"The normal demand baseline is approximately "
            f"{format_number(demand)} units per period. "
            f"The current inventory exceeds the 6-period "
            f"coverage threshold of "
            f"{format_number(threshold)} units."
        )

        message = (
            f"{product} may be overstocked."
        )

        recommendation = (
            "Review replenishment quantities and consider "
            "reducing future orders or using promotions "
            "to improve stock movement."
        )

        business_impact = (
            "Excess stock increases storage costs, ties up "
            "working capital, and may increase the risk of "
            "obsolete or slow-moving inventory."
        )

        evidence = {

            "average_inventory":
                inventory,

            "average_demand":
                demand,

            "coverage_periods":
                coverage,

            "overstock_threshold":
                threshold,

            "excess_inventory":
                excess
        }

        alerts.append(
            create_alert(

                alert_type=
                    "overstock",

                severity=
                    "medium",

                title=
                    "Potential Overstock",

                message=
                    message,

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "Inventory Coverage",

                value=
                    coverage,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    inventory,

                threshold=
                    threshold,

                baseline=
                    demand,

                difference=
                    excess,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# DEMAND VOLATILITY
# ============================================================

def detect_demand_volatility(
    dataset,
    columns
):

    alerts = []

    demand_col = columns.get(
        "demand"
    )

    product_col = columns.get(
        "product"
    )

    if demand_col is None:
        return alerts

    working = dataset.copy()

    working["_demand_numeric"] = (
        pd.to_numeric(
            working[demand_col],
            errors="coerce"
        )
    )

    working = working[
        working["_demand_numeric"].notna()
    ]

    if len(working) < 10:
        return alerts

    if product_col:

        groups = (
            working
            .groupby(
                product_col,
                dropna=False
            )["_demand_numeric"]
        )

    else:

        groups = [
            (
                "All Products",
                working["_demand_numeric"]
            )
        ]

    for product, values in groups:

        values = values.dropna()

        if len(values) < 5:
            continue

        mean_demand = values.mean()

        std_demand = values.std()

        if mean_demand <= 0:
            continue

        cv = (
            std_demand /
            mean_demand
        )

        if cv >= 1.0:

            severity = "high"

            threshold = 1.0

            title = (
                "Highly Volatile Demand"
            )

            recommendation = (
                "Review forecasting assumptions, promotions, "
                "seasonality, and supplier flexibility."
            )

        elif cv >= 0.60:

            severity = "medium"

            threshold = 0.60

            title = (
                "Demand Volatility Detected"
            )

            recommendation = (
                "Monitor demand closely and consider a "
                "larger safety-stock buffer."
            )

        else:

            continue

        reason = (
            f"{product} has a demand coefficient of "
            f"variation of {cv:.2f}."
        )

        explanation = (
            f"Demand variability is high relative to its "
            f"average. The coefficient of variation is "
            f"calculated as standard deviation divided by "
            f"mean demand. The detected value of {cv:.2f} "
            f"exceeds the alert threshold of {threshold:.2f}."
        )

        message = (
            f"{product} shows unstable demand patterns."
        )

        business_impact = (
            "High demand variability makes inventory planning "
            "less predictable and can increase both stockout "
            "and overstock risk."
        )

        evidence = {

            "mean_demand":
                mean_demand,

            "standard_deviation":
                std_demand,

            "coefficient_of_variation":
                cv,

            "threshold":
                threshold
        }

        alerts.append(
            create_alert(

                alert_type=
                    "demand_volatility",

                severity=
                    severity,

                title=
                    title,

                message=
                    message,

                recommendation=
                    recommendation,

                product=
                    str(product),

                metric=
                    "Demand Volatility",

                value=
                    cv,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    cv,

                threshold=
                    threshold,

                baseline=
                    0,

                difference=
                    cv - threshold,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# LEAD TIME RISK
# ============================================================

def detect_lead_time_risk(
    dataset,
    columns
):

    alerts = []

    lead_col = columns.get(
        "lead_time"
    )

    product_col = columns.get(
        "product"
    )

    if lead_col is None:
        return alerts

    lead_time = numeric_series(
        dataset,
        lead_col
    )

    valid = lead_time.dropna()

    if valid.empty:
        return alerts

    threshold = 14

    risky = dataset[
        lead_time >= threshold
    ].copy()

    if len(risky) > 20:

        risky = (
            risky
            .sort_values(
                by=lead_col,
                ascending=False
            )
            .head(20)
        )

    for _, row in risky.iterrows():

        actual = safe_float(
            row[lead_col]
        )

        product = (
            str(row[product_col])
            if product_col
            and pd.notna(row[product_col])
            else "Unknown Product"
        )

        reason = (
            f"{product} has a supplier lead time of "
            f"{format_number(actual)} days, which meets "
            f"or exceeds the {threshold}-day risk threshold."
        )

        explanation = (
            "Long supplier lead times increase the time "
            "required to replenish inventory. When lead time "
            "is high, a demand increase can cause stockouts "
            "before new stock arrives."
        )

        recommendation = (
            "Increase safety-stock planning, monitor supplier "
            "performance, and consider alternative suppliers."
        )

        business_impact = (
            "Long replenishment times increase stockout risk "
            "and make demand changes harder to respond to."
        )

        evidence = {

            "lead_time_days":
                actual,

            "risk_threshold_days":
                threshold,

            "days_above_threshold":
                actual - threshold
        }

        alerts.append(
            create_alert(

                alert_type=
                    "lead_time_risk",

                severity=
                    "medium",

                title=
                    "Long Supplier Lead Time",

                message=
                    f"{product} has a supplier lead time "
                    f"of {format_number(actual)} days.",

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "Lead Time",

                value=
                    actual,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    actual,

                threshold=
                    threshold,

                baseline=
                    threshold,

                difference=
                    actual - threshold,

                business_impact=
                    business_impact
            )
        )

    return alerts


# ============================================================
# DATA QUALITY ALERTS
# ============================================================

def detect_data_quality_alerts(
    dataset
):

    alerts = []

    total_rows = len(
        dataset
    )

    if total_rows == 0:
        return alerts

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_ratio = (
        dataset.isna()
        .mean()
    )

    for column, ratio in missing_ratio.items():

        if ratio < 0.20:
            continue

        percentage = (
            ratio * 100
        )

        reason = (
            f"{format_percent(percentage)} of values in "
            f"'{column}' are missing."
        )

        explanation = (
            f"The column contains {format_percent(percentage)} "
            f"missing values, exceeding the 20% data-quality "
            f"threshold. Missing values can reduce the reliability "
            f"of forecasting and alert calculations."
        )

        recommendation = (
            "Review the source data and fill, remove, or "
            "correct missing values before relying on the "
            "affected metric."
        )

        evidence = {

            "column":
                column,

            "missing_percentage":
                percentage,

            "threshold_percentage":
                20
        }

        alerts.append(
            create_alert(

                alert_type=
                    "data_quality",

                severity=
                    "medium",

                title=
                    "High Missing Data",

                message=
                    f"Column '{column}' contains "
                    f"{format_percent(percentage)} "
                    f"missing values.",

                recommendation=
                    recommendation,

                product=
                    None,

                metric=
                    str(column),

                value=
                    percentage,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    percentage,

                threshold=
                    20,

                baseline=
                    0,

                difference=
                    percentage - 20,

                business_impact=
                    "Incomplete data can reduce the accuracy "
                    "of demand forecasting and inventory decisions."
            )
        )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = int(
        dataset.duplicated().sum()
    )

    duplicate_ratio = (
        duplicate_count /
        total_rows
        if total_rows > 0
        else 0
    )

    if duplicate_ratio >= 0.05:

        percentage = (
            duplicate_ratio * 100
        )

        reason = (
            f"{duplicate_count:,} duplicate rows were found, "
            f"representing {format_percent(percentage)} "
            f"of the dataset."
        )

        explanation = (
            f"Duplicate rows exceed the 5% data-quality "
            f"threshold. Duplicate transactions can inflate "
            f"demand or sales measurements and therefore "
            f"distort forecasting and alert calculations."
        )

        recommendation = (
            "Review duplicate records and remove confirmed "
            "duplicates before using the dataset for decisions."
        )

        evidence = {

            "duplicate_rows":
                duplicate_count,

            "duplicate_percentage":
                percentage,

            "threshold_percentage":
                5
        }

        alerts.append(
            create_alert(

                alert_type=
                    "data_quality",

                severity=
                    "medium",

                title=
                    "Duplicate Records Detected",

                message=
                    f"{duplicate_count:,} duplicate rows "
                    f"were detected.",

                recommendation=
                    recommendation,

                metric=
                    "Duplicate Records",

                value=
                    duplicate_count,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    percentage,

                threshold=
                    5,

                baseline=
                    0,

                difference=
                    percentage - 5,

                business_impact=
                    "Duplicate transactions can artificially "
                    "increase demand and sales values."
            )
        )

    return alerts


# ============================================================
# ISOLATION FOREST ML ANOMALIES
# ============================================================
def detect_ml_anomalies(
    dataset,
    columns
):
    """
    Detect unusual combinations of numerical retail metrics
    using Isolation Forest.

    IMPORTANT:
    The original dataset rows are preserved so product_id,
    store_id and date remain available when explaining
    detected anomalies.
    """

    alerts = []

    # ========================================================
    # IDENTIFY ML FEATURES
    # ========================================================

    feature_columns = []

    candidate_features = [
        "demand",
        "inventory",
        "sales",
        "price"
    ]

    for feature_name in candidate_features:

        column = columns.get(
            feature_name
        )

        if column is not None:

            numeric = pd.to_numeric(
                dataset[column],
                errors="coerce"
            )

            if numeric.notna().sum() >= 10:

                feature_columns.append(
                    column
                )

    # Remove duplicate feature columns
    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    # ========================================================
    # NO FEATURES
    # ========================================================

    if len(feature_columns) == 0:

        return alerts, {

            "method":
                "Isolation Forest",

            "anomalies_detected":
                0,

            "anomaly_rate":
                0,

            "records_analyzed":
                0,

            "features_used":
                [],

            "contamination":
                None,

            "status":
                "Insufficient Features"
        }

    # ========================================================
    # PREPARE NUMERIC ML DATA
    # ========================================================

    ml_data = dataset[
        feature_columns
    ].copy()

    for column in feature_columns:

        ml_data[column] = pd.to_numeric(
            ml_data[column],
            errors="coerce"
        )

    # Only rows with complete ML features can be analyzed
    valid_mask = (
        ml_data.notna()
        .all(axis=1)
    )

    if valid_mask.sum() < 20:

        return alerts, {

            "method":
                "Isolation Forest",

            "anomalies_detected":
                0,

            "anomaly_rate":
                0,

            "records_analyzed":
                int(valid_mask.sum()),

            "features_used":
                feature_columns,

            "contamination":
                None,

            "status":
                "Insufficient Records"
        }

    # ========================================================
    # IMPORTANT FIX
    #
    # Preserve the ORIGINAL dataframe rows.
    #
    # This keeps:
    # product_id
    # store_id
    # date
    # current_stock
    # daily_demand
    # etc.
    #
    # The old code used ml_data here and therefore removed
    # product_id/date/store_id.
    # ========================================================

    valid_data = dataset.loc[
        valid_mask
    ].copy()

    # ========================================================
    # CREATE ML MATRIX
    # ========================================================

    X = valid_data[
        feature_columns
    ].copy()

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ========================================================
    # STANDARDIZE FEATURES
    # ========================================================

    for column in X.columns:

        mean = X[column].mean()

        std = X[column].std()

        if (
            pd.isna(std)
            or std == 0
        ):

            X[column] = 0.0

        else:

            X[column] = (
                X[column] - mean
            ) / std

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    contamination = min(
        max(
            0.02,
            1 / len(X)
        ),
        0.08
    )

    model = IsolationForest(

        n_estimators=200,

        contamination=contamination,

        random_state=42,

        n_jobs=-1
    )

    predictions = model.fit_predict(
        X
    )

    anomaly_scores = (
        -model.decision_function(X)
    )

    # ========================================================
    # STORE ML RESULTS ON ORIGINAL ROWS
    # ========================================================

    valid_data = valid_data.copy()

    valid_data[
        "_anomaly_prediction"
    ] = predictions

    valid_data[
        "_anomaly_score"
    ] = anomaly_scores

    # ========================================================
    # FIND ANOMALIES
    # ========================================================

    anomaly_rows = valid_data[
        valid_data[
            "_anomaly_prediction"
        ] == -1
    ].copy()

    anomaly_count = len(
        anomaly_rows
    )

    anomaly_rate = (
        anomaly_count /
        len(valid_data) *
        100
        if len(valid_data) > 0
        else 0
    )

    # ========================================================
    # IDENTIFY METADATA COLUMNS
    # ========================================================

    product_col = columns.get(
        "product"
    )

    date_col = columns.get(
        "date"
    )

    # Optional store column
    store_col = find_alert_column(
        dataset.columns,
        [
            "store_id",
            "store",
            "store_name",
            "location",
            "branch"
        ]
    )

    # ========================================================
    # KEEP MOST UNUSUAL ANOMALIES
    # ========================================================

    anomaly_rows = (
        anomaly_rows
        .sort_values(
            "_anomaly_score",
            ascending=False
        )
        .head(10)
    )

    # ========================================================
    # EXPLAIN EACH ANOMALY
    # ========================================================

    for _, row in anomaly_rows.iterrows():

        # ----------------------------------------------------
        # PRODUCT
        # ----------------------------------------------------

        if (
            product_col is not None
            and product_col in row.index
            and pd.notna(row[product_col])
        ):

            product = str(
                row[product_col]
            )

        else:

            product = "Unknown Product"

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if (
            date_col is not None
            and date_col in row.index
            and pd.notna(row[date_col])
        ):

            date = str(
                row[date_col]
            )

        else:

            date = None

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        if (
            store_col is not None
            and store_col in row.index
            and pd.notna(row[store_col])
        ):

            store_id = str(
                row[store_col]
            )

        else:

            store_id = None

        # ----------------------------------------------------
        # ANOMALY SCORE
        # ----------------------------------------------------

        score = safe_float(
            row["_anomaly_score"]
        )

        # ====================================================
        # FEATURE ANALYSIS
        # ====================================================

        unusual_features = []

        feature_evidence = {}

        for column in feature_columns:

            actual = safe_float(
                row[column]
            )

            feature_mean = safe_float(
                valid_data[column].mean()
            )

            feature_std = safe_float(
                valid_data[column].std()
            )

            if feature_std > 0:

                z_score = (
                    actual -
                    feature_mean
                ) / feature_std

            else:

                z_score = 0.0

            # ------------------------------------------------
            # Percentage deviation
            # ------------------------------------------------

            if feature_mean != 0:

                deviation_percent = (
                    (
                        actual -
                        feature_mean
                    )
                    /
                    abs(feature_mean)
                ) * 100

            else:

                deviation_percent = 0.0

            feature_evidence[
                str(column)
            ] = {

                "actual":
                    actual,

                "baseline":
                    feature_mean,

                "standard_deviation":
                    feature_std,

                "z_score":
                    round(
                        z_score,
                        3
                    ),

                "deviation_percent":
                    round(
                        deviation_percent,
                        2
                    )
            }

            # ------------------------------------------------
            # Identify unusual individual features
            # ------------------------------------------------

            if abs(z_score) >= 2:

                direction = (
                    "higher"
                    if z_score > 0
                    else "lower"
                )

                unusual_features.append(
                    (
                        f"{column} is {direction} "
                        f"than its normal level "
                        f"({z_score:+.2f} standard deviations)"
                    )
                )

        # ====================================================
        # BUILD EXPLANATION
        # ====================================================

        if unusual_features:

            feature_text = (
                "; ".join(
                    unusual_features[:3]
                )
            )

            explanation = (
                f"Isolation Forest identified this record "
                f"as an unusual combination of retail "
                f"metrics. The strongest individual deviations "
                f"were: {feature_text}. "
                f"The ML model considers the combined pattern "
                f"unusual compared with the normal records "
                f"in the dataset."
            )

        else:

            explanation = (
                "Isolation Forest identified this record as "
                "an unusual combination of the available "
                "retail metrics. No single metric crossed "
                "the individual 2-standard-deviation "
                "explanation threshold, but the combination "
                "of values differs from the normal patterns "
                "learned by the Isolation Forest model."
            )

        # ====================================================
        # MESSAGE
        # ====================================================

        message = (
            f"{product} contains an AI-detected unusual "
            f"combination of "
            f"{', '.join(feature_columns)}."
        )

        # ====================================================
        # REASON
        # ====================================================

        reason = (
            f"Isolation Forest classified the record for "
            f"{product} as an anomaly. "
            f"The calculated anomaly score is "
            f"{score:.4f}."
        )

        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommendation = (
            "Review the demand and inventory values for this "
            "product and verify whether the unusual pattern "
            "is caused by a genuine business event, promotion, "
            "sudden demand change, stock issue, or data-quality "
            "problem."
        )

        # ====================================================
        # BUSINESS IMPACT
        # ====================================================

        business_impact = (
            "An unusual combination of demand and inventory "
            "may indicate an emerging stockout, overstock, "
            "demand shock, operational issue, or abnormal "
            "business event that may require investigation."
        )

        # ====================================================
        # EVIDENCE
        # ====================================================

        evidence = {

            "anomaly_score":
                score,

            "features_used":
                feature_columns,

            "feature_analysis":
                feature_evidence,

            "unusual_features":
                unusual_features,

            "product_id":
                product,

            "store_id":
                store_id,

            "date":
                date
        }

        # ====================================================
        # CREATE ALERT
        # ====================================================

        alert = create_alert(

            alert_type=
                "ml_anomaly",

            severity=
                "medium",

            title=
                "AI Detected Unusual Pattern",

            message=
                message,

            recommendation=
                recommendation,

            product=
                product,

            metric=
                "ML Anomaly Score",

            value=
                score,

            reason=
                reason,

            why_triggered=
                explanation,

            evidence=
                evidence,

            actual_value=
                score,

            threshold=
                None,

            baseline=
                0,

            difference=
                score,

            date=
                date,

            business_impact=
                business_impact
        )

        # ----------------------------------------------------
        # Notification compatibility
        # ----------------------------------------------------

        alert["product_id"] = product

        alert["store_id"] = store_id

        alerts.append(
            alert
        )

    # ========================================================
    # ML INFORMATION
    # ========================================================

    ml_info = {

        "method":
            "Isolation Forest",

        "anomalies_detected":
            anomaly_count,

        "anomaly_rate":
            round(
                anomaly_rate,
                2
            ),

        "records_analyzed":
            len(valid_data),

        "features_used":
            feature_columns,

        "contamination":
            round(
                contamination,
                4
            ),

        "status":
            "Completed"
    }

    return alerts, ml_info
    # ========================================================
    # FILL EXTREME VALUES SAFELY
    # ========================================================

    X = valid_data.copy()

    # Standardization is useful because different retail
    # metrics may have completely different scales.

    for column in X.columns:

        mean = X[column].mean()

        std = X[column].std()

        if (
            pd.isna(std)
            or std == 0
        ):

            X[column] = 0

        else:

            X[column] = (
                X[column] - mean
            ) / std


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    contamination = min(
        max(
            0.02,
            1 / len(X)
        ),
        0.08
    )

    model = IsolationForest(

        n_estimators=200,

        contamination=contamination,

        random_state=42,

        n_jobs=-1
    )


    predictions = model.fit_predict(
        X
    )

    anomaly_scores = (
        -model.decision_function(X)
    )


    valid_data = valid_data.copy()

    valid_data[
        "_anomaly_prediction"
    ] = predictions

    valid_data[
        "_anomaly_score"
    ] = anomaly_scores


    anomaly_rows = valid_data[
        valid_data[
            "_anomaly_prediction"
        ] == -1
    ].copy()


    anomaly_count = len(
        anomaly_rows
    )

    anomaly_rate = (
        anomaly_count /
        len(valid_data) *
        100
        if len(valid_data) > 0
        else 0
    )


    # ========================================================
    # EXPLAIN ANOMALIES
    # ========================================================

    product_col = columns.get(
        "product"
    )

    date_col = columns.get(
        "date"
    )


    # Keep the most unusual anomalies first
    anomaly_rows = (
        anomaly_rows
        .sort_values(
            "_anomaly_score",
            ascending=False
        )
        .head(10)
    )


    for _, row in anomaly_rows.iterrows():

        product = (
            str(row[product_col])
            if product_col
            and pd.notna(row[product_col])
            else "Unknown Product"
        )

        date = (
            str(row[date_col])
            if date_col
            and pd.notna(row[date_col])
            else None
        )

        score = safe_float(
            row["_anomaly_score"]
        )


        # ----------------------------------------------------
        # FIND WHICH FEATURES ARE UNUSUAL
        # ----------------------------------------------------

        unusual_features = []

        feature_evidence = {}

        for column in feature_columns:

            actual = safe_float(
                row[column]
            )

            feature_mean = safe_float(
                valid_data[column].mean()
            )

            feature_std = safe_float(
                valid_data[column].std()
            )

            if feature_std > 0:

                z_score = (
                    actual -
                    feature_mean
                ) / feature_std

            else:

                z_score = 0


            feature_evidence[
                str(column)
            ] = {

                "actual":
                    actual,

                "baseline":
                    feature_mean,

                "standard_deviation":
                    feature_std,

                "z_score":
                    z_score
            }


            if abs(z_score) >= 2:

                direction = (
                    "higher"
                    if z_score > 0
                    else "lower"
                )

                unusual_features.append(
                    (
                        f"{column} is {direction} "
                        f"than its normal level "
                        f"({z_score:+.2f} standard deviations)"
                    )
                )


        # ----------------------------------------------------
        # IF NO SINGLE FEATURE IS EXTREME
        # ----------------------------------------------------

        if unusual_features:

            feature_text = (
                "; ".join(
                    unusual_features[:3]
                )
            )

            explanation = (
                f"Isolation Forest identified this record "
                f"as an unusual combination of retail metrics. "
                f"Specific unusual measurements include: "
                f"{feature_text}."
            )

        else:

            explanation = (
                "Isolation Forest identified this record as "
                "an unusual combination of the available "
                "retail metrics. No single metric necessarily "
                "crossed a fixed threshold, but the combination "
                "of values differs significantly from the "
                "normal patterns in the dataset."
            )


        # ----------------------------------------------------
        # BUILD MESSAGE
        # ----------------------------------------------------

        message = (
            f"{product} contains an unusual combination "
            f"of {', '.join(feature_columns)}."
        )


        reason = (
            f"Isolation Forest classified this record as "
            f"an anomaly with an anomaly score of "
            f"{score:.4f}."
        )


        recommendation = (
            "Review the underlying demand, inventory, sales, "
            "and price values for this record. Confirm whether "
            "the unusual pattern is caused by a genuine business "
            "event, promotion, data issue, or sudden demand change."
        )


        business_impact = (
            "Unusual combinations of demand, inventory, sales, "
            "or price may indicate an emerging stockout, "
            "overstock, demand shock, pricing issue, or data "
            "quality problem."
        )


        evidence = {

            "anomaly_score":
                score,

            "features_used":
                feature_columns,

            "feature_analysis":
                feature_evidence,

            "unusual_features":
                unusual_features
        }


        alerts.append(
            create_alert(

                alert_type=
                    "ml_anomaly",

                severity=
                    "medium",

                title=
                    "AI Detected Unusual Pattern",

                message=
                    message,

                recommendation=
                    recommendation,

                product=
                    product,

                metric=
                    "ML Anomaly Score",

                value=
                    score,

                reason=
                    reason,

                why_triggered=
                    explanation,

                evidence=
                    evidence,

                actual_value=
                    score,

                threshold=
                    None,

                baseline=
                    0,

                difference=
                    score,

                date=
                    date,

                business_impact=
                    business_impact
            )
        )


    ml_info = {

        "method":
            "Isolation Forest",

        "anomalies_detected":
            anomaly_count,

        "anomaly_rate":
            round(
                anomaly_rate,
                2
            ),

        "records_analyzed":
            len(valid_data),

        "features_used":
            feature_columns,

        "contamination":
            round(
                contamination,
                4
            ),

        "status":
            "Completed"
    }

    return alerts, ml_info


# ============================================================
# ALERT SUMMARY
# ============================================================

def calculate_alert_summary(
    alerts
):

    summary = {

        "total":
            len(alerts),

        "critical":
            0,

        "high":
            0,

        "medium":
            0,

        "low":
            0,

        "info":
            0
    }


    for alert in alerts:

        severity = (
            str(
                alert.get(
                    "severity",
                    "info"
                )
            )
            .lower()
        )

        if severity not in summary:

            severity = "info"

        summary[
            severity
        ] += 1

    return summary


# ============================================================
# PRIORITY SCORE
# ============================================================

def calculate_priority_score(
    alert
):

    severity = (
        str(
            alert.get(
                "severity",
                "info"
            )
        )
        .lower()
    )

    scores = {

        "critical":
            100,

        "high":
            75,

        "medium":
            50,

        "low":
            25,

        "info":
            10
    }

    return scores.get(
        severity,
        10
    )


# ============================================================
# SORT ALERTS
# ============================================================

def sort_alerts(
    alerts
):

    return sorted(

        alerts,

        key=calculate_priority_score,

        reverse=True
    )


# ============================================================
# REMOVE DUPLICATE ALERTS
# ============================================================

def remove_duplicate_alerts(
    alerts
):

    unique = []

    seen = set()

    for alert in alerts:

        key = (

            str(
                alert.get(
                    "alert_type",
                    ""
                )
            ),

            str(
                alert.get(
                    "severity",
                    ""
                )
            ),

            str(
                alert.get(
                    "product",
                    ""
                )
            ),

            str(
                alert.get(
                    "metric",
                    ""
                )
            ),

            str(
                alert.get(
                    "date",
                    ""
                )
            )
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            alert
        )

    return unique


# ============================================================
# GENERATE HUMAN-READABLE SUMMARY
# ============================================================

def generate_alert_summary_message(
    summary,
    columns,
    dataset
):

    total = summary.get(
        "total",
        0
    )

    critical = summary.get(
        "critical",
        0
    )

    high = summary.get(
        "high",
        0
    )

    medium = summary.get(
        "medium",
        0
    )

    low = summary.get(
        "low",
        0
    )


    if total == 0:

        available_metrics = [
            key
            for key, value
            in columns.items()
            if value is not None
        ]

        if available_metrics:

            return (
                "Smart Alerts completed successfully. "
                "No alert thresholds were exceeded in the "
                "available dataset. "
                f"Metrics analyzed: "
                f"{', '.join(available_metrics)}."
            )

        return (
            "Smart Alerts completed successfully, but "
            "no supported retail metrics were detected."
        )


    parts = []

    if critical:
        parts.append(
            f"{critical} critical"
        )

    if high:
        parts.append(
            f"{high} high"
        )

    if medium:
        parts.append(
            f"{medium} medium"
        )

    if low:
        parts.append(
            f"{low} low"
        )


    severity_text = (
        ", ".join(parts)
    )


    return (
        f"Smart Alerts detected {total} active alert"
        f"{'s' if total != 1 else ''}: "
        f"{severity_text}. "
        "Review the explanation and recommended action "
        "shown for each alert."
    )


# ============================================================
# MAIN SMART ALERT ENGINE
# ============================================================

def run_alerts_ml(
    dataset
):
    """
    Main entry point used by app.py.

    Returns both the modern nested structure and compatibility
    top-level keys expected by the Flask route.
    """

    # ========================================================
    # VALIDATE DATASET
    # ========================================================

    if dataset is None:

        return {

            "success":
                False,

            "message":
                "No dataset was provided.",

            "alerts":
                [],

            "total_alerts":
                0,

            "critical_alerts":
                0,

            "high_alerts":
                0,

            "medium_alerts":
                0,

            "low_alerts":
                0,

            "summary":
                calculate_alert_summary([]),

            "insights":
                [],

            "metrics":
                {},

            "ml":
                {

                    "method":
                        "Isolation Forest",

                    "anomalies_detected":
                        0,

                    "anomaly_rate":
                        0,

                    "records_analyzed":
                        0,

                    "features_used":
                        [],

                    "contamination":
                        None,

                    "status":
                        "No Dataset"
                },

            "analysis":
                {},

            "diagnostics":
                {

                    "reason":
                        "Dataset is None."
                }
        }


    if not isinstance(
        dataset,
        pd.DataFrame
    ):

        dataset = pd.DataFrame(
            dataset
        )


    if dataset.empty:

        return {

            "success":
                False,

            "message":
                "The dataset is empty.",

            "alerts":
                [],

            "total_alerts":
                0,

            "critical_alerts":
                0,

            "high_alerts":
                0,

            "medium_alerts":
                0,

            "low_alerts":
                0,

            "summary":
                calculate_alert_summary([]),

            "insights":
                [],

            "metrics":
                {},

            "ml":
                {

                    "method":
                        "Isolation Forest",

                    "anomalies_detected":
                        0,

                    "anomaly_rate":
                        0,

                    "records_analyzed":
                        0,

                    "features_used":
                        [],

                    "contamination":
                        None,

                    "status":
                        "Empty Dataset"
                },

            "analysis":
                {},

            "diagnostics":
                {

                    "reason":
                        "Dataset contains zero rows."
                }
        }


    # ========================================================
    # COPY DATASET
    # ========================================================

    df = dataset.copy()


    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    columns = detect_alert_columns(
        df
    )


    print()
    print("=" * 80)
    print("SMART ALERTS ENGINE")
    print("=" * 80)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )

    print(
        "Detected Columns:",
        columns
    )

    print("=" * 80)


    # ========================================================
    # RUN RULE-BASED DETECTION
    # ========================================================

    all_alerts = []


    # Demand spikes
    all_alerts.extend(
        detect_demand_spikes(
            df,
            columns
        )
    )


    # Demand drops
    all_alerts.extend(
        detect_demand_drops(
            df,
            columns
        )
    )


    # Stockout risk
    all_alerts.extend(
        detect_stockout_risk(
            df,
            columns
        )
    )


    # Overstock
    all_alerts.extend(
        detect_overstock(
            df,
            columns
        )
    )


    # Demand volatility
    all_alerts.extend(
        detect_demand_volatility(
            df,
            columns
        )
    )


    # Lead time
    all_alerts.extend(
        detect_lead_time_risk(
            df,
            columns
        )
    )


    # Data quality
    all_alerts.extend(
        detect_data_quality_alerts(
            df
        )
    )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    ml_alerts, ml_info = (
        detect_ml_anomalies(
            df,
            columns
        )
    )


    all_alerts.extend(
        ml_alerts
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    all_alerts = (
        remove_duplicate_alerts(
            all_alerts
        )
    )


    # ========================================================
    # SORT
    # ========================================================

    all_alerts = (
        sort_alerts(
            all_alerts
        )
    )


    # ========================================================
    # ASSIGN ALERT IDS
    # ========================================================

    for index, alert in enumerate(
        all_alerts,
        start=1
    ):

        alert[
            "alert_id"
        ] = f"SA-{index:04d}"


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = calculate_alert_summary(
        all_alerts
    )


    total_alerts = summary.get(
        "total",
        0
    )

    critical_alerts = summary.get(
        "critical",
        0
    )

    high_alerts = summary.get(
        "high",
        0
    )

    medium_alerts = summary.get(
        "medium",
        0
    )

    low_alerts = summary.get(
        "low",
        0
    )


    # ========================================================
    # ANALYSIS COUNTS
    # ========================================================

    analysis = {

        "demand_spikes":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "demand_spike"
            ),

        "demand_drops":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "demand_drop"
            ),

        "stockout_risks":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "stockout_risk"
            ),

        "overstock_risks":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "overstock"
            ),

        "volatility_risks":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "demand_volatility"
            ),

        "anomalies":
            len(
                ml_alerts
            ),

        "lead_time_risks":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "lead_time_risk"
            ),

        "data_quality_alerts":
            sum(
                1
                for a in all_alerts
                if a.get(
                    "alert_type"
                ) == "data_quality"
            )
    }


    # ========================================================
    # DATASET METRICS
    # ========================================================

    metrics = {

        "records":
            len(df),

        "columns":
            len(df.columns),

        "missing_values":
            int(
                df.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),

        "date_column":
            columns.get(
                "date"
            ),

        "product_column":
            columns.get(
                "product"
            ),

        "demand_column":
            columns.get(
                "demand"
            ),

        "inventory_column":
            columns.get(
                "inventory"
            )
    }


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "dataset_rows":
            len(df),

        "dataset_columns":
            len(df.columns),

        "detected_columns":
            columns,

        "rules_executed":
            [
                "Demand Spike",
                "Demand Drop",
                "Stockout Risk",
                "Overstock Risk",
                "Demand Volatility",
                "Lead Time Risk",
                "Data Quality",
                "Isolation Forest"
            ],

        "rule_alerts":
            len(all_alerts) -
            len(ml_alerts),

        "ml_alerts":
            len(ml_alerts)
    }


    # ========================================================
    # INSIGHTS
    # ========================================================

    insights = []


    if critical_alerts > 0:

        insights.append(
            (
                f"{critical_alerts} critical alert"
                f"{'s' if critical_alerts != 1 else ''} "
                "require immediate attention."
            )
        )


    if high_alerts > 0:

        insights.append(
            (
                f"{high_alerts} high-priority alert"
                f"{'s' if high_alerts != 1 else ''} "
                "indicate significant demand or inventory risk."
            )
        )


    if ml_info.get(
        "anomalies_detected",
        0
    ) > 0:

        insights.append(
            (
                f"Isolation Forest detected "
                f"{ml_info['anomalies_detected']} unusual "
                f"record"
                f"{'s' if ml_info['anomalies_detected'] != 1 else ''} "
                f"({ml_info['anomaly_rate']}% of analyzed records)."
            )
        )


    if analysis[
        "stockout_risks"
    ] > 0:

        insights.append(
            (
                "Inventory coverage indicates products "
                "that may require replenishment."
            )
        )


    if analysis[
        "overstock_risks"
    ] > 0:

        insights.append(
            (
                "Some products have inventory coverage "
                "above the configured overstock threshold."
            )
        )


    if analysis.get(
        "volatility_risks",
        0
    ) > 0:

        insights.append(
            (
                "Demand volatility was detected in one or "
                "more product groups."
            )
        )


    if not insights:

        insights.append(
            (
                "No significant alert conditions were "
                "detected using the configured rules and "
                "Isolation Forest."
            )
        )


    # ========================================================
    # SUMMARY MESSAGE
    # ========================================================

    summary_message = (
        generate_alert_summary_message(
            summary,
            columns,
            df
        )
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "success":
            True,

        "message":
            summary_message,

        "columns":
            columns,

        "alerts":
            all_alerts,

        # ----------------------------------------------------
        # TOP LEVEL COMPATIBILITY KEYS
        # ----------------------------------------------------

        "total_alerts":
            total_alerts,

        "critical_alerts":
            critical_alerts,

        "high_alerts":
            high_alerts,

        "medium_alerts":
            medium_alerts,

        "low_alerts":
            low_alerts,

        # ----------------------------------------------------
        # NESTED SUMMARY
        # ----------------------------------------------------

        "summary":
            summary,

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        "analysis":
            analysis,

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        "ml":
            ml_info,

        # ----------------------------------------------------
        # INSIGHTS
        # ----------------------------------------------------

        "insights":
            insights,

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        "metrics":
            metrics,

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        "diagnostics":
            diagnostics,

        # ----------------------------------------------------
        # MESSAGE SUMMARY
        # ----------------------------------------------------

        "message_summary":
            summary_message
    }


    # ========================================================
    # PRINT FINAL DEBUG INFORMATION
    # ========================================================

    print()
    print("=" * 80)
    print("SMART ALERTS FINAL RESULT")
    print("=" * 80)

    print(
        "Total:",
        total_alerts
    )

    print(
        "Critical:",
        critical_alerts
    )

    print(
        "High:",
        high_alerts
    )

    print(
        "Medium:",
        medium_alerts
    )

    print(
        "Low:",
        low_alerts
    )

    print(
        "ML Anomalies:",
        ml_info.get(
            "anomalies_detected",
            0
        )
    )

    print(
        "ML Anomaly Rate:",
        ml_info.get(
            "anomaly_rate",
            0
        ),
        "%"
    )

    print(
        "Final Alert List:",
        len(all_alerts)
    )

    print("=" * 80)


    return result