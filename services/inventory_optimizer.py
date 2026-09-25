# ============================================================
# INVENTORY OPTIMIZATION ENGINE
# Retail Demand Forecasting System
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# INVENTORY CALCULATION
# ============================================================

def calculate_inventory_metrics(
    df,
    product_column,
    demand_column,
    inventory_column,
    lead_time_column=None,
    product_value=None
):
    """
    Calculate inventory optimization metrics
    for a selected product.
    """

    # --------------------------------------------------------
    # Filter selected product
    # --------------------------------------------------------

    product_df = df.copy()

    if product_value is not None:

        product_df = product_df[
            product_df[product_column].astype(str)
            == str(product_value)
        ]

    if product_df.empty:

        return None


    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    product_df[demand_column] = pd.to_numeric(
        product_df[demand_column],
        errors="coerce"
    )

    product_df[inventory_column] = pd.to_numeric(
        product_df[inventory_column],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Average Daily Demand
    # --------------------------------------------------------

    average_daily_demand = (
        product_df[demand_column]
        .dropna()
        .mean()
    )

    if pd.isna(average_daily_demand):

        average_daily_demand = 0


    # --------------------------------------------------------
    # Current Stock
    # --------------------------------------------------------

    current_stock = (
        product_df[inventory_column]
        .dropna()
        .iloc[-1]
        if not product_df[inventory_column]
        .dropna()
        .empty
        else 0
    )


    # --------------------------------------------------------
    # Lead Time
    # --------------------------------------------------------

    if (
        lead_time_column
        and lead_time_column in product_df.columns
    ):

        product_df[lead_time_column] = pd.to_numeric(
            product_df[lead_time_column],
            errors="coerce"
        )

        lead_time = (
            product_df[lead_time_column]
            .dropna()
            .mean()
        )

    else:

        # Default lead time if dataset does not contain it
        lead_time = 7


    if pd.isna(lead_time):

        lead_time = 7


    # --------------------------------------------------------
    # Demand Standard Deviation
    # --------------------------------------------------------

    demand_std = (
        product_df[demand_column]
        .dropna()
        .std()
    )

    if pd.isna(demand_std):

        demand_std = 0


    # --------------------------------------------------------
    # Safety Stock
    # --------------------------------------------------------
    #
    # Safety Stock =
    # Demand Standard Deviation × sqrt(Lead Time)
    #
    # --------------------------------------------------------

    safety_stock = (
        demand_std
        * np.sqrt(lead_time)
    )


    # --------------------------------------------------------
    # Reorder Point
    # --------------------------------------------------------
    #
    # ROP =
    # Average Daily Demand × Lead Time
    # + Safety Stock
    #
    # --------------------------------------------------------

    reorder_point = (
        average_daily_demand
        * lead_time
        + safety_stock
    )


    # --------------------------------------------------------
    # Recommended Order Quantity
    # --------------------------------------------------------

    recommended_order = max(
        0,
        reorder_point - current_stock
    )


    # --------------------------------------------------------
    # Inventory Status
    # --------------------------------------------------------

    if current_stock <= 0:

        stock_status = "Out of Stock"

    elif current_stock < reorder_point:

        stock_status = "Reorder Required"

    elif current_stock > reorder_point * 2:

        stock_status = "Overstock"

    else:

        stock_status = "Healthy Stock"


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "product": str(product_value),

        "average_daily_demand": round(
            float(average_daily_demand),
            2
        ),

        "current_stock": round(
            float(current_stock),
            2
        ),

        "lead_time": round(
            float(lead_time),
            2
        ),

        "safety_stock": round(
            float(safety_stock),
            2
        ),

        "reorder_point": round(
            float(reorder_point),
            2
        ),

        "recommended_order": round(
            float(recommended_order),
            2
        ),

        "stock_status": stock_status
    }


# ============================================================
# INVENTORY OVERVIEW
# ============================================================

def calculate_inventory_overview(
    df,
    product_column,
    demand_column,
    inventory_column,
    lead_time_column=None
):
    """
    Calculate overall inventory statistics.
    """

    products = (
        df[product_column]
        .dropna()
        .astype(str)
        .unique()
    )


    total_products = len(products)

    low_stock = 0

    overstock = 0

    reorder_required = 0


    # --------------------------------------------------------
    # Analyze every product
    # --------------------------------------------------------

    for product in products:

        result = calculate_inventory_metrics(

            df,

            product_column,

            demand_column,

            inventory_column,

            lead_time_column,

            product

        )


        if result is None:

            continue


        status = result["stock_status"]


        if status == "Reorder Required":

            reorder_required += 1


        elif status == "Overstock":

            overstock += 1


        elif status == "Out of Stock":

            low_stock += 1


    return {

        "total_products": total_products,

        "low_stock": low_stock,

        "overstock": overstock,

        "reorder_required": reorder_required

    }