# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# ANALYTICS MACHINE LEARNING SERVICE
# ============================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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


def find_column(df, candidates):

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # Exact matching
    for candidate in candidates:

        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    # Partial matching
    for normalized_name, original_name in normalized_columns.items():

        for candidate in candidates:

            normalized_candidate = normalize_column_name(candidate)

            if (
                normalized_candidate in normalized_name
                or normalized_name in normalized_candidate
            ):
                return original_name

    return None


def detect_analytics_columns(df):

    return {

        "date": find_column(df, [
            "date",
            "order_date",
            "sale_date",
            "sales_date",
            "transaction_date"
        ]),

        "product": find_column(df, [
            "product",
            "product_name",
            "product_id",
            "sku",
            "item",
            "item_id"
        ]),

        "demand": find_column(df, [
            "demand",
            "quantity",
            "qty",
            "units",
            "units_sold",
            "sales_quantity",
            "quantity_sold",
            "sold_quantity"
        ]),

        "sales": find_column(df, [
            "sales",
            "sales_amount",
            "revenue",
            "total_sales",
            "amount"
        ]),

        "price": find_column(df, [
            "price",
            "unit_price",
            "selling_price",
            "product_price"
        ]),

        "discount": find_column(df, [
            "discount",
            "discount_percent",
            "discount_percentage",
            "discount_rate"
        ]),

        "promotion": find_column(df, [
            "promotion",
            "promotions",
            "promo",
            "campaign",
            "is_promotional"
        ]),

        "inventory": find_column(df, [
            "inventory",
            "stock",
            "current_stock",
            "stock_level",
            "stock_quantity",
            "available_stock",
            "quantity_in_stock",
            "on_hand"
        ]),

        "lead_time": find_column(df, [
            "lead_time",
            "leadtime",
            "delivery_time",
            "supplier_lead_time",
            "days_to_delivery"
        ])
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_analytics_data(df):

    if df is None:

        return {
            "success": False,
            "message": "No dataset available."
        }

    if df.empty:

        return {
            "success": False,
            "message": "The dataset is empty."
        }

    df = df.copy()

    columns = detect_analytics_columns(df)

    demand_column = (
        columns["demand"]
        or columns["sales"]
    )

    if demand_column is None:

        return {
            "success": False,
            "message": (
                "No demand or sales column "
                "was detected."
            )
        }

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    df["analytics_demand"] = pd.to_numeric(
        df[demand_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=["analytics_demand"]
    )

    df = df[
        df["analytics_demand"] >= 0
    ]

    if df.empty:

        return {
            "success": False,
            "message": (
                "No valid demand records "
                "were found."
            )
        }

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if columns["date"]:

        df["analytics_date"] = pd.to_datetime(
            df[columns["date"]],
            errors="coerce"
        )

    else:

        df["analytics_date"] = pd.NaT

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    if columns["price"]:

        df["analytics_price"] = pd.to_numeric(
            df[columns["price"]],
            errors="coerce"
        )

    else:

        df["analytics_price"] = np.nan

    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    if columns["discount"]:

        df["analytics_discount"] = pd.to_numeric(
            df[columns["discount"]],
            errors="coerce"
        )

    else:

        df["analytics_discount"] = np.nan

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    if columns["inventory"]:

        df["analytics_inventory"] = pd.to_numeric(
            df[columns["inventory"]],
            errors="coerce"
        )

    else:

        df["analytics_inventory"] = np.nan

    # --------------------------------------------------------
    # Lead Time
    # --------------------------------------------------------

    if columns["lead_time"]:

        df["analytics_lead_time"] = pd.to_numeric(
            df[columns["lead_time"]],
            errors="coerce"
        )

    else:

        df["analytics_lead_time"] = np.nan

    # --------------------------------------------------------
    # Promotion
    # --------------------------------------------------------

    if columns["promotion"]:

        promotion_values = (
            df[columns["promotion"]]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df["analytics_promotion"] = (
            promotion_values.isin([
                "1",
                "true",
                "yes",
                "y",
                "on",
                "active",
                "promotional"
            ]).astype(int)
        )

    else:

        df["analytics_promotion"] = 0

    # --------------------------------------------------------
    # Time Features
    # --------------------------------------------------------

    if df["analytics_date"].notna().any():

        df["year"] = (
            df["analytics_date"]
            .dt.year
            .fillna(0)
        )

        df["month"] = (
            df["analytics_date"]
            .dt.month
            .fillna(0)
        )

        df["day"] = (
            df["analytics_date"]
            .dt.day
            .fillna(0)
        )

        df["day_of_week"] = (
            df["analytics_date"]
            .dt.dayofweek
            .fillna(0)
        )

        df["quarter"] = (
            df["analytics_date"]
            .dt.quarter
            .fillna(0)
        )

        df["week_of_year"] = (
            df["analytics_date"]
            .dt.isocalendar()
            .week
            .astype(float)
            .fillna(0)
        )

    else:

        df["year"] = 0
        df["month"] = 0
        df["day"] = 0
        df["day_of_week"] = 0
        df["quarter"] = 0
        df["week_of_year"] = 0

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if df["analytics_date"].notna().any():

        df = df.sort_values(
            "analytics_date"
        )

    # --------------------------------------------------------
    # Lag Features
    #
    # These are important because they allow the ML model
    # to learn from previous demand rather than using the
    # current demand to predict itself.
    # --------------------------------------------------------

    df["previous_demand"] = (
        df["analytics_demand"]
        .shift(1)
    )

    df["demand_rolling_mean"] = (
        df["analytics_demand"]
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
        .shift(1)
    )

    df["demand_rolling_std"] = (
        df["analytics_demand"]
        .rolling(
            window=7,
            min_periods=2
        )
        .std()
        .shift(1)
    )

    # --------------------------------------------------------
    # Demand Change
    # --------------------------------------------------------

    df["demand_change"] = (
        df["analytics_demand"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return {
        "success": True,
        "message": (
            "Analytics data prepared successfully."
        ),
        "data": df,
        "columns": columns,
        "demand_column": demand_column
    }


# ============================================================
# DEMAND CLASSIFICATION
# ============================================================

def create_demand_classes(df):

    df = df.copy()

    demand = df["analytics_demand"]

    low_threshold = demand.quantile(0.33)
    high_threshold = demand.quantile(0.66)

    def classify(value):

        if value <= low_threshold:
            return "Low"
        elif value <= high_threshold:
            return "Medium"
        return "High"

    df["demand_class"] = demand.apply(classify)
    return df


# ============================================================
# RANDOM FOREST DEMAND CLASSIFICATION
# ============================================================

def train_demand_classification_model(df):

    df = create_demand_classes(df)

    # IMPORTANT:
    # Do NOT use analytics_demand itself as a feature.
    # The target is created from analytics_demand.
    # Using it directly would create data leakage.

    feature_columns = [
        "previous_demand",
        "demand_rolling_mean",
        "demand_rolling_std",
        "demand_change",
        "analytics_price",
        "analytics_discount",
        "analytics_promotion",
        "analytics_inventory",
        "analytics_lead_time",
        "month",
        "day_of_week",
        "quarter",
        "week_of_year"
    ]

    usable_features = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        if df[column].notna().sum() < max(20, int(0.2 * len(df))):
            continue
        usable_features.append(column)

    if len(df) < 20 or len(usable_features) < 3:
        return {
            "available": False,
            "message": "At least 20 records and at least 3 usable features are required for reliable demand classification."
        }

    X = df[usable_features].copy()
    for column in usable_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    X = X.fillna(0)

    y = df["demand_class"].copy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    if len(unique_classes) < 2:
        return {
            "available": False,
            "message": "The dataset does not contain enough demand variation for classification."
        }

    # 70/15/15 split (train/validation/test) with stratification.
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y_encoded,
            test_size=0.30,
            random_state=42,
            stratify=y_encoded
        )

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y_encoded,
            test_size=0.30,
            random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42
        )

    # Correct target creation is already coming from analytics_demand class bins.
    # Use cross-validated Random Forest hyperparameter search over a small grid.
    param_grid = {
        "n_estimators": [80, 150],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced", "balanced_subsample"]
    }

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    all_predictions = model.predict(X)
    predicted_labels = encoder.inverse_transform(all_predictions)
    actual_labels = encoder.inverse_transform(y_encoded)

    feature_importance = {}
    for feature, importance in zip(usable_features, model.feature_importances_):
        feature_importance[feature] = round(float(importance), 4)
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    return {
        "available": True,
        "model": "Random Forest Classifier",
        "accuracy": round(float(accuracy * 100), 2),
        "precision": round(float(precision * 100), 2),
        "recall": round(float(recall * 100), 2),
        "f1_score": round(float(f1 * 100), 2),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "testing_rows": int(len(X_test)),
        "features_used": usable_features,
        "feature_importance": feature_importance,
        "predicted_labels": predicted_labels.tolist(),
        "actual_labels": actual_labels.tolist(),
        "classes": encoder.classes_.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "message": "Random Forest demand classification completed successfully."
    }


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_demand_anomalies(df):

    if len(df) < 10:
        return {
            "available": False,
            "message": "At least 10 records are recommended for anomaly detection."
        }

    demand = df[["analytics_demand"]].copy()
    demand = demand.replace([np.inf, -np.inf], np.nan)
    demand = demand.dropna()

    if len(demand) < 10:
        return {
            "available": False,
            "message": "Not enough valid demand values for anomaly detection."
        }

    # Work with a fixed and explicit contamination level instead of auto.
    contamination = 0.05
    X = demand[["analytics_demand"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    predictions = model.fit_predict(X)
    scores = -model.decision_function(X)

    anomaly_mask = predictions == -1
    anomaly_count = int(anomaly_mask.sum())

    result_df = df.loc[X.index].copy()
    result_df["anomaly_score"] = scores.round(4)
    result_df["is_anomaly"] = anomaly_mask

    anomaly_records = result_df[result_df["is_anomaly"]].copy()
    anomaly_records = anomaly_records.sort_values("analytics_demand", ascending=False)

    anomalies = []
    for _, row in anomaly_records.head(20).iterrows():
        date_value = row.get("analytics_date")
        formatted_date = None
        if pd.notna(date_value):
            formatted_date = pd.to_datetime(date_value).strftime("%Y-%m-%d")

        product_name = None
        for product_key in ("product", "product_id", "product_name", "sku", "item_id"):
            if product_key in row.index and pd.notna(row[product_key]):
                product_name = str(row[product_key])
                break

        anomalies.append({
            "date": formatted_date,
            "product": product_name,
            "demand": round(float(row["analytics_demand"]), 2),
            "anomaly_score": round(float(row["anomaly_score"]), 4)
        })

    return {
        "available": True,
        "model": "Isolation Forest",
        "total_records": int(len(result_df)),
        "anomaly_count": anomaly_count,
        "normal_count": int(len(result_df) - anomaly_count),
        "anomaly_percentage": round((anomaly_count / len(result_df)) * 100, 2),
        "anomalies": anomalies,
        "message": "Demand anomaly detection completed successfully."
    }


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

def analyze_product_performance(
    df,
    columns
):

    product_column = columns.get(
        "product"
    )

    if product_column is None:

        return {

            "available": False,

            "message": (
                "No product column was detected."
            ),

            "products": []
        }

    product_df = df.copy()

    product_df["_product_name"] = (
        product_df[
            product_column
        ]
        .astype(str)
    )

    product_summary = (

        product_df

        .groupby(
            "_product_name"
        )

        .agg(

            total_demand=(
                "analytics_demand",
                "sum"
            ),

            average_demand=(
                "analytics_demand",
                "mean"
            ),

            transactions=(
                "analytics_demand",
                "count"
            )

        )

        .reset_index()
    )

    product_summary = (
        product_summary
        .sort_values(
            "total_demand",
            ascending=False
        )
    )

    products = []

    for _, row in product_summary.head(20).iterrows():

        products.append({

            "product": str(
                row["_product_name"]
            ),

            "total_demand": round(
                float(
                    row["total_demand"]
                ),
                2
            ),

            "average_demand": round(
                float(
                    row["average_demand"]
                ),
                2
            ),

            "transactions": int(
                row["transactions"]
            )
        })

    return {

        "available": True,

        "products": products,

        "total_products": int(
            len(product_summary)
        ),

        "message": (
            "Product performance analysis "
            "completed successfully."
        )
    }


# ============================================================
# DEMAND DRIVER ANALYSIS
# ============================================================

def analyze_demand_drivers(df):

    drivers = {}

    target = df[
        "analytics_demand"
    ]

    numeric_features = {

        "Price":
            "analytics_price",

        "Discount":
            "analytics_discount",

        "Promotion":
            "analytics_promotion",

        "Inventory":
            "analytics_inventory",

        "Lead Time":
            "analytics_lead_time",

        "Previous Demand":
            "previous_demand",

        "Rolling Demand":
            "demand_rolling_mean",

        "Day of Week":
            "day_of_week",

        "Month":
            "month"
    }

    for display_name, column in numeric_features.items():

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid = pd.DataFrame({

            "feature": values,

            "demand": target

        }).dropna()

        if len(valid) < 3:
            continue

        correlation = valid[
            "feature"
        ].corr(
            valid["demand"]
        )

        if pd.notna(correlation):

            drivers[
                display_name
            ] = round(
                float(correlation),
                4
            )

    drivers = dict(
        sorted(
            drivers.items(),
            key=lambda item: abs(
                item[1]
            ),
            reverse=True
        )
    )

    return {

        "available": len(drivers) > 0,

        "drivers": drivers,

        "message": (
            "Demand driver analysis completed."
        )
    }


# ============================================================
# SEASONAL ANALYSIS
# ============================================================

def analyze_seasonality(df):

    result = {

        "available": False,

        "weekly_pattern": None,

        "monthly_pattern": None,

        "weekly_data": [],

        "monthly_data": [],

        "message": (
            "Seasonal analysis is waiting "
            "for valid date information."
        )
    }

    if "analytics_date" not in df.columns:

        return result

    valid = df[
        df["analytics_date"].notna()
    ].copy()

    if valid.empty:

        return result

    # --------------------------------------------------------
    # Weekly
    # --------------------------------------------------------

    weekly = (

        valid

        .groupby(
            "day_of_week"
        )["analytics_demand"]

        .mean()

        .reset_index()
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

    for _, row in weekly.iterrows():

        day_index = int(
            row["day_of_week"]
        )

        weekly_data.append({

            "day": day_names[
                day_index
            ],

            "demand": round(
                float(
                    row["analytics_demand"]
                ),
                2
            )
        })

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    valid["month_period"] = (
        valid["analytics_date"]
        .dt.to_period("M")
    )

    monthly = (

        valid

        .groupby(
            "month_period"
        )["analytics_demand"]

        .sum()

        .reset_index()
    )

    monthly_data = []

    for _, row in monthly.iterrows():

        monthly_data.append({

            "month": str(
                row["month_period"]
            ),

            "demand": round(
                float(
                    row["analytics_demand"]
                ),
                2
            )
        })

    # --------------------------------------------------------
    # Highest patterns
    # --------------------------------------------------------

    if weekly_data:

        highest_week = max(
            weekly_data,
            key=lambda x: x["demand"]
        )

        weekly_pattern = (
            highest_week["day"]
        )

    else:

        weekly_pattern = None

    if monthly_data:

        highest_month = max(
            monthly_data,
            key=lambda x: x["demand"]
        )

        monthly_pattern = (
            highest_month["month"]
        )

    else:

        monthly_pattern = None

    result.update({

        "available": True,

        "weekly_pattern": weekly_pattern,

        "monthly_pattern": monthly_pattern,

        "weekly_data": weekly_data,

        "monthly_data": monthly_data,

        "message": (
            "Weekly and monthly demand "
            "patterns detected successfully."
        )
    })

    return result


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    classification,
    anomalies,
    product_analysis,
    demand_drivers,
    seasonality
):

    insights = []

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if classification.get("available"):

        accuracy = classification.get(
            "accuracy",
            0
        )

        insights.append(
            f"Random Forest demand classification "
            f"achieved {accuracy}% test accuracy."
        )

    # --------------------------------------------------------
    # Anomalies
    # --------------------------------------------------------

    if anomalies.get("available"):

        anomaly_count = anomalies.get(
            "anomaly_count",
            0
        )

        if anomaly_count > 0:

            insights.append(
                f"{anomaly_count} unusual demand "
                f"records were detected and should "
                f"be reviewed."
            )

        else:

            insights.append(
                "No significant demand anomalies "
                "were detected."
            )

    # --------------------------------------------------------
    # Products
    # --------------------------------------------------------

    if product_analysis.get("available"):

        products = product_analysis.get(
            "products",
            []
        )

        if products:

            top_product = products[0]

            insights.append(
                f"{top_product['product']} is the "
                f"highest-demand product based on "
                f"total demand."
            )

    # --------------------------------------------------------
    # Drivers
    # --------------------------------------------------------

    if demand_drivers.get("available"):

        drivers = demand_drivers.get(
            "drivers",
            {}
        )

        if drivers:

            strongest_driver = next(
                iter(drivers)
            )

            correlation = drivers[
                strongest_driver
            ]

            direction = (
                "positive"
                if correlation >= 0
                else "negative"
            )

            insights.append(
                f"{strongest_driver} shows the strongest "
                f"{direction} relationship with demand "
                f"among the analyzed variables."
            )

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    if seasonality.get("available"):

        weekly = seasonality.get(
            "weekly_pattern"
        )

        if weekly:

            insights.append(
                f"{weekly} shows the highest "
                f"average demand among weekdays."
            )

    if not insights:

        insights.append(
            "More compatible retail data is required "
            "to generate business insights."
        )

    return insights


# ============================================================
# COMPLETE ANALYTICS ML PIPELINE
# ============================================================

def run_analytics_ml(dataset):

    result = {

        "success": False,

        "message": (
            "Analytics could not be completed."
        ),

        "classification": {
            "available": False
        },

        "anomaly_detection": {
            "available": False
        },

        "product_analysis": {
            "available": False
        },

        "demand_drivers": {
            "available": False
        },

        "seasonality": {
            "available": False
        },

        "business_insights": []
    }

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    preparation = prepare_analytics_data(
        dataset
    )

    if not preparation["success"]:

        result["message"] = (
            preparation["message"]
        )

        return result

    df = preparation["data"]

    columns = preparation["columns"]

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    try:

        result["classification"] = (
            train_demand_classification_model(
                df
            )
        )

    except Exception as error:

        result["classification"] = {

            "available": False,

            "message": (
                f"Classification failed: {error}"
            )
        }

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    try:

        result["anomaly_detection"] = (
            detect_demand_anomalies(
                df
            )
        )

    except Exception as error:

        result["anomaly_detection"] = {

            "available": False,

            "message": (
                f"Anomaly detection failed: {error}"
            )
        }

    # --------------------------------------------------------
    # Product Analysis
    # --------------------------------------------------------

    try:

        result["product_analysis"] = (
            analyze_product_performance(
                df,
                columns
            )
        )

    except Exception as error:

        result["product_analysis"] = {

            "available": False,

            "message": (
                f"Product analysis failed: {error}"
            )
        }

    # --------------------------------------------------------
    # Demand Drivers
    # --------------------------------------------------------

    try:

        result["demand_drivers"] = (
            analyze_demand_drivers(
                df
            )
        )

    except Exception as error:

        result["demand_drivers"] = {

            "available": False,

            "message": (
                f"Demand driver analysis failed: {error}"
            )
        }

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    try:

        result["seasonality"] = (
            analyze_seasonality(
                df
            )
        )

    except Exception as error:

        result["seasonality"] = {

            "available": False,

            "message": (
                f"Seasonality analysis failed: {error}"
            )
        }

    # --------------------------------------------------------
    # Business Insights
    # --------------------------------------------------------

    try:

        result["business_insights"] = (
            generate_business_insights(

                result["classification"],

                result["anomaly_detection"],

                result["product_analysis"],

                result["demand_drivers"],

                result["seasonality"]
            )
        )

    except Exception as error:

        result["business_insights"] = [

            f"Business insight generation failed: {error}"
        ]

    result["success"] = True

    result["message"] = (
        "Analytics ML pipeline completed successfully."
    )

    return result