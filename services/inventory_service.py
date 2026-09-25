# ============================================================
# INVENTORY OPTIMIZATION SERVICE
# Retail Demand Forecasting System
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SERVICE_LEVEL = 0.95
Z_SCORE = 1.645

# Number of days used as inventory review period
REVIEW_PERIOD_DAYS = 7

# Overstock tolerance
OVERSTOCK_FACTOR = 1.20

# Minimum rows required to train ML model
MIN_TRAINING_ROWS = 50

# Output directory
INVENTORY_OUTPUT_DIR = os.path.join(
    "data",
    "reports"
)

INVENTORY_RESULT_PATH = os.path.join(
    INVENTORY_OUTPUT_DIR,
    "inventory_optimization_results.csv"
)

INVENTORY_MODEL_INFO_PATH = os.path.join(
    INVENTORY_OUTPUT_DIR,
    "inventory_model_info.json"
)


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "product_id": [
        "product_id",
        "product",
        "product_name",
        "sku",
        "item_id",
        "item"
    ],

    "current_stock": [
        "current_stock",
        "stock",
        "inventory",
        "stock_level",
        "stock_quantity",
        "available_stock",
        "quantity_in_stock",
        "on_hand"
    ],

    "daily_demand": [
        "daily_demand",
        "demand",
        "daily_sales",
        "sales_per_day",
        "units_per_day",
        "quantity",
        "qty",
        "units_sold",
        "sales_quantity",
        "quantity_sold"
    ],

    "lead_time_days": [
        "lead_time_days",
        "lead_time",
        "leadtime",
        "delivery_time",
        "supplier_lead_time",
        "days_to_delivery"
    ],

    "supplier_reliability_score": [
        "supplier_reliability_score",
        "supplier_reliability",
        "reliability_score",
        "supplier_score"
    ],

    "promotion_active": [
        "promotion_active",
        "promotion",
        "promo_active",
        "is_promotion",
        "promotional"
    ],

    "weather_impact": [
        "weather_impact",
        "weather",
        "weather_effect",
        "weather_score"
    ],

    "stockout_risk": [
        "stockout_risk",
        "stockout",
        "stockout_status",
        "risk",
        "stockout_probability"
    ]
}


# ============================================================
# COLUMN FINDER
# ============================================================

def normalize_column_name(column):
    """
    Convert column name into a normalized form.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, possible_names):
    """
    Find a matching column from the dataframe.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for name in possible_names:

        normalized_name = normalize_column_name(name)

        if normalized_name in normalized:
            return normalized[normalized_name]

    return None


# ============================================================
# STANDARDIZE DATASET
# ============================================================

def standardize_inventory_dataset(dataset):
    """
    Convert different possible dataset column names
    into the standard inventory column names.
    """

    if dataset is None:
        return None, {
            "valid": False,
            "message": "No dataset was provided."
        }

    if dataset.empty:
        return None, {
            "valid": False,
            "message": "The dataset is empty."
        }

    df = dataset.copy()

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        found_column = find_column(
            df,
            aliases
        )

        if found_column is not None:

            rename_map[found_column] = standard_name

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "current_stock",
        "daily_demand",
        "lead_time_days"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        return None, {
            "valid": False,
            "message": (
                "Inventory analysis requires these columns: "
                + ", ".join(required_columns)
                + ". Missing: "
                + ", ".join(missing_columns)
            ),
            "missing_columns": missing_columns
        }

    # Create product ID if dataset doesn't have one
    if "product_id" not in df.columns:

        df["product_id"] = [
            f"PRODUCT-{index + 1}"
            for index in range(len(df))
        ]

    # Convert numerical columns
    numeric_columns = [
        "current_stock",
        "daily_demand",
        "lead_time_days",
        "supplier_reliability_score"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Default supplier reliability
    if "supplier_reliability_score" not in df.columns:

        df["supplier_reliability_score"] = 80.0

    # Default promotion
    if "promotion_active" not in df.columns:

        df["promotion_active"] = "No"

    # Default weather
    if "weather_impact" not in df.columns:

        df["weather_impact"] = "Normal"

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "current_stock",
            "daily_demand",
            "lead_time_days"
        ]
    ).copy()

    # Negative values are invalid
    df = df[
        (df["current_stock"] >= 0)
        &
        (df["daily_demand"] >= 0)
        &
        (df["lead_time_days"] >= 0)
    ].copy()

    if df.empty:

        return None, {
            "valid": False,
            "message": (
                "No valid inventory records remain "
                "after data validation."
            )
        }

    # Limit reliability score
    df["supplier_reliability_score"] = (
        df["supplier_reliability_score"]
        .clip(0, 100)
    )

    return df, {
        "valid": True,
        "message": "Inventory dataset validated successfully.",
        "rows": len(df)
    }


# ============================================================
# ENCODE STOCKOUT LABEL
# ============================================================

def normalize_stockout_label(value):
    """
    Convert different stockout labels into:
    0 = No
    1 = Yes
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    positive_values = {
        "yes",
        "y",
        "true",
        "1",
        "high",
        "critical",
        "risk",
        "at_risk",
        "stockout",
        "likely"
    }

    negative_values = {
        "no",
        "n",
        "false",
        "0",
        "low",
        "safe",
        "normal",
        "healthy",
        "unlikely"
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    # Numeric values
    try:

        numeric_value = float(text)

        if numeric_value in [0, 1]:
            return int(numeric_value)

    except Exception:
        pass

    return np.nan


# ============================================================
# PREPARE ML FEATURES
# ============================================================

def prepare_ml_features(df):

    ml_df = df.copy()

    # Numeric features
    numeric_features = [
        "current_stock",
        "daily_demand",
        "lead_time_days",
        "supplier_reliability_score"
    ]

    # Categorical features
    categorical_features = [
        "promotion_active",
        "weather_impact"
    ]

    # Ensure categorical values are strings
    for column in categorical_features:

        ml_df[column] = (
            ml_df[column]
            .fillna("Unknown")
            .astype(str)
        )

    # Ensure numeric values are valid
    for column in numeric_features:

        ml_df[column] = pd.to_numeric(
            ml_df[column],
            errors="coerce"
        )

    ml_df = ml_df.dropna(
        subset=numeric_features
    ).copy()

    return (
        ml_df,
        numeric_features,
        categorical_features
    )


# ============================================================
# TRAIN STOCKOUT ML MODEL
# ============================================================

def train_stockout_model(df):

    result = {
        "model_available": False,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "training_rows": 0,
        "validation_rows": 0,
        "testing_rows": 0,
        "message": "",
        "model": None,
        "feature_importance": {}
    }

    if "stockout_risk" not in df.columns:
        result["message"] = (
            "The dataset does not contain a stockout_risk target column. "
            "Rule-based inventory optimization will be used."
        )
        return result

    ml_df = df.copy()
    ml_df["stockout_target"] = ml_df["stockout_risk"].apply(normalize_stockout_label)
    ml_df = ml_df.dropna(subset=["stockout_target"]).copy()

    if len(ml_df) < MIN_TRAINING_ROWS:
        result["message"] = (
            f"Only {len(ml_df)} labelled rows are available. At least {MIN_TRAINING_ROWS} rows are recommended for ML training."
        )
        return result

    if ml_df["stockout_target"].nunique() < 2:
        result["message"] = (
            "The stockout_risk column contains only one class. A binary ML classifier cannot be trained."
        )
        return result

    # Prepare features with a categorical pass-through to keep strings safe.
    ml_df, numeric_features, categorical_features = prepare_ml_features(ml_df)

    if len(ml_df) < MIN_TRAINING_ROWS:
        result["message"] = "Not enough valid rows remain after feature preparation."
        return result

    feature_columns = numeric_features + categorical_features
    X = ml_df[feature_columns]
    y = ml_df["stockout_target"].astype(int)

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
    except ValueError:
        result["message"] = (
            "The dataset does not contain enough samples in each stockout class for a stratified split."
        )
        return result

    # Enforce preprocessing and cross-validation inside the pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    est = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", est)
    ])

    parameter_grid = {
        "classifier__n_estimators": [80, 150],
        "classifier__max_depth": [4, 8, None],
        "classifier__min_samples_split": [2, 4],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__class_weight": ["balanced", "balanced_subsample"]
    }

    try:
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1_weighted",
            cv=3,
            n_jobs=-1,
            refit=True,
            return_train_score=False
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        # Wrap final evaluation on the untouched test split.
        predictions = best_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        result["model_available"] = True
        result["accuracy"] = round(float(accuracy) * 100, 2)
        result["precision"] = round(float(precision) * 100, 2)
        result["recall"] = round(float(recall) * 100, 2)
        result["f1_score"] = round(float(f1) * 100, 2)
        result["training_rows"] = len(ml_df)
        result["validation_rows"] = int(len(X_valid))
        result["testing_rows"] = int(len(X_test))
        result["model"] = best_pipeline
        result["message"] = "Random Forest stockout-risk model trained successfully."

        try:
            classifier = best_pipeline.named_steps["classifier"]
            importances = classifier.feature_importances_
            feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            result["feature_importance"] = {key: round(float(value), 4) for key, value in sorted_importance[:10]}
        except Exception:
            result["feature_importance"] = {}

    except Exception as error:
        result["message"] = "ML training failed: " + str(error)

    return result


# ============================================================
# CALCULATE INVENTORY OPTIMIZATION
# ============================================================

def calculate_inventory_optimization(
    dataset,
    service_level=SERVICE_LEVEL,
    review_period_days=REVIEW_PERIOD_DAYS
):

    df, validation = standardize_inventory_dataset(
        dataset
    )

    if df is None:

        return {
            "available": False,
            "message": validation["message"],
            "summary": {},
            "products": [],
            "ml": {}
        }

    # ========================================================
    # DEMAND VARIABILITY
    # ========================================================

    global_demand_std = float(
        df["daily_demand"]
        .std()
    )

    if np.isnan(global_demand_std):
        global_demand_std = 0.0

    # Fallback if demand variability is zero
    if global_demand_std == 0:

        global_demand_std = max(
            float(df["daily_demand"].mean()) * 0.10,
            1.0
        )

    # ========================================================
    # ML MODEL
    # ========================================================

    ml_result = train_stockout_model(
        df
    )

    model = ml_result.get(
        "model"
    )

    # ========================================================
    # INVENTORY CALCULATIONS
    # ========================================================

    result_df = df.copy()

    # --------------------------------------------------------
    # Lead-time demand
    # --------------------------------------------------------

    result_df["lead_time_demand"] = (
        result_df["daily_demand"]
        *
        result_df["lead_time_days"]
    )

    # --------------------------------------------------------
    # Safety stock
    #
    # Base formula:
    #
    # Z × demand variability × sqrt(lead time)
    #
    # Supplier reliability adjustment:
    # Lower reliability → larger safety stock
    # --------------------------------------------------------

    reliability_adjustment = (
        1
        +
        (
            100
            -
            result_df["supplier_reliability_score"]
        )
        / 100
        *
        0.50
    )

    result_df["safety_stock"] = (
        Z_SCORE
        *
        global_demand_std
        *
        np.sqrt(
            result_df["lead_time_days"]
            .clip(lower=0)
        )
        *
        reliability_adjustment
    )

    # --------------------------------------------------------
    # Reorder point
    #
    # Lead-time demand + safety stock
    # --------------------------------------------------------

    result_df["reorder_point"] = (
        result_df["lead_time_demand"]
        +
        result_df["safety_stock"]
    )

    # --------------------------------------------------------
    # Target stock
    #
    # Demand during lead time + review period + safety stock
    # --------------------------------------------------------

    result_df["target_stock"] = (
        result_df["daily_demand"]
        *
        (
            result_df["lead_time_days"]
            +
            review_period_days
        )
        +
        result_df["safety_stock"]
    )

    # --------------------------------------------------------
    # Recommended order
    # --------------------------------------------------------

    result_df["recommended_order"] = (
        result_df["target_stock"]
        -
        result_df["current_stock"]
    ).clip(
        lower=0
    )

    # --------------------------------------------------------
    # Stock coverage
    # --------------------------------------------------------

    result_df["stock_cover_days"] = np.where(
        result_df["daily_demand"] > 0,

        result_df["current_stock"]
        /
        result_df["daily_demand"],

        np.inf
    )

    # ========================================================
    # ML STOCKOUT PROBABILITY
    # ========================================================

    if model is not None:

        ml_features = result_df[
            [
                "current_stock",
                "daily_demand",
                "lead_time_days",
                "supplier_reliability_score",
                "promotion_active",
                "weather_impact"
            ]
        ].copy()

        try:

            probabilities = model.predict_proba(
                ml_features
            )

            # Find probability for positive class
            class_index = list(
                model.classes_
            ).index(1)

            result_df["stockout_probability"] = (
                probabilities[:, class_index]
                * 100
            )

            result_df["ml_risk"] = np.where(
                result_df["stockout_probability"] >= 70,
                "High",

                np.where(
                    result_df["stockout_probability"] >= 40,
                    "Medium",
                    "Low"
                )
            )

        except Exception:

            result_df["stockout_probability"] = np.nan
            result_df["ml_risk"] = "Unavailable"

    else:

        result_df["stockout_probability"] = np.nan
        result_df["ml_risk"] = "Unavailable"

    # ========================================================
    # INVENTORY STATUS
    # ========================================================

    statuses = []

    for _, row in result_df.iterrows():

        current_stock = float(
            row["current_stock"]
        )

        reorder_point = float(
            row["reorder_point"]
        )

        target_stock = float(
            row["target_stock"]
        )

        risk = row["ml_risk"]

        if current_stock <= reorder_point:

            status = "Critical - Reorder Now"

        elif (
            risk == "High"
            and current_stock <= target_stock
        ):

            status = "High Risk"

        elif current_stock > (
            target_stock
            *
            OVERSTOCK_FACTOR
        ):

            status = "Overstock"

        elif current_stock < target_stock:

            status = "Low Stock"

        else:

            status = "Healthy"

        statuses.append(status)

    result_df["status"] = statuses

    # ========================================================
    # ROUND VALUES
    # ========================================================

    decimal_columns = [
        "current_stock",
        "daily_demand",
        "lead_time_days",
        "supplier_reliability_score",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order",
        "stock_cover_days",
        "stockout_probability"
    ]

    for column in decimal_columns:

        if column in result_df.columns:

            result_df[column] = result_df[
                column
            ].round(2)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "total_products": int(
            len(result_df)
        ),

        "low_stock": int(
            result_df["status"]
            .isin(
                [
                    "Low Stock",
                    "Critical - Reorder Now",
                    "High Risk"
                ]
            )
            .sum()
        ),

        "overstock": int(
            (
                result_df["status"]
                ==
                "Overstock"
            )
            .sum()
        ),

        "reorder_required": int(
            (
                result_df["recommended_order"]
                > 0
            )
            .sum()
        ),

        "critical_items": int(
            (
                result_df["status"]
                ==
                "Critical - Reorder Now"
            )
            .sum()
        ),

        "high_risk_items": int(
            (
                result_df["ml_risk"]
                ==
                "High"
            )
            .sum()
        ),

        "healthy_items": int(
            (
                result_df["status"]
                ==
                "Healthy"
            )
            .sum()
        ),

        "average_daily_demand": round(
            float(
                result_df["daily_demand"].mean()
            ),
            2
        ),

        "average_lead_time": round(
            float(
                result_df["lead_time_days"].mean()
            ),
            2
        ),

        "average_safety_stock": round(
            float(
                result_df["safety_stock"].mean()
            ),
            2
        ),

        "average_reorder_point": round(
            float(
                result_df["reorder_point"].mean()
            ),
            2
        ),

        "total_recommended_order": round(
            float(
                result_df["recommended_order"].sum()
            ),
            2
        ),

        "average_stock_cover_days": round(
            float(
                result_df[
                    "stock_cover_days"
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
                .mean()
            ),
            2
        )
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    try:

        os.makedirs(
            INVENTORY_OUTPUT_DIR,
            exist_ok=True
        )

        result_df.to_csv(
            INVENTORY_RESULT_PATH,
            index=False
        )

        model_info = {

            "model": (
                "Random Forest Classifier"
                if ml_result["model_available"]
                else "Rule-based optimization"
            ),

            "ml_available": ml_result[
                "model_available"
            ],

            "accuracy": ml_result[
                "accuracy"
            ],

            "training_rows": ml_result[
                "training_rows"
            ],

            "service_level": service_level,

            "z_score": Z_SCORE,

            "review_period_days": review_period_days,

            "demand_std_proxy": round(
                global_demand_std,
                4
            ),

            "message": ml_result[
                "message"
            ],

            "feature_importance": ml_result[
                "feature_importance"
            ]
        }

        with open(
            INVENTORY_MODEL_INFO_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                model_info,
                file,
                indent=4
            )

    except Exception as error:

        print(
            "Inventory result save warning:",
            error
        )

    # ========================================================
    # SELECT PRODUCT COLUMNS FOR FRONTEND
    # ========================================================

    display_columns = [
        "product_id",
        "current_stock",
        "daily_demand",
        "lead_time_days",
        "supplier_reliability_score",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order",
        "stock_cover_days",
        "stockout_probability",
        "ml_risk",
        "status"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in result_df.columns
    ]

    products = (
        result_df[
            display_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    return {

        "available": True,

        "message": (
            "Inventory optimization completed "
            "using the shared cleaned dataset."
        ),

        "summary": summary,

        "products": products,

        "ml": {

            "available": ml_result[
                "model_available"
            ],

            "accuracy": ml_result[
                "accuracy"
            ],

            "training_rows": ml_result[
                "training_rows"
            ],

            "message": ml_result[
                "message"
            ],

            "feature_importance": ml_result[
                "feature_importance"
            ]
        },

        "result_dataframe": result_df
    }


# ============================================================
# SIMPLE FUNCTION FOR API / ROUTE USE
# ============================================================

def run_inventory_optimization(dataset):

    return calculate_inventory_optimization(
        dataset
    )