# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# FORECASTING SERVICE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PREPARE FORECASTING DATA
# ============================================================

def prepare_forecasting_data(
    df,
    date_column,
    target_column
):

    data = df.copy()

    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    if date_column not in data.columns:

        raise ValueError(
            f"Date column '{date_column}' was not found."
        )

    if target_column not in data.columns:

        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CONVERT TARGET TO NUMERIC
    # --------------------------------------------------------

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            date_column,
            target_column
        ]
    )

    # --------------------------------------------------------
    # SORT AND AGGREGATE TO ONE DAILY SERIES
    # --------------------------------------------------------

    data = data.sort_values(
        date_column
    )

    data = (
        data
        .groupby(date_column, as_index=False)[target_column]
        .sum()
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CREATE DATE FEATURES
    # --------------------------------------------------------

    data["year"] = (
        data[date_column].dt.year
    )

    data["month"] = (
        data[date_column].dt.month
    )

    data["day"] = (
        data[date_column].dt.day
    )

    data["day_of_week"] = (
        data[date_column].dt.dayofweek
    )

    data["day_of_year"] = (
        data[date_column].dt.dayofyear
    )

    # --------------------------------------------------------
    # CAUSAL DEMAND FEATURES
    # --------------------------------------------------------

    lag_features = []
    for lag in (1, 7, 14, 28):
        if len(data) >= lag + 8:
            column = f"lag_{lag}"
            data[column] = data[target_column].shift(lag)
            lag_features.append(column)

    rolling_features = []
    for window in (7, 14, 28):
        if len(data) >= window + 8:
            column = f"rolling_mean_{window}"
            data[column] = (
                data[target_column]
                .shift(1)
                .rolling(window)
                .mean()
            )
            rolling_features.append(column)

    feature_columns = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year"
    ] + lag_features + rolling_features

    data = data.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    X = data[feature_columns]

    y = data[
        target_column
    ]

    return (
        data,
        X,
        y,
        feature_columns
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(
    X,
    y
):

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# GRADIENT BOOSTING
# ============================================================

def train_gradient_boosting(
    X,
    y
):

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        loss="huber",
        random_state=42
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "mse": round(
            float(mse),
            2
        ),

        "mae": round(
            float(mae),
            2
        ),

        "rmse": round(
            float(rmse),
            2
        ),

        "r2": round(
            float(r2),
            4
        )

    }


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

def train_forecasting_models(
    df,
    date_column,
    target_column
):

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    (
        data,
        X,
        y,
        feature_columns
    ) = prepare_forecasting_data(
        df,
        date_column,
        target_column
    )

    # --------------------------------------------------------
    # CHECK DATA SIZE
    # --------------------------------------------------------

    if len(data) < 10:

        raise ValueError(
            "At least 10 valid rows are required "
            "for model training."
        )

    # --------------------------------------------------------
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = max(
        int(len(X) * 0.8),
        1
    )

    if split_index >= len(X):
        split_index = len(X) - 1

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    random_forest = train_random_forest(
        X_train,
        y_train
    )

    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    gradient_boosting = train_gradient_boosting(
        X_train,
        y_train
    )

    gb_metrics = evaluate_model(
        gradient_boosting,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    models = {

        "Random Forest": {
            "model": random_forest,
            "metrics": rf_metrics
        },

        "Gradient Boosting": {
            "model": gradient_boosting,
            "metrics": gb_metrics
        }

    }

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------
    #
    # RMSE is primary because it penalizes large misses.
    #

    best_model_name = min(
        models,
        key=lambda name:
            models[name]["metrics"]["rmse"]
    )

    best_model = models[
        best_model_name
    ]["model"]

    best_metrics = models[
        best_model_name
    ]["metrics"]

    return {

        "models": models,

        "best_model_name":
            best_model_name,

        "best_model":
            best_model,

        "best_metrics":
            best_metrics,

        "feature_columns":
            feature_columns,

        "data":
            data

    }


# ============================================================
# GENERATE FUTURE FORECAST
# ============================================================

def generate_forecast(
    model,
    last_date,
    periods=30,
    history=None,
    target_column=None,
    feature_columns=None
):

    future_dates = pd.date_range(

        start=(
            last_date
            + pd.Timedelta(days=1)
        ),

        periods=periods,

        freq="D"

    )

    future = pd.DataFrame({

        "date":
            future_dates

    })

    # --------------------------------------------------------
    # CREATE SAME FEATURES USED DURING TRAINING
    # --------------------------------------------------------

    future["year"] = (
        future["date"].dt.year
    )

    future["month"] = (
        future["date"].dt.month
    )

    future["day"] = (
        future["date"].dt.day
    )

    future["day_of_week"] = (
        future["date"].dt.dayofweek
    )

    future["day_of_year"] = (
        future["date"].dt.dayofyear
    )

    calendar_features = [

        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year"

    ]

    feature_columns = feature_columns or calendar_features

    if feature_columns != calendar_features:
        if history is None or target_column is None:
            raise ValueError(
                "History and target column are required for lag forecasting."
            )

        history_values = (
            history[target_column]
            .dropna()
            .astype(float)
            .tolist()
        )
        predictions = []

        for _, row in future.iterrows():
            values = {
                column: row[column]
                for column in calendar_features
            }

            for column in feature_columns:
                if column.startswith("lag_"):
                    lag = int(column.removeprefix("lag_"))
                    values[column] = history_values[-lag]
                elif column.startswith("rolling_mean_"):
                    window = int(
                        column.removeprefix("rolling_mean_")
                    )
                    values[column] = np.mean(
                        history_values[-window:]
                    )

            prediction = max(
                float(
                    model.predict(
                        pd.DataFrame([values])[feature_columns]
                    )[0]
                ),
                0
            )
            predictions.append(prediction)
            history_values.append(prediction)
    else:
        predictions = model.predict(
            future[feature_columns]
        )

    future["forecast"] = (
        predictions
    )

    future["forecast"] = (
        future["forecast"]
        .clip(lower=0)
        .round(2)
    )

    return future