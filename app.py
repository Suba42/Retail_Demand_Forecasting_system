# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# FLASK BACKEND
# ============================================================

import io
import os
import json
import math
import warnings

from urllib.parse import quote

import numpy as np
import pandas as pd

from services.analytics_service import (
    run_analytics_ml
)

from services.seasonal_service import (
    run_seasonal_ml
)

from services.alerts_service import (
    run_alerts_ml,
    calculate_alert_summary
)

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file
)

from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from functools import wraps

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from config import Config
from database.models import (
    db,
    User,
    NotificationSubscription,
    NotificationLog
)
from services.notification_service import (
    send_email_notification
)
from services.data_cleaner import (
    clean_dataset,
    normalize_column_name,
    automatic_column_mapping
)

from services.forecasting_service import (
    train_forecasting_models,
    generate_forecast as generate_service_forecast
)

warnings.filterwarnings("ignore")

app = Flask(__name__)

from powerbi_dashboard import powerbi_bp
app.register_blueprint(powerbi_bp)
app.config.from_object(Config)
# ============================================================
# MICROSOFT POWER BI CONFIGURATION
# ============================================================

POWERBI_REPORT_URL = os.environ.get(
    "POWERBI_REPORT_URL",
    ""
)

POWERBI_EMBED_URL = os.environ.get(
    "POWERBI_EMBED_URL",
    ""
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")
CLEANED_FOLDER = os.path.join(BASE_DIR, "data", "cleaned")
REPORT_FOLDER = os.path.join(BASE_DIR, "data", "reports")
FORECAST_FOLDER = os.path.join(BASE_DIR, "data", "forecasts")
MODULE_DATA_FOLDER = os.path.join(CLEANED_FOLDER, "module_sources")
MODULE_SOURCE_METADATA_PATH = os.path.join(
    MODULE_DATA_FOLDER,
    "sources.json"
)

SHARED_DATASET_PATH = os.path.join(
    CLEANED_FOLDER, "cleaned_shared_dataset.csv"
)

SHARED_REPORT_PATH = os.path.join(
    REPORT_FOLDER, "cleaning_report.json"
)

FORECAST_RESULT_PATH = os.path.join(
    FORECAST_FOLDER, "forecast_results.csv"
)

FORECAST_REPORT_PATH = os.path.join(
    FORECAST_FOLDER, "forecast_report.json"
)

REPORTS_DASHBOARD_PATH = os.path.join(
    REPORT_FOLDER,
    "reports_dashboard.json"
)

INVENTORY_RESULT_PATH = os.path.join(
    REPORT_FOLDER,
    "inventory_optimization_results.csv"
)

INVENTORY_MODEL_INFO_PATH = os.path.join(
    REPORT_FOLDER,
    "inventory_model_info.json"
)

MODULE_RESULT_CACHE = {}


def generate_reports_dashboard(dashboard_path=REPORTS_DASHBOARD_PATH):
    """
    Generate or refresh the local reports dashboard payload from the
    artifact files already delivered by the workspace pipeline. This file
    should remain artifact-backed and should avoid dataset reprocessing.
    """
    if os.path.exists(dashboard_path):
        try:
            with open(dashboard_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass

    dashboard = {
        "available": True,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": {
            "dataset_file": SHARED_DATASET_PATH,
            "dataset_rows": int(get_shared_dataset_info().get("rows", 0)),
            "dataset_columns": int(get_shared_dataset_info().get("columns", 0)),
            "forecast_file": FORECAST_REPORT_PATH,
            "inventory_file": INVENTORY_RESULT_PATH,
            "inventory_model_file": INVENTORY_MODEL_INFO_PATH,
        },
        "kpis": {
            "records": int(get_shared_dataset_info().get("rows", 0)),
            "columns": int(get_shared_dataset_info().get("columns", 0)),
            "products": int(get_shared_dataset_info().get("rows", 0)),
            "forecast_horizon": 30,
            "forecast_best_model": "Random Forest",
            "forecast_best_mae": 0.0,
            "forecast_best_rmse": 0.0,
            "inventory_rows": int(get_shared_dataset_info().get("rows", 0)),
            "stockout_risk_rows": 0,
            "low_stock_rows": 0,
            "overstock_rows": 0,
            "stock_cover_days": 0.0,
        },
        "sales_demand_trend": [],
        "alerts": {
            "total": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "message": "No alert summary available."
        },
        "tables": {
            "top_products": [],
            "inventory_snapshot": []
        },
        "recommendations": []
    }

    with open(dashboard_path, "w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, indent=2)

    return dashboard


def load_reports_dashboard(dashboard_path=REPORTS_DASHBOARD_PATH):
    """
    Load the cached artifact-backed reports dashboard. If the artifact is
    missing, generate an artifact from the existing files in the workspace.
    """
    if os.path.exists(dashboard_path):
        try:
            with open(dashboard_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass

    return generate_reports_dashboard(dashboard_path)


def build_reports_powerbi_context(
    shared_dataset_path=SHARED_DATASET_PATH,
    forecast_report_path=FORECAST_REPORT_PATH,
    inventory_results_path=INVENTORY_RESULT_PATH,
):
    """
    Compatibility shim for the older Power BI reports contract. It reads
    existing workspace artifacts instead of forcing a manual report URL.
    """
    context = {
        "enabled": True,
        "report_title": "Automated Retail Dashboard",
        "report_load_mode": "iframe",
        "report_url": "",
        "fallback_message": "Local artifact-backed reports dashboard available.",
        "dataset_rows": 0,
        "dataset_columns": 0,
        "forecast_metrics_available": False,
        "inventory_available": False,
        "inventory_rows": 0,
        "forecast_horizon": 30,
        "forecast_best_model": "Random Forest",
        "forecast_best_mae": 0.0,
        "forecast_best_rmse": 0.0,
        "forecast_best_mape": 0.0,
        "forecast_file": forecast_report_path,
        "inventory_file": inventory_results_path,
    }

    if os.path.exists(shared_dataset_path):
        try:
            df = pd.read_csv(shared_dataset_path)
            context["dataset_rows"] = int(len(df))
            context["dataset_columns"] = int(len(df.columns))
        except Exception:
            pass

    if os.path.exists(forecast_report_path):
        try:
            with open(forecast_report_path, "r", encoding="utf-8") as handle:
                forecast_data = json.load(handle)
            if isinstance(forecast_data, dict):
                context["forecast_metrics_available"] = True
                context["forecast_best_model"] = (
                    forecast_data.get("best_model")
                    or forecast_data.get("model")
                    or context["forecast_best_model"]
                )
                context["forecast_best_mae"] = (
                    forecast_data.get("mae")
                    or forecast_data.get("mean_absolute_error")
                    or context["forecast_best_mae"]
                )
                context["forecast_best_rmse"] = (
                    forecast_data.get("rmse")
                    or forecast_data.get("root_mean_squared_error")
                    or context["forecast_best_rmse"]
                )
                context["forecast_best_mape"] = (
                    forecast_data.get("mape")
                    or forecast_data.get("mean_absolute_percentage_error")
                    or context["forecast_best_mape"]
                )
        except Exception:
            pass

    if os.path.exists(inventory_results_path):
        try:
            inventory_df = pd.read_csv(inventory_results_path)
            context["inventory_available"] = True
            context["inventory_rows"] = int(len(inventory_df))
        except Exception:
            pass

    context["report_url"] = ""
    context["report_load_mode"] = "fallback" if not context["report_url"] else "iframe"

    return context


def get_cached_module_result(
    module_name,
    source_context,
    dataset,
    compute_fn
):
    """
    Cache expensive ML/service outputs per module and dataset
    fingerprint so the route page stays fast for repeated page visits.
    """
    try:
        source = (
            source_context.get(
                "selected_dataset_source",
                "shared"
            )
            or "shared"
        )

        dataset_filename = (
            source_context.get(
                "selected_dataset_filename"
            )
            or "shared_dataset"
        )

        dataset_rows = (
            len(dataset)
            if dataset is not None
            and hasattr(dataset, "shape")
            else 0
        )

        dataset_columns = (
            len(dataset.columns)
            if dataset is not None
            and hasattr(dataset, "columns")
            else 0
        )

        cache_key = (
            f"{module_name}|{source}|{dataset_filename}|"
            f"{dataset_rows}|{dataset_columns}"
        )

        cache_store = MODULE_RESULT_CACHE.setdefault(
            module_name,
            {}
        )

        if cache_key in cache_store:
            return cache_store[cache_key]

        result = compute_fn()
        cache_store[cache_key] = result
        return result

    except Exception:
        return compute_fn()


for folder in [
    UPLOAD_FOLDER,
    RAW_FOLDER,
    CLEANED_FOLDER,
    REPORT_FOLDER,
    FORECAST_FOLDER,
    MODULE_DATA_FOLDER
]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please login to access this page."


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def role_required(*allowed_roles):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please login to continue.", "warning")
                return redirect(url_for("login"))

            if current_user.role not in allowed_roles:
                flash(
                    "You do not have permission to access this page.",
                    "danger"
                )
                return redirect(url_for("dashboard"))

            return function(*args, **kwargs)

        return wrapper

    return decorator


def read_csv_safely(file_path):
    try:
        return pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=False
        )


def save_cleaning_report(report):
    try:
        with open(
            SHARED_REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )
        return True
    except Exception:
        return False


def load_shared_dataset():
    if not os.path.exists(SHARED_DATASET_PATH):
        return None

    try:
        return read_csv_safely(SHARED_DATASET_PATH)
    except Exception:
        return None


def shared_dataset_available():
    return os.path.exists(SHARED_DATASET_PATH)


def get_shared_dataset_info():
    result = {
        "available": False,
        "filename": None,
        "rows": 0,
        "columns": 0,
        "missing_values": 0,
        "duplicate_rows": 0,
        "columns_list": [],
        "mapping": {}
    }

    if not shared_dataset_available():
        return result


    try:
        df = load_shared_dataset()

        if df is None or df.empty:
            return result

        result["available"] = True
        result["filename"] = os.path.basename(SHARED_DATASET_PATH)
        result["rows"] = int(len(df))
        result["columns"] = int(len(df.columns))
        result["missing_values"] = int(df.isnull().sum().sum())
        result["duplicate_rows"] = int(df.duplicated().sum())
        result["columns_list"] = list(df.columns)
        result["mapping"] = automatic_column_mapping(df)

        return result

    except Exception:
        return result


MODULE_NAMES = {
    "forecasting",
    "inventory",
    "analytics",
    "seasonal",
    "alerts",
    "reports"
}


def module_source_path(module_name):
    return os.path.join(
        MODULE_DATA_FOLDER,
        f"{module_name}_independent.csv"
    )


def load_module_source_metadata():
    try:
        with open(
            MODULE_SOURCE_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_module_source_metadata(metadata):
    with open(
        MODULE_SOURCE_METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4, default=str)


def get_module_dataset(module_name, source="shared"):
    if source == "independent":
        path = module_source_path(module_name)
        if os.path.exists(path):
            try:
                return read_csv_safely(path), "independent", path
            except Exception:
                pass

    if shared_dataset_available():
        dataset = load_shared_dataset()
        if dataset is not None:
            return dataset, "shared", SHARED_DATASET_PATH

    return None, None, None


def get_module_source_context(module_name):
    shared_info = get_shared_dataset_info()
    independent_path = module_source_path(module_name)
    metadata = load_module_source_metadata().get(module_name, {})
    independent_available = os.path.exists(independent_path)
    selected_source = request.args.get("source", "shared")

    if selected_source == "independent" and not independent_available:
        selected_source = "shared"

    dataset, source, path = get_module_dataset(
        module_name,
        selected_source
    )

    return {
        "shared_dataset_available": shared_info["available"],
        "shared_dataset_filename": shared_info["filename"],
        "shared_dataset_rows": shared_info["rows"],
        "shared_dataset_columns": shared_info["columns"],
        "independent_dataset_available": independent_available,
        "independent_dataset_filename": metadata.get("filename"),
        "selected_dataset_source": source or "shared",
        "selected_dataset_filename": (
            os.path.basename(path) if path else None
        ),
        "selected_dataset": dataset
    }


# ============================================================
# REPORTS POWER BI CONTEXT
# ============================================================


def find_column(df, candidates):
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    for normalized_name, original_name in normalized.items():
        for candidate in candidates:
            candidate_normalized = normalize_column_name(candidate)

            if (
                candidate_normalized in normalized_name
                or normalized_name in candidate_normalized
            ):
                return original_name

    return None


def calculate_inventory_statistics(dataset=None):
    result = {
        "inventory_available": False,
        "total_products": None,
        "low_stock": None,
        "overstock": None,
        "reorder_required": None,
        "average_daily_demand": None,
        "lead_time": None,
        "safety_stock": None,
        "reorder_point": None,
        "inventory_message": "No shared dataset has been uploaded yet."
    }

    df = dataset if dataset is not None else load_shared_dataset()

    if df is None:
        return result

    if df.empty:
        result["inventory_message"] = (
            "The shared dataset contains no data."
        )
        return result

    product_column = find_column(
        df,
        [
            "product",
            "product_name",
            "product_id",
            "sku",
            "item",
            "item_id"
        ]
    )

    inventory_column = find_column(
        df,
        [
            "inventory",
            "stock",
            "current_stock",
            "stock_level",
            "stock_quantity",
            "available_stock",
            "quantity_in_stock",
            "on_hand"
        ]
    )

    demand_column = find_column(
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
    )

    sales_column = find_column(
        df,
        [
            "sales",
            "sales_amount",
            "revenue",
            "total_sales"
        ]
    )

    lead_time_column = find_column(
        df,
        [
            "lead_time",
            "leadtime",
            "delivery_time",
            "supplier_lead_time",
            "days_to_delivery"
        ]
    )

    reorder_column = find_column(
        df,
        [
            "reorder_level",
            "reorder_point",
            "reorder_threshold",
            "minimum_stock",
            "min_stock",
            "safety_level"
        ]
    )

    safety_stock_column = find_column(
        df,
        ["safety_stock"]
    )

    maximum_stock_column = find_column(
        df,
        [
            "maximum_stock",
            "max_stock",
            "max_stock_level",
            "overstock_threshold"
        ]
    )

    if product_column:
        result["total_products"] = int(
            df[product_column].nunique()
        )
    else:
        result["total_products"] = int(len(df))

    if inventory_column:
        stock = pd.to_numeric(
            df[inventory_column],
            errors="coerce"
        )

        valid_stock = stock.dropna()

        if not valid_stock.empty:
            if reorder_column:
                threshold = pd.to_numeric(
                    df[reorder_column],
                    errors="coerce"
                )

                result["low_stock"] = int(
                    (stock < threshold)
                    .fillna(False)
                    .sum()
                )

                result["reorder_required"] = int(
                    (stock <= threshold)
                    .fillna(False)
                    .sum()
                )

            if maximum_stock_column:
                maximum_stock = pd.to_numeric(
                    df[maximum_stock_column],
                    errors="coerce"
                )

                result["overstock"] = int(
                    (stock > maximum_stock)
                    .fillna(False)
                    .sum()
                )

    actual_demand_column = demand_column or sales_column

    if actual_demand_column:
        demand_values = pd.to_numeric(
            df[actual_demand_column],
            errors="coerce"
        ).dropna()

        if not demand_values.empty:
            result["average_daily_demand"] = round(
                float(demand_values.mean()),
                2
            )

    if lead_time_column:
        lead_values = pd.to_numeric(
            df[lead_time_column],
            errors="coerce"
        ).dropna()

        if not lead_values.empty:
            result["lead_time"] = round(
                float(lead_values.mean()),
                2
            )

    if safety_stock_column:
        safety_values = pd.to_numeric(
            df[safety_stock_column],
            errors="coerce"
        ).dropna()

        if not safety_values.empty:
            result["safety_stock"] = round(
                float(safety_values.mean()),
                2
            )

    average_demand = result["average_daily_demand"]
    lead_time = result["lead_time"]

    if average_demand is not None and lead_time is not None:
        result["reorder_point"] = round(
            average_demand * lead_time,
            2
        )

    result["inventory_available"] = True
    result["inventory_message"] = (
        "Inventory information is calculated "
        "from the shared cleaned dataset."
    )

    return result


# ============================================================
# FORECASTING ENGINE
# ============================================================

FORECAST_HORIZON = 30


def get_forecasting_columns(df):
    mapping = automatic_column_mapping(df)

    date_column = mapping.get("date")
    demand_column = mapping.get("demand")
    sales_column = mapping.get("sales")

    target_column = demand_column or sales_column

    return date_column, target_column, mapping


def prepare_forecasting_data(df):
    date_column, target_column, mapping = (
        get_forecasting_columns(df)
    )

    result = {
        "success": False,
        "message": "",
        "date_column": date_column,
        "target_column": target_column,
        "mapping": mapping,
        "data": None
    }

    if not date_column:
        result["message"] = (
            "No date column was detected for forecasting."
        )
        return result

    if not target_column:
        result["message"] = (
            "No demand or sales column was detected for forecasting."
        )
        return result

    if (
        date_column not in df.columns
        or target_column not in df.columns
    ):
        result["message"] = (
            "Required forecasting columns are not available."
        )
        return result

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column, target_column]
    )

    if data.empty:
        result["message"] = (
            "No valid date and demand records are available."
        )
        return result

    data = data[data[target_column] >= 0]

    if data.empty:
        result["message"] = (
            "No valid non-negative demand records are available."
        )
        return result

    data = (
        data
        .groupby(
            date_column,
            as_index=False
        )[target_column]
        .sum()
    )

    data = data.sort_values(date_column)
    data = data.reset_index(drop=True)

    if len(data) < 10:
        result["message"] = (
            "At least 10 valid time-series records "
            "are recommended for forecasting."
        )
        return result

    result["success"] = True
    result["message"] = (
        "Forecasting data prepared successfully."
    )
    result["data"] = data

    return result


def create_time_features(dates):
    features = pd.DataFrame(
        index=range(len(dates))
    )

    dates = pd.to_datetime(dates)

    features["year"] = dates.dt.year
    features["month"] = dates.dt.month
    features["day"] = dates.dt.day
    features["day_of_week"] = dates.dt.dayofweek
    features["day_of_year"] = dates.dt.dayofyear

    features["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
        .values
    )

    features["quarter"] = dates.dt.quarter

    return features


def run_random_forest_forecast(
    time_series,
    horizon=30
):
    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    dates = time_series[date_column]
    values = time_series[target_column]

    features = create_time_features(dates)

    total_records = len(features)

    test_size = max(
        3,
        int(total_records * 0.2)
    )

    if total_records - test_size < 5:
        test_size = 3

    X_train = features.iloc[:-test_size]
    X_test = features.iloc[-test_size:]

    y_train = values.iloc[:-test_size]
    y_test = values.iloc[-test_size:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------
    #
    # Refit the model on all historical data so that the chart
    # shows predictions for the complete historical dataset.
    #
    # --------------------------------------------------------

    historical_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    historical_model.fit(
        features,
        values
    )

    historical_predictions = historical_model.predict(
        features
    )

    historical_comparison = pd.DataFrame({
        "date": dates,
        "actual": values,
        "predicted": np.maximum(
            historical_predictions,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    last_date = pd.to_datetime(
        dates.max()
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    future_features = create_time_features(
        future_dates
    )

    future_predictions = historical_model.predict(
        future_features
    )

    future_predictions = np.maximum(
        future_predictions,
        0
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": np.round(
            future_predictions,
            2
        )
    })

    return {
        "model": "Random Forest",
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": forecast_df,
        "historical_comparison": historical_comparison
    }


def run_gradient_boosting_forecast(
    time_series,
    horizon=30
):
    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    dates = time_series[date_column]
    values = time_series[target_column]

    features = create_time_features(dates)

    total_records = len(features)

    test_size = max(
        3,
        int(total_records * 0.2)
    )

    if total_records - test_size < 5:
        test_size = 3

    X_train = features.iloc[:-test_size]
    X_test = features.iloc[-test_size:]

    y_train = values.iloc[:-test_size]
    y_test = values.iloc[-test_size:]

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------

    historical_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    historical_model.fit(
        features,
        values
    )

    historical_predictions = historical_model.predict(
        features
    )

    historical_comparison = pd.DataFrame({
        "date": dates,
        "actual": values,
        "predicted": np.maximum(
            historical_predictions,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    last_date = pd.to_datetime(
        dates.max()
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    future_features = create_time_features(
        future_dates
    )

    future_predictions = historical_model.predict(
        future_features
    )

    future_predictions = np.maximum(
        future_predictions,
        0
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": np.round(
            future_predictions,
            2
        )
    })

    return {
        "model": "Gradient Boosting",
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": forecast_df,
        "historical_comparison": historical_comparison
    }


def run_prophet_forecast(
    time_series,
    horizon=30
):
    try:
        from prophet import Prophet
    except ImportError:
        return {
            "model": "Prophet",
            "available": False,
            "error": "Prophet is not installed.",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None
        }

    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    prophet_df = time_series[
        [date_column, target_column]
    ].copy()

    prophet_df.columns = ["ds", "y"]

    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"]
    )

    prophet_df["y"] = pd.to_numeric(
        prophet_df["y"],
        errors="coerce"
    )

    prophet_df = prophet_df.dropna()

    if len(prophet_df) < 10:
        return {
            "model": "Prophet",
            "available": True,
            "error": "Not enough records for Prophet.",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None
        }

    test_size = max(
        3,
        int(len(prophet_df) * 0.2)
    )

    train_df = prophet_df.iloc[:-test_size]
    test_df = prophet_df.iloc[-test_size:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df)

    test_dates = test_df[["ds"]]

    test_prediction = model.predict(
        test_dates
    )

    predicted_test = test_prediction["yhat"].values
    actual_test = test_df["y"].values

    mae = mean_absolute_error(
        actual_test,
        predicted_test
    )

    rmse = math.sqrt(
        mean_squared_error(
            actual_test,
            predicted_test
        )
    )

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    final_model.fit(prophet_df)

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------

    historical_forecast = final_model.predict(
        prophet_df[["ds"]]
    )

    historical_comparison = pd.DataFrame({
        "date": historical_forecast["ds"],
        "actual": prophet_df["y"].values,
        "predicted": np.maximum(
            historical_forecast["yhat"].values,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    future = final_model.make_future_dataframe(
        periods=horizon,
        freq="D"
    )

    prediction = final_model.predict(
        future
    )

    future_prediction = (
        prediction.tail(horizon)[
            ["ds", "yhat"]
        ]
        .copy()
    )

    future_prediction.columns = [
        "date",
        "predicted_demand"
    ]

    future_prediction["predicted_demand"] = np.maximum(
        future_prediction["predicted_demand"],
        0
    )

    future_prediction["predicted_demand"] = (
        future_prediction["predicted_demand"]
        .round(2)
    )

    return {
        "model": "Prophet",
        "available": True,
        "error": None,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": future_prediction,
        "historical_comparison": historical_comparison
    }


def save_forecast_report(report):
    try:
        with open(
            FORECAST_REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        return True

    except Exception:
        return False


def generate_service_forecast_result(dataset):
    result = {
        "success": False,
        "message": "",
        "date_column": None,
        "target_column": None,
        "models": [],
        "best_model": None,
        "best_mse": None,
        "best_mae": None,
        "best_rmse": None,
        "forecast": None,
        "forecast_rows": 0,
        "historical_comparison": None
    }

    if dataset is None:
        result["message"] = "No dataset is available for forecasting."
        return result

    if dataset.empty:
        result["message"] = "The selected dataset is empty."
        return result

    date_column, target_column, mapping = get_forecasting_columns(dataset)
    result["date_column"] = date_column
    result["target_column"] = target_column

    if not date_column or not target_column:
        result["message"] = (
            "A valid date and demand or sales column are required."
        )
        return result

    try:
        trained = train_forecasting_models(
            dataset,
            date_column,
            target_column
        )
        prepared_data = trained["data"]
        best_model = trained["best_model"]
        best_name = trained["best_model_name"]
        metrics = trained["best_metrics"]

        future_forecasts = {}
        for name, model_data in trained["models"].items():
            model_forecast = generate_service_forecast(
                model_data["model"],
                prepared_data[date_column].max(),
                FORECAST_HORIZON,
                history=prepared_data,
                target_column=target_column,
                feature_columns=trained["feature_columns"]
            )
            future_forecasts[name] = model_forecast.rename(
                columns={"forecast": "predicted_demand"}
            )

        future = future_forecasts[best_name]

        features = prepared_data[trained["feature_columns"]]
        historical_predictions = best_model.predict(features)
        historical = pd.DataFrame({
            "date": prepared_data[date_column],
            "actual": prepared_data[target_column].round(2),
            "predicted": np.maximum(historical_predictions, 0).round(2)
        })

        result["models"] = []
        for name, model_data in trained["models"].items():
            model_metrics = model_data["metrics"]
            result["models"].append({
                "model": name,
                "mse": model_metrics.get("mse"),
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"],
                "forecast": future_forecasts[name],
                "available": True,
                "error": None
            })

        result["best_model"] = best_name
        result["best_mse"] = metrics.get("mse")
        result["best_mae"] = metrics["mae"]
        result["best_rmse"] = metrics["rmse"]
        result["forecast"] = future
        result["forecast_rows"] = int(len(future))
        result["historical_comparison"] = historical

        save_df = future.copy()
        save_df["model"] = best_name
        save_df.to_csv(FORECAST_RESULT_PATH, index=False)
        save_forecast_report({
            "date_column": date_column,
            "target_column": target_column,
            "forecast_horizon": FORECAST_HORIZON,
            "best_model": best_name,
            "best_mse": metrics.get("mse"),
            "best_mae": metrics["mae"],
            "best_rmse": metrics["rmse"],
            "models": [
                {
                    "model": item["model"],
                    "mse": item.get("mse"),
                    "mae": item["mae"],
                    "rmse": item["rmse"],
                    "r2": item["r2"]
                }
                for item in result["models"]
            ],
            "forecast_file": FORECAST_RESULT_PATH
        })

        result["success"] = True
        result["message"] = "Forecast generated using the forecasting service."
        return result

    except Exception as error:
        result["message"] = f"Forecasting service failed: {error}"
        return result


def generate_forecast(dataset=None):
    return generate_service_forecast_result(
        dataset if dataset is not None else load_shared_dataset()
    )

    # Legacy inline implementation retained below for compatibility.
    result = {
        "success": False,
        "message": "",
        "date_column": None,
        "target_column": None,
        "models": [],
        "best_model": None,
        "best_mae": None,
        "best_rmse": None,
        "forecast": None,
        "forecast_rows": 0,
        "historical_comparison": None
    }

    df = dataset if dataset is not None else load_shared_dataset()

    if df is None:
        result["message"] = (
            "No shared dataset is available."
        )
        return result

    if df.empty:
        result["message"] = (
            "The shared dataset is empty."
        )
        return result

    preparation = prepare_forecasting_data(df)

    if not preparation["success"]:
        result["message"] = preparation["message"]
        result["date_column"] = preparation["date_column"]
        result["target_column"] = preparation["target_column"]
        return result

    time_series = preparation["data"]

    result["date_column"] = preparation["date_column"]
    result["target_column"] = preparation["target_column"]

    try:
        rf_result = run_random_forest_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(rf_result)

    except Exception as e:
        result["models"].append({
            "model": "Random Forest",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    try:
        gb_result = run_gradient_boosting_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(gb_result)

    except Exception as e:
        result["models"].append({
            "model": "Gradient Boosting",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    try:
        prophet_result = run_prophet_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(prophet_result)

    except Exception as e:
        result["models"].append({
            "model": "Prophet",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    valid_models = [
        model
        for model in result["models"]
        if (
            model.get("forecast") is not None
            and model.get("rmse") is not None
        )
    ]

    if not valid_models:
        result["message"] = (
            "No forecasting model could be executed successfully."
        )
        return result

    best_model = min(
        valid_models,
        key=lambda x: x["rmse"]
    )

    result["best_model"] = best_model["model"]
    result["best_mae"] = best_model["mae"]
    result["best_rmse"] = best_model["rmse"]
    result["forecast"] = best_model["forecast"]
    result["forecast_rows"] = int(
        len(best_model["forecast"])
    )

    # ========================================================
    # HISTORICAL CHART DATA
    # ========================================================

    historical_comparison = best_model.get(
        "historical_comparison"
    )

    if historical_comparison is not None:
        result["historical_comparison"] = (
            historical_comparison.copy()
        )

    try:
        save_df = best_model["forecast"].copy()

        save_df["model"] = best_model["model"]

        save_df.to_csv(
            FORECAST_RESULT_PATH,
            index=False
        )

    except Exception as e:
        result["message"] = (
            f"Forecast generated, but result "
            f"could not be saved: {str(e)}"
        )
        return result

    model_report = []

    for model in result["models"]:
        model_report.append({
            "model": model.get("model"),
            "mae": model.get("mae"),
            "rmse": model.get("rmse"),
            "available": model.get("available", True),
            "error": model.get("error")
        })

    report = {
        "date_column": result["date_column"],
        "target_column": result["target_column"],
        "forecast_horizon": FORECAST_HORIZON,
        "best_model": result["best_model"],
        "best_mae": result["best_mae"],
        "best_rmse": result["best_rmse"],
        "models": model_report,
        "forecast_file": FORECAST_RESULT_PATH
    }

    save_forecast_report(report)

    result["success"] = True
    result["message"] = (
        "Forecast generated successfully."
    )

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        role = request.form.get(
            "role",
            "Business Analyst"
        )

        if not name:
            flash("Please enter your name.", "danger")
            return redirect(url_for("register"))

        if not email:
            flash("Please enter your email.", "danger")
            return redirect(url_for("register"))

        if not password:
            flash("Please enter a password.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )
            return redirect(url_for("register"))

        allowed_roles = {
            "Admin",
            "Inventory Manager",
            "Business Analyst"
        }

        if role not in allowed_roles:
            role = "Business Analyst"

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash(
                "An account with this email already exists.",
                "warning"
            )
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            role=role
        )

        db.session.add(new_user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if (
            user
            and
            check_password_hash(
                user.password,
                password
            )
        ):
            login_user(user)

            flash(
                "Welcome back!",
                "success"
            )

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template("login.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    dataset_info = get_shared_dataset_info()

    return render_template(
        "dashboard.html",
        user=current_user,
        shared_dataset_available=dataset_info["available"],
        shared_dataset_rows=dataset_info["rows"],
        shared_dataset_columns=dataset_info["columns"]
    )


# ============================================================
# DATASET MANAGEMENT
# ADMIN ONLY
# ============================================================

@app.route("/datasets")
@login_required
@role_required("Admin")
def datasets():
    dataset_info = get_shared_dataset_info()

    return render_template(
        "dataset_upload.html",
        shared_dataset_available=dataset_info["available"],
        shared_dataset_rows=dataset_info["rows"],
        shared_dataset_columns=dataset_info["columns"]
    )


# ============================================================
# DATASET UPLOAD + AUTOMATIC CLEANING
# ADMIN ONLY
# ============================================================

@app.route(
    "/upload-dataset",
    methods=["POST"]
)
@login_required
@role_required("Admin")
def upload_dataset():

    dataset_type = request.form.get(
        "dataset_type",
        ""
    ).strip().lower()

    allowed_dataset_types = {
        "sales",
        "inventory",
        "product",
        "supplier",
        "weather",
        "holiday",
        "other"
    }

    if dataset_type not in allowed_dataset_types:
        flash(
            "Please select a valid dataset type.",
            "danger"
        )
        return redirect(url_for("datasets"))

    if "dataset" not in request.files:
        flash(
            "No dataset file was selected.",
            "danger"
        )
        return redirect(url_for("datasets"))

    file = request.files["dataset"]

    if file.filename == "":
        flash(
            "Please select a CSV file.",
            "danger"
        )
        return redirect(url_for("datasets"))

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        flash(
            "Only CSV files are currently supported.",
            "danger"
        )
        return redirect(url_for("datasets"))

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:
        file.save(upload_path)

    except Exception as e:
        flash(
            f"Could not save the uploaded file: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    try:
        original_df = read_csv_safely(upload_path)

    except pd.errors.EmptyDataError:
        flash(
            "The uploaded CSV file is empty.",
            "danger"
        )
        return redirect(url_for("datasets"))

    except pd.errors.ParserError:
        flash(
            "The CSV file has an invalid structure and could not be read.",
            "danger"
        )
        return redirect(url_for("datasets"))

    except Exception as e:
        flash(
            f"Could not process the CSV file: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    if original_df.empty:
        flash(
            "The uploaded CSV contains no data.",
            "danger"
        )
        return redirect(url_for("datasets"))

    original_rows = len(original_df)
    original_columns = len(original_df.columns)

    original_missing_values = int(
        original_df.isnull().sum().sum()
    )

    original_duplicate_rows = int(
        original_df.duplicated().sum()
    )

    raw_filename = f"{dataset_type}_{filename}"
    raw_path = os.path.join(
        RAW_FOLDER,
        raw_filename
    )

    try:
        original_df.to_csv(
            raw_path,
            index=False
        )
    except Exception:
        flash(
            "Dataset was uploaded, but the raw copy could not be saved.",
            "warning"
        )

    try:
        cleaned_df, cleaning_report = clean_dataset(
            original_df,
            remove_outliers=False
        )

    except Exception as e:
        flash(
            f"Dataset cleaning failed: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    try:
        cleaned_df.to_csv(
            SHARED_DATASET_PATH,
            index=False
        )

    except Exception as e:
        flash(
            f"Cleaned dataset could not be saved: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    report_saved = save_cleaning_report(
        cleaning_report
    )

    column_mapping = cleaning_report.get(
        "automatic_column_mapping",
        {}
    )

    mapped_count = sum(
        1
        for value in column_mapping.values()
        if value is not None
    )

    total_mapping_fields = len(column_mapping)

    has_date = (
        column_mapping.get("date") is not None
    )

    has_demand = (
        column_mapping.get("demand") is not None
    )

    has_sales = (
        column_mapping.get("sales") is not None
    )

    forecasting_ready = (
        has_date
        and
        (has_demand or has_sales)
    )

    cleaned_rows = len(cleaned_df)
    cleaned_columns = len(cleaned_df.columns)

    cleaned_missing_values = int(
        cleaned_df.isnull().sum().sum()
    )

    cleaned_duplicate_rows = int(
        cleaned_df.duplicated().sum()
    )

    preview_df = cleaned_df.head(10)

    preview_html = preview_df.to_html(
        classes=(
            "table "
            "table-hover "
            "table-bordered"
        ),
        index=False
    )

    flash(
        "Dataset uploaded and cleaned successfully. "
        "The cleaned dataset is now the shared dataset "
        "for Forecasting, Inventory, Analytics and Reports.",
        "success"
    )

    return render_template(
        "dataset_upload.html",
        upload_success=True,
        filename=filename,
        dataset_type=dataset_type,
        total_rows=original_rows,
        total_columns=original_columns,
        missing_values=original_missing_values,
        duplicate_rows=original_duplicate_rows,
        cleaned_rows=cleaned_rows,
        cleaned_columns=cleaned_columns,
        cleaned_missing_values=cleaned_missing_values,
        cleaned_duplicate_rows=cleaned_duplicate_rows,
        columns=list(cleaned_df.columns),
        preview_html=preview_html,
        column_mapping=column_mapping,
        mapped_count=mapped_count,
        total_mapping_fields=total_mapping_fields,
        cleaned_success=True,
        cleaning_report=cleaning_report,
        report_saved=report_saved,
        forecasting_ready=forecasting_ready,
        shared_dataset_available=True,
        shared_dataset_rows=cleaned_rows,
        shared_dataset_columns=cleaned_columns
    )


# ============================================================
# INDEPENDENT MODULE DATASET UPLOAD
# ============================================================

@app.route(
    "/module-upload/<module_name>",
    methods=["POST"]
)
@login_required
def upload_module_dataset(module_name):

    if module_name not in MODULE_NAMES:
        flash("That module does not support independent datasets.", "danger")
        return redirect(url_for("dashboard"))

    file = request.files.get("dataset")

    if file is None or file.filename == "":
        flash("Please select a CSV file.", "danger")
        return redirect(url_for(module_name))

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        flash("Only CSV files are currently supported.", "danger")
        return redirect(url_for(module_name))

    temporary_path = os.path.join(
        UPLOAD_FOLDER,
        f"{module_name}_{filename}"
    )
    independent_path = module_source_path(module_name)

    try:
        file.save(temporary_path)
        original_df = read_csv_safely(temporary_path)

        if original_df.empty:
            raise ValueError("The uploaded CSV contains no data.")

        cleaned_df, cleaning_report = clean_dataset(
            original_df,
            remove_outliers=False
        )
        cleaned_df.to_csv(independent_path, index=False)

        metadata = load_module_source_metadata()
        metadata[module_name] = {
            "filename": filename,
            "rows": int(len(cleaned_df)),
            "columns": int(len(cleaned_df.columns)),
            "cleaning_report": cleaning_report
        }
        save_module_source_metadata(metadata)

    except Exception as error:
        flash(f"Independent dataset could not be processed: {error}", "danger")
        return redirect(url_for(module_name))

    flash(
        f"{filename} is ready for independent {module_name} analysis. ",
        "success"
    )
    return redirect(
        url_for(module_name, source="independent")
    )


# ============================================================
# FORECASTING PAGE
# ============================================================

@app.route("/forecasting")
@login_required
def forecasting():

    source_context = get_module_source_context("forecasting")
    dataset_info = get_shared_dataset_info()

    forecast_result = None
    forecast_models = []
    best_model = None
    best_mae = None
    best_rmse = None
    forecast_rows = []
    forecasting_message = None

    # --------------------------------------------------------
    # EMPTY CHART VARIABLES
    # --------------------------------------------------------

    chart_dates = []
    chart_actual = []
    chart_predicted = []

    if source_context["selected_dataset"] is not None:

        df = source_context["selected_dataset"]

        if df is not None:

            preparation = prepare_forecasting_data(df)

            if preparation["success"]:

                forecast_result = get_cached_module_result(
                    "forecasting",
                    source_context,
                    df,
                    lambda: generate_forecast(df)
                )

                if forecast_result["success"]:

                    forecast_models = (
                        forecast_result["models"]
                    )

                    best_model = (
                        forecast_result["best_model"]
                    )

                    best_mae = (
                        forecast_result["best_mae"]
                    )

                    best_rmse = (
                        forecast_result["best_rmse"]
                    )

                    if forecast_result["forecast"] is not None:

                        forecast_rows = (
                            forecast_result["forecast"]
                            .copy()
                            .to_dict(
                                orient="records"
                            )
                        )

                    # ------------------------------------------------
                    # HISTORICAL CHART DATA
                    # ------------------------------------------------

                    historical_df = (
                        forecast_result
                        .get("historical_comparison")
                    )

                    if historical_df is not None:
                        historical_df = (
                            historical_df.copy()
                        )

                        historical_df["date"] = (
                            pd.to_datetime(
                                historical_df["date"]
                            )
                            .dt.strftime("%Y-%m-%d")
                        )

                        chart_dates = (
                            historical_df["date"]
                            .tolist()
                        )

                        chart_actual = (
                            historical_df["actual"]
                            .tolist()
                        )

                        chart_predicted = (
                            historical_df["predicted"]
                            .tolist()
                        )

                    forecasting_message = (
                        forecast_result["message"]
                    )

                else:

                    forecasting_message = (
                        forecast_result["message"]
                    )

            else:

                forecasting_message = (
                    preparation["message"]
                )

    return render_template(
        "forecasting.html",

        module_name="forecasting",

        shared_dataset_available=
            dataset_info["available"],

        shared_dataset_rows=
            dataset_info["rows"],

        shared_dataset_columns=
            dataset_info["columns"],

        shared_dataset_columns_list=
            dataset_info["columns_list"],

        column_mapping=
            dataset_info["mapping"],

        forecast_result=
            forecast_result,

        forecast_models=
            forecast_models,

        best_model=
            best_model,

        best_mae=
            best_mae,

        best_mse=
            forecast_result.get("best_mse")
            if forecast_result
            else None,

        best_rmse=
            best_rmse,

        forecast_rows=
            forecast_rows,

        forecasting_message=
            forecasting_message,

        # --------------------------------------------------------
        # CHART DATA
        # --------------------------------------------------------

        chart_dates=
            chart_dates,

        chart_actual=
            chart_actual,

        chart_predicted=
            chart_predicted,

        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )


# ============================================================
# RUN FORECASTING MANUALLY
# ============================================================

@app.route(
    "/run-forecasting",
    methods=["POST"]
)
@login_required
def run_forecasting():

    source_context = get_module_source_context("forecasting")

    if source_context["selected_dataset"] is None:

        flash(
            "No shared dataset is available. "
            "Please upload a dataset from Data Management first.",
            "warning"
        )

        return redirect(
            url_for("forecasting")
        )

    result = generate_forecast(source_context["selected_dataset"])

    if result["success"]:

        flash(
            f"Forecast generated successfully using "
            f"{result['best_model']}. "
            f"Forecast horizon: {FORECAST_HORIZON} days.",
            "success"
        )

    else:

        flash(
            result["message"],
            "danger"
        )

    return redirect(
        url_for("forecasting")
    )


# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

@app.route("/inventory")
@login_required
def inventory():

    # --------------------------------------------------------
    # Get the dataset selected by the shared dataset system
    # --------------------------------------------------------

    source_context = get_module_source_context("inventory")

    selected_dataset = source_context["selected_dataset"]


    # --------------------------------------------------------
    # Import the new inventory optimization service
    # --------------------------------------------------------

    from services.inventory_service import (
        run_inventory_optimization
    )


    # --------------------------------------------------------
    # Run AI + inventory optimization with request-level cache
    # --------------------------------------------------------

    inventory_result = get_cached_module_result(
        "inventory",
        source_context,
        selected_dataset,
        lambda: run_inventory_optimization(selected_dataset)
    )


    # --------------------------------------------------------
    # Extract summary information
    # --------------------------------------------------------

    summary = inventory_result.get(
        "summary",
        {}
    )


    # --------------------------------------------------------
    # Extract ML information
    # --------------------------------------------------------

    ml_data = inventory_result.get(
        "ml",
        {}
    )


    # --------------------------------------------------------
    # Render Inventory page
    # --------------------------------------------------------

    return render_template(

        "inventory.html",


        # ----------------------------------------------------
        # Module information
        # ----------------------------------------------------

        module_name="inventory",


        # ----------------------------------------------------
        # Dataset availability
        # ----------------------------------------------------

        inventory_available=inventory_result.get(
            "available",
            False
        ),


        inventory_message=inventory_result.get(
            "message",
            "No inventory data available."
        ),


        # ----------------------------------------------------
        # Main inventory statistics
        # ----------------------------------------------------

        total_products=summary.get(
            "total_products"
        ),

        low_stock=summary.get(
            "low_stock"
        ),

        overstock=summary.get(
            "overstock"
        ),

        reorder_required=summary.get(
            "reorder_required"
        ),


        # ----------------------------------------------------
        # Detailed inventory statistics
        # ----------------------------------------------------

        critical_items=summary.get(
            "critical_items",
            0
        ),

        high_risk_items=summary.get(
            "high_risk_items",
            0
        ),

        healthy_items=summary.get(
            "healthy_items",
            0
        ),


        average_daily_demand=summary.get(
            "average_daily_demand"
        ),

        lead_time=summary.get(
            "average_lead_time"
        ),

        safety_stock=summary.get(
            "average_safety_stock"
        ),

        reorder_point=summary.get(
            "average_reorder_point"
        ),


        total_recommended_order=summary.get(
            "total_recommended_order"
        ),

        average_stock_cover_days=summary.get(
            "average_stock_cover_days"
        ),


        # ----------------------------------------------------
        # Product-level inventory results
        # ----------------------------------------------------

        inventory_products=inventory_result.get(
            "products",
            []
        ),


        # ----------------------------------------------------
        # ML information
        # ----------------------------------------------------

        inventory_ml_available=ml_data.get(
            "available",
            False
        ),

        inventory_ml_accuracy=ml_data.get(
            "accuracy"
        ),

        inventory_ml_training_rows=ml_data.get(
            "training_rows"
        ),

        inventory_ml_message=ml_data.get(
            "message",
            ""
        ),

        inventory_feature_importance=ml_data.get(
            "feature_importance",
            {}
        ),


        # ----------------------------------------------------
        # Shared dataset context
        # ----------------------------------------------------

        **{
            key: value
            for key, value in source_context.items()
            if key != "selected_dataset"
        }

    )

# ============================================================
# ANALYTICS MODULE
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    # --------------------------------------------------------
    # Get shared / independent dataset
    # --------------------------------------------------------

    source_context = get_module_source_context("analytics")

    selected_dataset = source_context.get("selected_dataset")

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    analytics_available = False

    analytics_message = (
        "No shared dataset has been uploaded yet."
    )

    total_records = 0
    total_columns = 0

    total_sales = None
    total_demand = None
    products_analyzed = None
    data_period = None

    chart_dates = []
    chart_demand = []

    date_column = None
    demand_column = None
    sales_column = None
    product_column = None

    # --------------------------------------------------------
    # IMPORTANT:
    # Initialize every variable used by analytics.html
    # --------------------------------------------------------

    demand_drivers = {}

    product_analysis = []

    weekly_pattern = None
    monthly_pattern = None

    anomalies = []

    anomaly_count = 0
    anomaly_percentage = 0

    ml_accuracy = None
    ml_model = None

    feature_importance = {}

    business_insights = []

    # --------------------------------------------------------
    # Default ML result structure
    # --------------------------------------------------------

    analytics_ml = {

        "success": False,

        "classification": {
            "available": False,
            "message": "ML classification has not run yet."
        },

        "anomaly_detection": {
            "available": False,
            "message": "Anomaly detection has not run yet."
        },

        "product_analysis": {
            "available": False,
            "message": "Product analysis has not run yet."
        },

        "demand_drivers": {
            "available": False,
            "drivers": {}
        },

        "seasonality": {
            "available": False,
            "weekly_pattern": None,
            "monthly_pattern": None
        },

        "business_insights": []
    }

    # ========================================================
    # PROCESS DATASET
    # ========================================================

    if selected_dataset is not None:

        df = selected_dataset.copy()

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if not df.empty:

            total_records = int(len(df))
            total_columns = int(len(df.columns))

            # ------------------------------------------------
            # Automatic column mapping
            # ------------------------------------------------

            mapping = automatic_column_mapping(df)

            # ------------------------------------------------
            # Date column
            # ------------------------------------------------

            date_column = (
                mapping.get("date")
                or find_column(
                    df,
                    [
                        "date",
                        "order_date",
                        "sale_date",
                        "sales_date",
                        "transaction_date"
                    ]
                )
            )

            # ------------------------------------------------
            # Demand column
            # ------------------------------------------------

            demand_column = (
                mapping.get("demand")
                or find_column(
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
                )
            )

            # ------------------------------------------------
            # Sales column
            # ------------------------------------------------

            sales_column = (
                mapping.get("sales")
                or find_column(
                    df,
                    [
                        "sales",
                        "sales_amount",
                        "revenue",
                        "total_sales",
                        "amount"
                    ]
                )
            )

            # ------------------------------------------------
            # Product column
            # ------------------------------------------------

            product_column = (
                mapping.get("product")
                or find_column(
                    df,
                    [
                        "product",
                        "product_name",
                        "product_id",
                        "sku",
                        "item",
                        "item_id"
                    ]
                )
            )

            # =================================================
            # TOTAL SALES
            # =================================================

            if sales_column:

                sales_values = pd.to_numeric(
                    df[sales_column],
                    errors="coerce"
                ).dropna()

                if not sales_values.empty:

                    total_sales = round(
                        float(sales_values.sum()),
                        2
                    )

            # =================================================
            # TOTAL DEMAND
            # =================================================

            actual_demand_column = (
                demand_column
                or sales_column
            )

            if actual_demand_column:

                demand_values = pd.to_numeric(
                    df[actual_demand_column],
                    errors="coerce"
                )

                valid_demand = demand_values.dropna()

                # --------------------------------------------
                # Valid demand exists
                # --------------------------------------------

                if not valid_demand.empty:

                    analytics_available = True

                    total_demand = round(
                        float(valid_demand.sum()),
                        2
                    )

                    # ----------------------------------------
                    # Product count
                    # ----------------------------------------

                    if product_column:

                        products_analyzed = int(
                            df[product_column]
                            .nunique()
                        )

                    else:

                        products_analyzed = total_records

                    # ----------------------------------------
                    # Date information
                    # ----------------------------------------

                    if date_column:

                        date_values = pd.to_datetime(
                            df[date_column],
                            errors="coerce"
                        ).dropna()

                        if not date_values.empty:

                            start_date = (
                                date_values.min()
                                .strftime("%d %b %Y")
                            )

                            end_date = (
                                date_values.max()
                                .strftime("%d %b %Y")
                            )

                            data_period = (
                                f"{start_date} → {end_date}"
                            )

                            # --------------------------------
                            # Demand trend
                            # --------------------------------

                            trend_df = pd.DataFrame(
                                {
                                    "date": pd.to_datetime(
                                        df[date_column],
                                        errors="coerce"
                                    ),

                                    "demand": demand_values
                                }
                            )

                            trend_df = trend_df.dropna(
                                subset=[
                                    "date",
                                    "demand"
                                ]
                            )

                            if not trend_df.empty:

                                daily_data = (
                                    trend_df
                                    .groupby(
                                        "date",
                                        as_index=False
                                    )["demand"]
                                    .sum()
                                    .sort_values("date")
                                )

                                chart_dates = (
                                    daily_data["date"]
                                    .dt.strftime(
                                        "%Y-%m-%d"
                                    )
                                    .tolist()
                                )

                                chart_demand = [
                                    round(
                                        float(value),
                                        2
                                    )
                                    for value in
                                    daily_data["demand"]
                                ]

                    # =================================================
                    # RUN ANALYTICS ML WITH ROUTE CACHE
                    # =================================================

                    try:

                        analytics_ml = get_cached_module_result(
                            "analytics",
                            source_context,
                            selected_dataset,
                            lambda: run_analytics_ml(selected_dataset)
                        )

                    except Exception as error:

                        analytics_ml = {

                            "success": False,

                            "classification": {
                                "available": False,
                                "message": (
                                    f"Classification unavailable: {error}"
                                )
                            },

                            "anomaly_detection": {
                                "available": False
                            },

                            "product_analysis": {
                                "available": False
                            },

                            "demand_drivers": {
                                "available": False,
                                "drivers": {}
                            },

                            "seasonality": {
                                "available": False
                            },

                            "business_insights": [
                                (
                                    "Analytics ML pipeline "
                                    f"encountered an error: {error}"
                                )
                            ]
                        }

                    # =================================================
                    # CLASSIFICATION
                    # =================================================

                    classification = analytics_ml.get(
                        "classification",
                        {}
                    )

                    if classification.get("available"):

                        ml_accuracy = classification.get(
                            "accuracy"
                        )

                        ml_model = classification.get(
                            "model"
                        )

                        feature_importance = (
                            classification.get(
                                "feature_importance",
                                {}
                            )
                        )

                    # =================================================
                    # PRODUCT ANALYSIS
                    # =================================================

                    product_result = analytics_ml.get(
                        "product_analysis",
                        {}
                    )

                    if product_result.get("available"):

                        product_analysis = (
                            product_result.get(
                                "products",
                                []
                            )
                        )

                    # =================================================
                    # DEMAND DRIVERS
                    # =================================================

                    driver_result = analytics_ml.get(
                        "demand_drivers",
                        {}
                    )

                    if driver_result.get("available"):

                        demand_drivers = (
                            driver_result.get(
                                "drivers",
                                {}
                            )
                        )

                    else:

                        demand_drivers = {}

                    # =================================================
                    # SEASONALITY
                    # =================================================

                    seasonality_result = analytics_ml.get(
                        "seasonality",
                        {}
                    )

                    if seasonality_result.get("available"):

                        weekly_pattern = (
                            seasonality_result.get(
                                "weekly_pattern"
                            )
                        )

                        monthly_pattern = (
                            seasonality_result.get(
                                "monthly_pattern"
                            )
                        )

                    # =================================================
                    # ANOMALY DETECTION
                    # =================================================

                    anomaly_result = analytics_ml.get(
                        "anomaly_detection",
                        {}
                    )

                    if anomaly_result.get("available"):

                        anomalies = (
                            anomaly_result.get(
                                "anomalies",
                                []
                            )
                        )

                        anomaly_count = int(
                            anomaly_result.get(
                                "anomaly_count",
                                0
                            )
                        )

                        anomaly_percentage = float(
                            anomaly_result.get(
                                "anomaly_percentage",
                                0
                            )
                        )

                    # =================================================
                    # BUSINESS INSIGHTS
                    # =================================================

                    business_insights = (
                        analytics_ml.get(
                            "business_insights",
                            []
                        )
                    )

                    analytics_message = (
                        "Analytics and machine learning "
                        "analysis completed successfully."
                    )

                else:

                    analytics_message = (
                        "A demand or sales column was detected, "
                        "but it does not contain valid numeric values."
                    )

            else:

                analytics_message = (
                    "No demand or sales column could be detected "
                    "in the selected dataset."
                )

        else:

            analytics_message = (
                "The selected dataset is empty."
            )

    # ========================================================
    # RENDER ANALYTICS PAGE
    # ========================================================

    return render_template(

        "analytics.html",

        # ----------------------------------------------------
        # Basic analytics
        # ----------------------------------------------------

        analytics_available=analytics_available,

        analytics_message=analytics_message,

        total_records=total_records,

        total_columns=total_columns,

        total_sales=total_sales,

        total_demand=total_demand,

        products_analyzed=products_analyzed,

        data_period=data_period,

        # ----------------------------------------------------
        # Column mapping
        # ----------------------------------------------------

        date_column=date_column,

        demand_column=demand_column,

        sales_column=sales_column,

        product_column=product_column,

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        chart_dates=chart_dates,

        chart_demand=chart_demand,

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        analytics_ml=analytics_ml,

        ml_accuracy=ml_accuracy,

        ml_model=ml_model,

        feature_importance=feature_importance,

        # ----------------------------------------------------
        # Product analysis
        # ----------------------------------------------------

        product_analysis=product_analysis,

        # ----------------------------------------------------
        # Demand drivers
        # ----------------------------------------------------

        demand_drivers=demand_drivers,

        # ----------------------------------------------------
        # Seasonality
        # ----------------------------------------------------

        weekly_pattern=weekly_pattern,

        monthly_pattern=monthly_pattern,

        # ----------------------------------------------------
        # Anomalies
        # ----------------------------------------------------

        anomalies=anomalies,

        anomaly_count=anomaly_count,

        anomaly_percentage=anomaly_percentage,

        # ----------------------------------------------------
        # Business insights
        # ----------------------------------------------------

        business_insights=business_insights,

        # ----------------------------------------------------
        # Dataset source information
        # ----------------------------------------------------

        **{
            key: value
            for key, value in source_context.items()
            if key != "selected_dataset"
        }
    )
# ============================================================
# SEASONAL ANALYSIS MODULE
# ============================================================

@app.route("/seasonal")
@login_required
def seasonal():

    # --------------------------------------------------------
    # Get shared / independent dataset
    # --------------------------------------------------------

    source_context = get_module_source_context(
        "seasonal"
    )

    selected_dataset = source_context.get(
        "selected_dataset"
    )

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    seasonal_available = False

    seasonal_message = (
        "No shared dataset has been uploaded yet."
    )

    total_records = 0
    total_demand = None

    date_column = None
    demand_column = None
    product_column = None

    data_period = None

    # --------------------------------------------------------
    # Weekly
    # --------------------------------------------------------

    weekly_data = []

    peak_day = None
    peak_day_demand = None

    lowest_day = None
    lowest_day_demand = None

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    monthly_data = []

    peak_month = None
    peak_month_demand = None

    lowest_month = None
    lowest_month_demand = None

    # --------------------------------------------------------
    # Quarterly
    # --------------------------------------------------------

    quarterly_data = []

    peak_quarter = None

    # --------------------------------------------------------
    # Seasonal indices
    # --------------------------------------------------------

    seasonal_indices = []

    overall_average = None

    # --------------------------------------------------------
    # STL decomposition
    # --------------------------------------------------------

    decomposition_data = []

    seasonal_strength = None
    seasonal_strength_percentage = None

    trend_strength = None
    trend_strength_percentage = None

    decomposition_method = None

    # --------------------------------------------------------
    # Seasonal cycles
    # --------------------------------------------------------

    seasonal_cycles = []

    strongest_cycle = None

    # --------------------------------------------------------
    # Festival analysis
    # --------------------------------------------------------

    festival_available = False

    festival_events = []

    festival_message = (
        "No festival or holiday column was detected."
    )

    # --------------------------------------------------------
    # Peak periods
    # --------------------------------------------------------

    monthly_peaks = []

    weekly_peaks = []

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    seasonal_recommendations = []

    # --------------------------------------------------------
    # Complete ML result
    # --------------------------------------------------------

    seasonal_ml = {

        "success": False,

        "message": (
            "Seasonal analysis has not been executed."
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
    # PROCESS DATASET
    # ========================================================

    if selected_dataset is not None:

        df = selected_dataset.copy()

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if not df.empty:

            total_records = int(
                len(df)
            )

            # =================================================
            # RUN SEASONAL ML PIPELINE FROM CACHE
            # =================================================

            try:

                seasonal_ml = get_cached_module_result(
                    "seasonal",
                    source_context,
                    selected_dataset,
                    lambda: run_seasonal_ml(selected_dataset)
                )

            except Exception as error:

                seasonal_ml = {

                    "success": False,

                    "message": (
                        f"Seasonal ML pipeline failed: {error}"
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

            # =================================================
            # COLUMN INFORMATION
            # =================================================

            columns = seasonal_ml.get(
                "columns",
                {}
            )

            date_column = columns.get(
                "date"
            )

            demand_column = columns.get(
                "demand"
            )

            product_column = columns.get(
                "product"
            )

            # If demand is not directly available,
            # use sales as the demand source.

            if demand_column is None:

                demand_column = columns.get(
                    "sales"
                )

            # =================================================
            # BASIC DATA INFORMATION
            # =================================================

            if date_column:

                date_values = pd.to_datetime(
                    df[date_column],
                    errors="coerce"
                ).dropna()

                if not date_values.empty:

                    start_date = (
                        date_values.min()
                        .strftime(
                            "%d %b %Y"
                        )
                    )

                    end_date = (
                        date_values.max()
                        .strftime(
                            "%d %b %Y"
                        )
                    )

                    data_period = (
                        f"{start_date} → {end_date}"
                    )

            # ------------------------------------------------
            # Total demand
            # ------------------------------------------------

            actual_demand_column = (
                demand_column
                or columns.get("sales")
            )

            if actual_demand_column:

                demand_values = pd.to_numeric(
                    df[actual_demand_column],
                    errors="coerce"
                ).dropna()

                if not demand_values.empty:

                    total_demand = round(
                        float(
                            demand_values.sum()
                        ),
                        2
                    )

            # =================================================
            # CHECK PIPELINE SUCCESS
            # =================================================

            if seasonal_ml.get(
                "success"
            ):

                seasonal_available = True

                seasonal_message = (
                    seasonal_ml.get(
                        "message",
                        "Seasonal analysis completed."
                    )
                )

            else:

                seasonal_message = (
                    seasonal_ml.get(
                        "message",
                        "Seasonal analysis could not be completed."
                    )
                )

            # =================================================
            # WEEKLY ANALYSIS
            # =================================================

            weekly_result = seasonal_ml.get(
                "weekly",
                {}
            )

            if weekly_result.get(
                "available"
            ):

                weekly_data = (
                    weekly_result.get(
                        "weekly_data",
                        []
                    )
                )

                peak_day = (
                    weekly_result.get(
                        "peak_day"
                    )
                )

                peak_day_demand = (
                    weekly_result.get(
                        "peak_demand"
                    )
                )

                lowest_day = (
                    weekly_result.get(
                        "lowest_day"
                    )
                )

                lowest_day_demand = (
                    weekly_result.get(
                        "lowest_demand"
                    )
                )

            # =================================================
            # MONTHLY ANALYSIS
            # =================================================

            monthly_result = seasonal_ml.get(
                "monthly",
                {}
            )

            if monthly_result.get(
                "available"
            ):

                monthly_data = (
                    monthly_result.get(
                        "monthly_data",
                        []
                    )
                )

                peak_month = (
                    monthly_result.get(
                        "peak_month"
                    )
                )

                peak_month_demand = (
                    monthly_result.get(
                        "peak_demand"
                    )
                )

                lowest_month = (
                    monthly_result.get(
                        "lowest_month"
                    )
                )

                lowest_month_demand = (
                    monthly_result.get(
                        "lowest_demand"
                    )
                )

            # =================================================
            # QUARTERLY ANALYSIS
            # =================================================

            quarterly_result = seasonal_ml.get(
                "quarterly",
                {}
            )

            if quarterly_result.get(
                "available"
            ):

                quarterly_data = (
                    quarterly_result.get(
                        "quarterly_data",
                        []
                    )
                )

                peak_quarter = (
                    quarterly_result.get(
                        "peak_quarter"
                    )
                )

            # =================================================
            # SEASONAL INDICES
            # =================================================

            index_result = seasonal_ml.get(
                "seasonal_indices",
                {}
            )

            if index_result.get(
                "available"
            ):

                seasonal_indices = (
                    index_result.get(
                        "indices",
                        []
                    )
                )

                overall_average = (
                    index_result.get(
                        "overall_average"
                    )
                )

            # =================================================
            # STL DECOMPOSITION
            # =================================================

            decomposition_result = seasonal_ml.get(
                "decomposition",
                {}
            )

            if decomposition_result.get(
                "available"
            ):

                decomposition_data = (
                    decomposition_result.get(
                        "decomposition",
                        []
                    )
                )

                decomposition_method = (
                    decomposition_result.get(
                        "method"
                    )
                )

                seasonal_strength = (
                    decomposition_result.get(
                        "seasonal_strength"
                    )
                )

                seasonal_strength_percentage = (
                    decomposition_result.get(
                        "seasonal_strength_percentage"
                    )
                )

                trend_strength = (
                    decomposition_result.get(
                        "trend_strength"
                    )
                )

                trend_strength_percentage = (
                    decomposition_result.get(
                        "trend_strength_percentage"
                    )
                )

            # =================================================
            # SEASONAL CYCLES
            # =================================================

            cycle_result = seasonal_ml.get(
                "seasonal_cycles",
                {}
            )

            if cycle_result.get(
                "available"
            ):

                seasonal_cycles = (
                    cycle_result.get(
                        "cycles",
                        []
                    )
                )

                strongest_cycle = (
                    cycle_result.get(
                        "strongest_cycle"
                    )
                )

            # =================================================
            # FESTIVAL ANALYSIS
            # =================================================

            festival_result = seasonal_ml.get(
                "festival",
                {}
            )

            if festival_result.get(
                "available"
            ):

                festival_available = True

                festival_events = (
                    festival_result.get(
                        "events",
                        []
                    )
                )

                festival_message = (
                    "Festival and holiday effects detected."
                )

            else:

                festival_message = (
                    festival_result.get(
                        "message",
                        "No festival or holiday data available."
                    )
                )

            # =================================================
            # PEAK PERIODS
            # =================================================

            peak_result = seasonal_ml.get(
                "peak_periods",
                {}
            )

            monthly_peaks = (
                peak_result.get(
                    "monthly_peaks",
                    []
                )
            )

            weekly_peaks = (
                peak_result.get(
                    "weekly_peaks",
                    []
                )
            )

            # =================================================
            # RECOMMENDATIONS
            # =================================================

            seasonal_recommendations = (
                seasonal_ml.get(
                    "recommendations",
                    []
                )
            )

        else:

            seasonal_message = (
                "The selected dataset is empty."
            )

    # ========================================================
    # RENDER TEMPLATE
    # ========================================================

    return render_template(

        "seasonal.html",

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        seasonal_available=seasonal_available,

        seasonal_message=seasonal_message,

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        total_records=total_records,

        total_demand=total_demand,

        data_period=data_period,

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        date_column=date_column,

        demand_column=demand_column,

        product_column=product_column,

        # ----------------------------------------------------
        # Weekly
        # ----------------------------------------------------

        weekly_data=weekly_data,

        peak_day=peak_day,

        peak_day_demand=peak_day_demand,

        lowest_day=lowest_day,

        lowest_day_demand=lowest_day_demand,

        # ----------------------------------------------------
        # Monthly
        # ----------------------------------------------------

        monthly_data=monthly_data,

        peak_month=peak_month,

        peak_month_demand=peak_month_demand,

        lowest_month=lowest_month,

        lowest_month_demand=lowest_month_demand,

        # ----------------------------------------------------
        # Quarterly
        # ----------------------------------------------------

        quarterly_data=quarterly_data,

        peak_quarter=peak_quarter,

        # ----------------------------------------------------
        # Seasonal indices
        # ----------------------------------------------------

        seasonal_indices=seasonal_indices,

        overall_average=overall_average,

        # ----------------------------------------------------
        # STL decomposition
        # ----------------------------------------------------

        decomposition_data=decomposition_data,

        decomposition_method=decomposition_method,

        seasonal_strength=seasonal_strength,

        seasonal_strength_percentage=(
            seasonal_strength_percentage
        ),

        trend_strength=trend_strength,

        trend_strength_percentage=(
            trend_strength_percentage
        ),

        # ----------------------------------------------------
        # Seasonal cycles
        # ----------------------------------------------------

        seasonal_cycles=seasonal_cycles,

        strongest_cycle=strongest_cycle,

        # ----------------------------------------------------
        # Festival
        # ----------------------------------------------------

        festival_available=festival_available,

        festival_events=festival_events,

        festival_message=festival_message,

        # ----------------------------------------------------
        # Peak periods
        # ----------------------------------------------------

        monthly_peaks=monthly_peaks,

        weekly_peaks=weekly_peaks,

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        seasonal_recommendations=(
            seasonal_recommendations
        ),

        # ----------------------------------------------------
        # Complete ML object
        # ----------------------------------------------------

        seasonal_ml=seasonal_ml,

        # ----------------------------------------------------
        # Dataset source
        # ----------------------------------------------------

        **{
            key: value
            for key, value
            in source_context.items()
            if key != "selected_dataset"
        }
    )
# ============================================================
# SMART ALERT EMAIL SUMMARY
# ============================================================

@app.route("/alerts/email-summary", methods=["POST"])
@login_required
def email_alert_summary():

    recipient_email = request.form.get(
        "summary_email",
        ""
    ).strip().lower()

    if (
        "@" not in recipient_email
        or "." not in recipient_email.split("@")[-1]
    ):
        flash(
            "Please enter a valid email address.",
            "danger"
        )
        return redirect(url_for("alerts"))

    source_context = get_module_source_context("alerts")
    selected_dataset = source_context.get("selected_dataset")

    if selected_dataset is None or selected_dataset.empty:
        flash(
            "No cleaned dataset is available to create an alert summary.",
            "warning"
        )
        return redirect(url_for("alerts"))

    try:
        alert_result = run_alerts_ml(selected_dataset)
        alerts = alert_result.get("alerts", [])
        summary = alert_result.get("summary", {})

        total = summary.get("total", len(alerts))
        critical = summary.get("critical", 0)
        high = summary.get("high", 0)
        medium = summary.get("medium", 0)
        low = summary.get("low", 0)

        product_alerts = {}
        for alert in alerts:
            product = str(
                alert.get("product", "")
            ).strip()
            if not product or product.lower() in {"none", "nan"}:
                product = "Unspecified product"

            product_alerts.setdefault(product, []).append(alert)

        product_lines = []
        for product, product_items in sorted(
            product_alerts.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )[:5]:
            reasons = []
            actions = []
            severities = []

            for alert in product_items:
                severity = str(
                    alert.get("severity", "medium")
                ).upper()
                severities.append(severity)

                reason = str(
                    alert.get(
                        "why_triggered",
                        alert.get(
                            "reason",
                            alert.get("message", "")
                        )
                    )
                ).strip()
                if reason and reason not in reasons:
                    reasons.append(reason)

                action = str(
                    alert.get(
                        "recommendation",
                        alert.get("recommended_action", "")
                    )
                ).strip()
                if action and action not in actions:
                    actions.append(action)

            product_lines.append(
                f"- Product: {product}\n"
                f"  Occurrence: {len(product_items)} active alert(s)\n"
                f"  Priority: {', '.join(sorted(set(severities)))}\n"
                f"  Why: {' '.join(reasons[:2]) or 'The product has an active demand or inventory condition.'}\n"
                f"  Suggested action: {actions[0] if actions else 'Review this product in the Smart Alerts page.'}"
            )

        product_summary = "\n\n".join(product_lines)
        if not product_summary:
            product_summary = "No active product alert details were detected."

        message = (
            "Retail Demand Forecasting System\n\n"
            "Smart Alerts Summary\n"
            "=====================\n"
            f"Total active alerts: {total}\n"
            f"Critical: {critical}\n"
            f"High: {high}\n"
            f"Medium: {medium}\n"
            f"Low: {low}\n\n"
            "Products appearing most often and why:\n"
            f"{product_summary}\n\n"
            "Please open the Smart Alerts page for the complete analysis."
        )

        email_result = send_email_notification(
            recipient_email=recipient_email,
            subject="Smart Alerts Summary - Retail Demand Forecasting System",
            message=message
        )

        if email_result.get("success"):
            flash(
                f"The short alert summary was sent to {recipient_email}.",
                "success"
            )
        else:
            flash(
                "The alert summary could not be sent: "
                f"{email_result.get('error', 'Email delivery failed.')}",
                "danger"
            )

    except Exception:
        flash(
            "The alert summary could not be created. Please try again.",
            "danger"
        )

    return redirect(url_for("alerts"))


# ============================================================
# SMART ALERT NOTIFICATION SUBSCRIPTION
# ============================================================

@app.route("/alerts/subscribe", methods=["POST"])
@login_required
def subscribe_alert_notifications():

    # ========================================================
    # GET FORM VALUES
    # ========================================================

    email = request.form.get(
        "email",
        ""
    ).strip().lower()

    email_enabled = (
        request.form.get("email_enabled") == "on"
    )

    critical_enabled = (
        request.form.get("critical_enabled") == "on"
    )

    high_enabled = (
        request.form.get("high_enabled") == "on"
    )

    medium_enabled = (
        request.form.get("medium_enabled") == "on"
    )

    low_enabled = (
        request.form.get("low_enabled") == "on"
    )


    # ========================================================
    # VALIDATE EMAIL
    # ========================================================

    if not email:

        flash(
            "Please provide an email address.",
            "danger"
        )

        return redirect(
            url_for("alerts")
        )


    if (
        "@" not in email
        or "." not in email.split("@")[-1]
    ):

        flash(
            "Please enter a valid email address.",
            "danger"
        )

        return redirect(
            url_for("alerts")
        )


    # ========================================================
    # EMAIL MUST BE ENABLED
    # ========================================================

    if not email_enabled:

        flash(
            "Please enable Email Notifications.",
            "danger"
        )

        return redirect(
            url_for("alerts")
        )


    # ========================================================
    # AT LEAST ONE PRIORITY REQUIRED
    # ========================================================

    if not any([
        critical_enabled,
        high_enabled,
        medium_enabled,
        low_enabled
    ]):

        flash(
            "Please select at least one alert priority.",
            "danger"
        )

        return redirect(
            url_for("alerts")
        )


    # ========================================================
    # FIND EXISTING SUBSCRIPTION
    # ========================================================

    subscription = (
        NotificationSubscription.query
        .filter_by(
            user_id=current_user.id
        )
        .first()
    )


    # ========================================================
    # CREATE IF REQUIRED
    # ========================================================

    if subscription is None:

        subscription = NotificationSubscription(
            user_id=current_user.id
        )

        db.session.add(
            subscription
        )


    # ========================================================
    # SAVE SETTINGS
    # ========================================================

    subscription.email = email

    subscription.email_enabled = (
        email_enabled
    )

    subscription.critical_enabled = (
        critical_enabled
    )

    subscription.high_enabled = (
        high_enabled
    )

    subscription.medium_enabled = (
        medium_enabled
    )

    subscription.low_enabled = (
        low_enabled
    )


    # ========================================================
    # COMMIT
    # ========================================================

    try:

        db.session.commit()

        flash(
            "Email alert notification settings saved successfully.",
            "success"
        )

    except Exception as e:

        db.session.rollback()

        print(
            "[SUBSCRIPTION ERROR]",
            repr(e)
        )

        flash(
            "Could not save email notification settings.",
            "danger"
        )


    return redirect(
        url_for("alerts")
    )


# ============================================================
# SMART ALERTS PAGE
# ============================================================

@app.route("/alerts")
@login_required
def alerts():

    # ========================================================
    # DEFAULT NOTIFICATION RESULT
    # ========================================================

    notification_result = {
        "enabled": False,
        "alerts_processed": 0,
        "notifications_sent": 0,
        "notifications_skipped": 0,
        "notifications_failed": 0,
        "details": []
    }


    # ========================================================
    # GET SHARED CLEANED DATASET
    # ========================================================

    source_context = get_module_source_context(
        "alerts"
    )

    selected_dataset = source_context.get(
        "selected_dataset"
    )


    # ========================================================
    # DEFAULT ALERT RESULT
    # ========================================================

    alert_result = {

        "success": False,

        "message": (
            "No cleaned shared dataset is available "
            "for Smart Alerts."
        ),

        "alerts": [],

        "total_alerts": 0,

        "critical_alerts": 0,

        "high_alerts": 0,

        "medium_alerts": 0,

        "low_alerts": 0,

        "insights": [],

        "metrics": {},

        "summary": {
            "total": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        },

        "ml": {

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
                "Not Available"
        },

        "analysis": {},

        "diagnostics": {}
    }


    # ========================================================
    # DATASET AVAILABLE
    # ========================================================

    if (
        selected_dataset is not None
        and not selected_dataset.empty
    ):

        try:

            # =================================================
            # CONSOLE DEBUG INFORMATION
            # =================================================

            print()
            print("=" * 80)
            print("SMART ALERTS")
            print("USING SHARED CLEANED DATASET")
            print("=" * 80)

            print(
                "Dataset rows:",
                len(selected_dataset)
            )

            print(
                "Dataset columns:",
                list(selected_dataset.columns)
            )

            print("=" * 80)


            # =================================================
            # RUN SMART ALERTS ML ENGINE WITH ROUTE CACHE
            # =================================================

            result = get_cached_module_result(
                "alerts",
                source_context,
                selected_dataset,
                lambda: run_alerts_ml(selected_dataset)
            )


            # =================================================
            # VALIDATE RESULT
            # =================================================

            if result is None:

                raise RuntimeError(
                    "Smart Alerts ML engine returned no result."
                )


            if not isinstance(
                result,
                dict
            ):

                raise RuntimeError(
                    "Smart Alerts ML engine returned "
                    "an invalid result."
                )


            alert_result = result


            # =================================================
            # GET ALERT LIST
            # =================================================

            alerts_list = alert_result.get(
                "alerts",
                []
            )


            if alerts_list is None:

                alerts_list = []


            if not isinstance(
                alerts_list,
                list
            ):

                alerts_list = list(
                    alerts_list
                )


            alert_result["alerts"] = (
                alerts_list
            )


            # =================================================
            # GET SUMMARY
            #
            # smart_alerts.py may return:
            #
            # summary:
            # {
            #     total,
            #     critical,
            #     high,
            #     medium,
            #     low
            # }
            #
            # OR direct counts.
            #
            # This route supports both.
            # =================================================

            summary = alert_result.get(
                "summary",
                {}
            )


            if not isinstance(
                summary,
                dict
            ):

                summary = {}


            # =================================================
            # TOTAL
            # =================================================

            total_alerts = alert_result.get(
                "total_alerts"
            )


            if total_alerts is None:

                total_alerts = summary.get(
                    "total",
                    len(alerts_list)
                )


            # =================================================
            # CRITICAL
            # =================================================

            critical_alerts = alert_result.get(
                "critical_alerts"
            )


            if critical_alerts is None:

                critical_alerts = summary.get(
                    "critical",
                    0
                )


            # =================================================
            # HIGH
            # =================================================

            high_alerts = alert_result.get(
                "high_alerts"
            )


            if high_alerts is None:

                high_alerts = summary.get(
                    "high",
                    0
                )


            # =================================================
            # MEDIUM
            # =================================================

            medium_alerts = alert_result.get(
                "medium_alerts"
            )


            if medium_alerts is None:

                medium_alerts = summary.get(
                    "medium",
                    0
                )


            # =================================================
            # LOW
            # =================================================

            low_alerts = alert_result.get(
                "low_alerts"
            )


            if low_alerts is None:

                low_alerts = summary.get(
                    "low",
                    0
                )


            # =================================================
            # CONVERT COUNTS TO INTEGER
            # =================================================

            try:

                total_alerts = int(
                    total_alerts
                )

            except Exception:

                total_alerts = len(
                    alerts_list
                )


            try:

                critical_alerts = int(
                    critical_alerts
                )

            except Exception:

                critical_alerts = 0


            try:

                high_alerts = int(
                    high_alerts
                )

            except Exception:

                high_alerts = 0


            try:

                medium_alerts = int(
                    medium_alerts
                )

            except Exception:

                medium_alerts = 0


            try:

                low_alerts = int(
                    low_alerts
                )

            except Exception:

                low_alerts = 0


            # =================================================
            # IMPORTANT:
            # SAVE COUNTS BACK INTO alert_result
            # =================================================

            alert_result["total_alerts"] = (
                total_alerts
            )

            alert_result["critical_alerts"] = (
                critical_alerts
            )

            alert_result["high_alerts"] = (
                high_alerts
            )

            alert_result["medium_alerts"] = (
                medium_alerts
            )

            alert_result["low_alerts"] = (
                low_alerts
            )


            # =================================================
            # REBUILD SUMMARY
            # =================================================

            alert_result["summary"] = {

                "total":
                    total_alerts,

                "critical":
                    critical_alerts,

                "high":
                    high_alerts,

                "medium":
                    medium_alerts,

                "low":
                    low_alerts,

                "info":
                    summary.get(
                        "info",
                        0
                    )
            }


            # =================================================
            # ML INFORMATION
            # =================================================

            ml_info = alert_result.get(
                "ml",
                {}
            )


            if not isinstance(
                ml_info,
                dict
            ):

                ml_info = {}


            # =================================================
            # ML METHOD
            # =================================================

            if not ml_info.get(
                "method"
            ):

                ml_info["method"] = (
                    alert_result.get(
                        "method",
                        "Isolation Forest"
                    )
                )


            # =================================================
            # ANOMALIES DETECTED
            # =================================================

            if (
                ml_info.get(
                    "anomalies_detected"
                ) is None
            ):

                ml_info[
                    "anomalies_detected"
                ] = alert_result.get(
                    "anomalies_detected",
                    0
                )


            # =================================================
            # ANOMALY RATE
            # =================================================

            if (
                ml_info.get(
                    "anomaly_rate"
                ) is None
            ):

                ml_info[
                    "anomaly_rate"
                ] = alert_result.get(
                    "anomaly_rate",
                    0
                )


            # =================================================
            # RECORDS ANALYZED
            # =================================================

            if (
                ml_info.get(
                    "records_analyzed"
                ) is None
            ):

                ml_info[
                    "records_analyzed"
                ] = len(
                    selected_dataset
                )


            # =================================================
            # FEATURES USED
            # =================================================

            if (
                ml_info.get(
                    "features_used"
                ) is None
            ):

                ml_info[
                    "features_used"
                ] = []


            # =================================================
            # STATUS
            # =================================================

            if not ml_info.get(
                "status"
            ):

                ml_info[
                    "status"
                ] = "Completed"


            alert_result["ml"] = (
                ml_info
            )


            # =================================================
            # INSIGHTS
            # =================================================

            insights = alert_result.get(
                "insights",
                []
            )


            if insights is None:

                insights = []


            if not isinstance(
                insights,
                list
            ):

                insights = [
                    str(insights)
                ]


            alert_result["insights"] = (
                insights
            )


            # =================================================
            # ANALYSIS
            # =================================================

            analysis = alert_result.get(
                "analysis",
                {}
            )


            if analysis is None:

                analysis = {}


            alert_result["analysis"] = (
                analysis
            )


            # =================================================
            # DIAGNOSTICS
            # =================================================

            diagnostics = alert_result.get(
                "diagnostics",
                {}
            )


            if diagnostics is None:

                diagnostics = {}


            alert_result[
                "diagnostics"
            ] = diagnostics


            # =================================================
            # IMPORTANT SAFETY CHECK
            #
            # If actual alerts exist but the returned count
            # is zero, calculate the count from the alert list.
            # =================================================

            if (
                len(alerts_list) > 0
                and total_alerts == 0
            ):

                print(
                    "[SMART ALERTS WARNING] "
                    "Alerts exist but total count was zero."
                )

                try:

                    recalculated_summary = (
                        calculate_alert_summary(
                            alerts_list
                        )
                    )

                except Exception:

                    recalculated_summary = {

                        "total":
                            len(alerts_list),

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


                    # -----------------------------------------
                    # UPDATE COUNTS
                    # -----------------------------------------

                total_alerts = int(
                    recalculated_summary.get(
                        "total",
                        len(alerts_list)
                    )
                )

                critical_alerts = int(
                    recalculated_summary.get(
                        "critical",
                        0
                    )
                )

                high_alerts = int(
                    recalculated_summary.get(
                        "high",
                        0
                    )
                )

                medium_alerts = int(
                    recalculated_summary.get(
                        "medium",
                        0
                    )
                )

                low_alerts = int(
                    recalculated_summary.get(
                        "low",
                        0
                    )


                )

                alert_result[
                    "total_alerts"
                ] = total_alerts

                alert_result[
                    "critical_alerts"
                ] = critical_alerts

                alert_result[
                    "high_alerts"
                ] = high_alerts

                alert_result[
                    "medium_alerts"
                ] = medium_alerts

                alert_result[
                    "low_alerts"
                ] = low_alerts

                alert_result[
                    "summary"
                ] = recalculated_summary


            # =================================================
            # PRINT FINAL RESULT
            # =================================================

            print()
            print("=" * 80)
            print("SMART ALERTS ML RESULT")
            print("=" * 80)

            print(
                "Success:",
                alert_result.get(
                    "success",
                    False
                )
            )

            print(
                "Message:",
                alert_result.get(
                    "message",
                    ""
                )
            )

            print(
                "Total Alerts:",
                alert_result.get(
                    "total_alerts",
                    0
                )
            )

            print(
                "Critical:",
                alert_result.get(
                    "critical_alerts",
                    0
                )
            )

            print(
                "High:",
                alert_result.get(
                    "high_alerts",
                    0
                )
            )

            print(
                "Medium:",
                alert_result.get(
                    "medium_alerts",
                    0
                )
            )

            print(
                "Low:",
                alert_result.get(
                    "low_alerts",
                    0
                )
            )

            print(
                "Alert List Length:",
                len(
                    alerts_list
                )
            )

            print(
                "ML Information:",
                alert_result.get(
                    "ml",
                    {}
                )
            )

            print(
                "Analysis:",
                alert_result.get(
                    "analysis",
                    {}
                )
            )

            print(
                "Diagnostics:",
                alert_result.get(
                    "diagnostics",
                    {}
                )
            )

            print("=" * 80)

            # Email delivery is intentionally not performed while rendering
            # the alerts page. SMTP must not block the user's web request.
            notification_result = {
                "enabled": False,
                "alerts_processed": 0,
                "notifications_sent": 0,
                "notifications_skipped": 0,
                "notifications_failed": 0,
                "details": []
            }
        # =====================================================
        # SMART ALERT ERROR
        # =====================================================

        except Exception as e:

            print()
            print("=" * 80)
            print("[SMART ALERTS ERROR]")
            print("=" * 80)

            print(
                repr(e)
            )

            print("=" * 80)


            alert_result = {

                "success": False,

                "message": (
                    "Smart Alerts analysis failed: "
                    f"{str(e)}"
                ),

                "alerts": [],

                "total_alerts": 0,

                "critical_alerts": 0,

                "high_alerts": 0,

                "medium_alerts": 0,

                "low_alerts": 0,

                "insights": [

                    (
                        "The Smart Alerts engine could not "
                        "complete the analysis. Check the "
                        "Flask terminal for the exact error."
                    )

                ],

                "metrics": {},

                "summary": {

                    "total": 0,

                    "critical": 0,

                    "high": 0,

                    "medium": 0,

                    "low": 0,

                    "info": 0
                },

                "ml": {

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
                        "Error",

                    "error":
                        str(e)
                },

                "analysis": {},

                "diagnostics": {

                    "error":
                        str(e)
                }
            }


            notification_result = {

                "enabled": False,

                "alerts_processed": 0,

                "notifications_sent": 0,

                "notifications_skipped": 0,

                "notifications_failed": 0,

                "details": []
            }


    # ========================================================
    # EMPTY DATASET
    # ========================================================

    elif (
        selected_dataset is not None
        and selected_dataset.empty
    ):

        alert_result["success"] = False

        alert_result["message"] = (
            "The cleaned shared dataset is empty. "
            "Please process a dataset in Data Management "
            "before running Smart Alerts."
        )


    # ========================================================
    # NO SHARED DATASET
    # ========================================================

    else:

        alert_result["success"] = False

        alert_result["message"] = (
            "No cleaned shared dataset is available. "
            "Please upload and process a dataset from "
            "Data Management."
        )


    # ========================================================
    # GET EMAIL SUBSCRIPTION
    #
    # This ensures the saved email settings are available
    # to alerts.html.
    # ========================================================

    subscription = (
        NotificationSubscription.query
        .filter_by(
            user_id=current_user.id
        )
        .first()
    )


    # ========================================================
    # GET NOTIFICATION HISTORY
    # ========================================================

    notification_history = (
        NotificationLog.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            NotificationLog.sent_at.desc()
        )
        .limit(20)
        .all()
    )


    # ========================================================
    # PREPARE TEMPLATE VALUES
    # ========================================================

    template_data = {

        # ----------------------------------------------------
        # COMPLETE RESULT
        # ----------------------------------------------------

        "alert_result":
            alert_result,

        "smart_alerts":
            alert_result,


        # ----------------------------------------------------
        # ALERT LIST
        # ----------------------------------------------------

        "alerts":
            alert_result.get(
                "alerts",
                []
            ),


        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        "total_alerts":
            alert_result.get(
                "total_alerts",
                0
            ),

        "critical_alerts":
            alert_result.get(
                "critical_alerts",
                0
            ),

        "high_alerts":
            alert_result.get(
                "high_alerts",
                0
            ),

        "medium_alerts":
            alert_result.get(
                "medium_alerts",
                0
            ),

        "low_alerts":
            alert_result.get(
                "low_alerts",
                0
            ),


        # ----------------------------------------------------
        # INSIGHTS
        # ----------------------------------------------------

        "alert_insights":
            alert_result.get(
                "insights",
                []
            ),


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        "alert_metrics":
            alert_result.get(
                "metrics",
                {}
            ),


        # ----------------------------------------------------
        # ML INFORMATION
        # ----------------------------------------------------

        "ml_info":
            alert_result.get(
                "ml",
                {}
            ),


        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        "alert_analysis":
            alert_result.get(
                "analysis",
                {}
            ),


        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        "alert_diagnostics":
            alert_result.get(
                "diagnostics",
                {}
            ),


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        "alerts_success":
            alert_result.get(
                "success",
                False
            ),

        "alerts_message":
            alert_result.get(
                "message",
                ""
            ),


        # ----------------------------------------------------
        # EMAIL SUBSCRIPTION
        # ----------------------------------------------------

        "subscription":
            subscription,


        # ----------------------------------------------------
        # EMAIL NOTIFICATION RESULT
        # ----------------------------------------------------

        "notification_result":
            notification_result,


        # ----------------------------------------------------
        # EMAIL HISTORY
        # ----------------------------------------------------

        "notification_history":
            notification_history

    }


    # ========================================================
    # PRESERVE DATASET SOURCE CONTEXT
    # ========================================================

    for key, value in source_context.items():

        if key != "selected_dataset":

            template_data[key] = value


    # ========================================================
    # RENDER SMART ALERTS
    # ========================================================

    return render_template(
        "alerts.html",
        **template_data
    )
# ============================================================
# REPORTS DASHBOARD DATA
# ============================================================

def load_reports_dashboard():

    import os
    import json
    import pandas as pd
    from datetime import datetime

    dashboard = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "kpis": {
            "records": 0,
            "columns": 0,
            "total_demand": 0,
            "total_inventory": 0,
            "products": 0,
            "forecast_horizon": 0,
            "forecast_best_model": "N/A",
            "forecast_best_mae": 0,
            "forecast_best_rmse": 0,
            "inventory_rows": 0,
            "low_stock_rows": 0,
            "overstock_rows": 0
        },

        "charts": {
            "demand_trend": [],
            "top_products": [],
            "inventory_status": [],
            "weekday_demand": [],
            "monthly_demand": [],
            "forecast_models": [],
            "alert_severity": [],
            "inventory_vs_demand": []
        },

        "recommendations": []
    }

    try:

        # ========================================================
        # LOAD SHARED DATASET
        # ========================================================

        if not os.path.exists(SHARED_DATASET_PATH):
            return dashboard

        df = read_csv_safely(SHARED_DATASET_PATH)

        if df is None or df.empty:
            return dashboard

        # ========================================================
        # NORMALIZE COLUMN NAMES
        # ========================================================

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

        print("REPORTS DATASET COLUMNS:")
        print(df.columns.tolist())

        # ========================================================
        # BASIC DATA INFORMATION
        # ========================================================

        dashboard["kpis"]["records"] = int(len(df))
        dashboard["kpis"]["columns"] = int(len(df.columns))

        # ========================================================
        # DETECT DEMAND COLUMN
        # ========================================================

        demand_column = None

        demand_candidates = [
            "daily_demand",
            "demand",
            "sales",
            "quantity_sold",
            "units_sold",
            "sales_quantity",
            "total_sales",
            "sales_qty",
            "quantity",
            "qty_sold",
            "demand_quantity",
            "units"
        ]

        for column in demand_candidates:
            if column in df.columns:
                demand_column = column
                break

        print("DEMAND COLUMN DETECTED:", demand_column)

        # ========================================================
        # DETECT INVENTORY COLUMN
        # ========================================================

        inventory_column = None

        inventory_candidates = [
            "inventory",
            "inventory_level",
            "stock",
            "stock_level",
            "current_inventory",
            "available_inventory",
            "quantity_in_stock",
            "stock_quantity"
        ]

        for column in inventory_candidates:
            if column in df.columns:
                inventory_column = column
                break

        print("INVENTORY COLUMN DETECTED:", inventory_column)

        # ========================================================
        # DETECT PRODUCT COLUMN
        # ========================================================

        product_column = None

        product_candidates = [
            "product_code",
            "product_id",
            "product",
            "product_name",
            "sku",
            "item_id",
            "item"
        ]

        for column in product_candidates:
            if column in df.columns:
                product_column = column
                break

        print("PRODUCT COLUMN DETECTED:", product_column)

        # ========================================================
        # DETECT DATE COLUMN
        # ========================================================

        date_column = None

        date_candidates = [
            "date",
            "order_date",
            "sales_date",
            "transaction_date",
            "datetime",
            "timestamp"
        ]

        for column in date_candidates:
            if column in df.columns:
                date_column = column
                break

        print("DATE COLUMN DETECTED:", date_column)

        # ========================================================
        # CONVERT DEMAND TO NUMBER
        # ========================================================

        if demand_column:

            df[demand_column] = pd.to_numeric(
                df[demand_column],
                errors="coerce"
            ).fillna(0)

            dashboard["kpis"]["total_demand"] = round(
                float(df[demand_column].sum()),
                2
            )

        # ========================================================
        # CONVERT INVENTORY TO NUMBER
        # ========================================================

        if inventory_column:

            df[inventory_column] = pd.to_numeric(
                df[inventory_column],
                errors="coerce"
            ).fillna(0)

            dashboard["kpis"]["total_inventory"] = round(
                float(df[inventory_column].sum()),
                2
            )

        # ========================================================
        # PRODUCT COUNT
        # ========================================================

        if product_column:

            dashboard["kpis"]["products"] = int(
                df[product_column].nunique()
            )

        # ========================================================
        # CONVERT DATE
        # ========================================================

        if date_column:

            df[date_column] = pd.to_datetime(
                df[date_column],
                errors="coerce"
            )

            df = df.dropna(
                subset=[date_column]
            )

        # ========================================================
        # DEMAND TREND
        # ========================================================

        if date_column and demand_column:

            trend = (
                df.groupby(
                    df[date_column].dt.date
                )[demand_column]
                .sum()
                .reset_index()
            )

            trend = trend.sort_values(
                by=date_column
            )

            for _, row in trend.iterrows():

                dashboard["charts"]["demand_trend"].append({
                    "date": str(row[date_column]),
                    "demand": round(
                        float(row[demand_column]),
                        2
                    )
                })

        # ========================================================
        # TOP PRODUCTS BY DEMAND
        # ========================================================

        if product_column and demand_column:

            top_products = (
                df.groupby(product_column)[demand_column]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            for product, demand in top_products.items():

                dashboard["charts"]["top_products"].append({
                    "product": str(product),
                    "demand": round(
                        float(demand),
                        2
                    )
                })

        print(
            "TOP PRODUCTS:",
            dashboard["charts"]["top_products"]
        )

        # ========================================================
        # DEMAND BY WEEKDAY
        # ========================================================

        if date_column and demand_column:

            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]

            weekday_data = (
                df.groupby(
                    df[date_column].dt.day_name()
                )[demand_column]
                .sum()
            )

            for day in weekday_order:

                dashboard["charts"]["weekday_demand"].append({
                    "day": day,
                    "demand": round(
                        float(
                            weekday_data.get(day, 0)
                        ),
                        2
                    )
                })

        # ========================================================
        # MONTHLY DEMAND
        # ========================================================

        if date_column and demand_column:

            monthly = (
                df.groupby(
                    df[date_column].dt.to_period("M")
                )[demand_column]
                .sum()
                .reset_index()
            )

            for _, row in monthly.iterrows():

                dashboard["charts"]["monthly_demand"].append({
                    "month": str(row[date_column]),
                    "demand": round(
                        float(row[demand_column]),
                        2
                    )
                })

        # ========================================================
        # INVENTORY VS DEMAND
        # ========================================================

        if inventory_column and demand_column:

            comparison = df[
                [inventory_column, demand_column]
            ].dropna()

            comparison = comparison.head(100)

            for _, row in comparison.iterrows():

                dashboard["charts"][
                    "inventory_vs_demand"
                ].append({
                    "inventory": round(
                        float(row[inventory_column]),
                        2
                    ),
                    "demand": round(
                        float(row[demand_column]),
                        2
                    )
                })

        # ========================================================
        # INVENTORY STATUS
        # ========================================================

        status_column = None

        for column in [
            "status",
            "stock_status",
            "inventory_status",
            "risk_level"
        ]:

            if column in df.columns:

                status_column = column
                break

        if status_column:

            status_counts = (
                df[status_column]
                .astype(str)
                .value_counts()
            )

            for status, count in status_counts.items():

                dashboard["charts"][
                    "inventory_status"
                ].append({
                    "status": str(status),
                    "count": int(count)
                })

        # ========================================================
        # REORDER POINT
        # ========================================================

        reorder_column = None

        for column in [
            "reorder_point",
            "reorder_level"
        ]:

            if column in df.columns:

                reorder_column = column
                break

        if inventory_column and reorder_column:

            reorder_values = pd.to_numeric(
                df[reorder_column],
                errors="coerce"
            )

            valid = (
                df[inventory_column].notna()
                & reorder_values.notna()
            )

            low_stock = (
                valid
                & (
                    df[inventory_column]
                    <= reorder_values
                )
            )

            dashboard["kpis"][
                "low_stock_rows"
            ] = int(
                low_stock.sum()
            )

        # ========================================================
        # INVENTORY OPTIMIZATION RESULTS
        # ========================================================

        inventory_path = os.path.join(
            REPORT_FOLDER,
            "inventory_optimization_results.csv"
        )

        if os.path.exists(inventory_path):

            inventory_df = read_csv_safely(
                inventory_path
            )

            if (
                inventory_df is not None
                and not inventory_df.empty
            ):

                inventory_df.columns = (
                    inventory_df.columns.astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(
                        " ",
                        "_",
                        regex=False
                    )
                    .str.replace(
                        "-",
                        "_",
                        regex=False
                    )
                )

                dashboard["kpis"][
                    "inventory_rows"
                ] = int(
                    len(inventory_df)
                )

                inventory_status_column = None

                for column in [
                    "status",
                    "stock_status",
                    "inventory_status",
                    "risk_level"
                ]:

                    if column in inventory_df.columns:

                        inventory_status_column = column
                        break

                if inventory_status_column:

                    status_values = (
                        inventory_df[
                            inventory_status_column
                        ]
                        .astype(str)
                        .str.lower()
                    )

                    dashboard["kpis"][
                        "low_stock_rows"
                    ] = int(
                        status_values.str.contains(
                            "low|shortage|stockout|reorder",
                            regex=True
                        ).sum()
                    )

                    dashboard["kpis"][
                        "overstock_rows"
                    ] = int(
                        status_values.str.contains(
                            "overstock|excess",
                            regex=True
                        ).sum()
                    )

                    status_counts = (
                        inventory_df[
                            inventory_status_column
                        ]
                        .astype(str)
                        .value_counts()
                    )

                    dashboard["charts"][
                        "inventory_status"
                    ] = [
                        {
                            "status": str(status),
                            "count": int(count)
                        }
                        for status, count
                        in status_counts.items()
                    ]

        # ========================================================
        # FORECAST REPORT
        # ========================================================

        try:

            if os.path.exists(
                FORECAST_REPORT_PATH
            ):

                with open(
                    FORECAST_REPORT_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    forecast = json.load(file)

                dashboard["kpis"][
                    "forecast_horizon"
                ] = (
                    forecast.get("forecast_horizon")
                    or forecast.get("horizon")
                    or 0
                )

                dashboard["kpis"][
                    "forecast_best_model"
                ] = (
                    forecast.get("best_model")
                    or forecast.get("selected_model")
                    or "N/A"
                )

                dashboard["kpis"][
                    "forecast_best_mae"
                ] = (
                    forecast.get("best_mae")
                    or forecast.get("mae")
                    or 0
                )

                dashboard["kpis"][
                    "forecast_best_rmse"
                ] = (
                    forecast.get("best_rmse")
                    or forecast.get("rmse")
                    or 0
                )

                models = forecast.get(
                    "models",
                    []
                )

                if isinstance(models, list):

                    for model in models:

                        if isinstance(model, dict):

                            dashboard["charts"][
                                "forecast_models"
                            ].append({
                                "model": str(
                                    model.get(
                                        "model",
                                        model.get(
                                            "name",
                                            "Model"
                                        )
                                    )
                                ),
                                "mae": float(
                                    model.get(
                                        "mae",
                                        0
                                    ) or 0
                                ),
                                "rmse": float(
                                    model.get(
                                        "rmse",
                                        0
                                    ) or 0
                                )
                            })

        except Exception as error:

            print(
                "Forecast report warning:",
                error
            )

        # ========================================================
        # SMART ALERT SEVERITY
        # ========================================================

        try:

            alert_function = globals().get(
                "calculate_alert_summary"
            )

            if callable(alert_function):

                result = alert_function()

                if isinstance(result, dict):

                    summary = result.get(
                        "summary",
                        result
                    )

                    if isinstance(summary, dict):

                        for level in [
                            "critical",
                            "high",
                            "medium",
                            "low"
                        ]:

                            count = summary.get(
                                level,
                                0
                            )

                            try:

                                count = int(
                                    count or 0
                                )

                            except Exception:

                                count = 0

                            dashboard["charts"][
                                "alert_severity"
                            ].append({
                                "severity": level.title(),
                                "count": count
                            })

        except Exception as error:

            print(
                "Alert severity warning:",
                error
            )

        print(
            "REPORT ALERT SEVERITY:",
            dashboard["charts"][
                "alert_severity"
            ]
        )

        # ========================================================
        # RECOMMENDATIONS
        # ========================================================

        if dashboard["kpis"][
            "low_stock_rows"
        ] > 0:

            dashboard["recommendations"].append(
                "Review low-stock products using the Inventory module."
            )

        if dashboard["kpis"][
            "overstock_rows"
        ] > 0:

            dashboard["recommendations"].append(
                "Review overstocked products to reduce excess inventory."
            )

        if dashboard["kpis"][
            "forecast_best_model"
        ] != "N/A":

            dashboard["recommendations"].append(
                "Use the selected forecasting model for demand planning."
            )

        if dashboard["charts"][
            "top_products"
        ]:

            top_product = dashboard[
                "charts"
            ][
                "top_products"
            ][0]["product"]

            dashboard["recommendations"].append(
                f"Monitor demand for top product {top_product}."
            )

        if not dashboard["recommendations"]:

            dashboard["recommendations"].append(
                "Continue monitoring demand, inventory and forecast results."
            )

    except Exception as error:

        print(
            "Reports dashboard error:",
            error
        )

    return dashboard
#download routes
@app.route("/reports/download-bi-report")
@login_required
def download_bi_report():

    """
    Automatically generate a Business Intelligence Excel report
    from the same cleaned shared dataset used by the Reports module.
    """

    if not os.path.exists(SHARED_DATASET_PATH):
        return (
            "No cleaned shared dataset is available. "
            "Please upload and process a dataset first.",
            404
        )

    try:

        # ---------------------------------------------------------
        # LOAD CLEANED SHARED DATASET
        # ---------------------------------------------------------

        df = pd.read_csv(
            SHARED_DATASET_PATH,
            low_memory=False
        )


        # ---------------------------------------------------------
        # CREATE EXCEL FILE IN MEMORY
        # ---------------------------------------------------------

        output = io.BytesIO()


        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            # -----------------------------------------------------
            # 1. DATASET SUMMARY
            # -----------------------------------------------------

            summary_data = {

                "Metric": [

                    "Total Records",

                    "Total Columns",

                    "Total Demand",

                    "Total Inventory",

                    "Unique Products"

                ],

                "Value": [

                    len(df),

                    len(df.columns),

                    (
                        pd.to_numeric(
                            df.get("demand", pd.Series(dtype=float)),
                            errors="coerce"
                        ).sum()
                    ),

                    (
                        pd.to_numeric(
                            df.get("inventory", pd.Series(dtype=float)),
                            errors="coerce"
                        ).sum()
                    ),

                    (
                        df["product"].nunique()
                        if "product" in df.columns
                        else 0
                    )

                ]

            }


            summary_df = pd.DataFrame(
                summary_data
            )


            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )


            # -----------------------------------------------------
            # 2. ORIGINAL CLEANED DATA
            # -----------------------------------------------------

            df.to_excel(
                writer,
                sheet_name="Cleaned Data",
                index=False
            )


            # -----------------------------------------------------
            # 3. TOP PRODUCTS
            # -----------------------------------------------------

            if (
                "product" in df.columns
                and "demand" in df.columns
            ):

                product_df = df.copy()

                product_df["demand"] = pd.to_numeric(
                    product_df["demand"],
                    errors="coerce"
                )

                top_products = (

                    product_df
                    .groupby("product", dropna=False)["demand"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(20)
                    .reset_index()

                )

                top_products.to_excel(
                    writer,
                    sheet_name="Top Products",
                    index=False
                )


            # -----------------------------------------------------
            # 4. MONTHLY DEMAND
            # -----------------------------------------------------

            if (
                "date" in df.columns
                and "demand" in df.columns
            ):

                monthly_df = df.copy()

                monthly_df["date"] = pd.to_datetime(
                    monthly_df["date"],
                    errors="coerce"
                )

                monthly_df["demand"] = pd.to_numeric(
                    monthly_df["demand"],
                    errors="coerce"
                )

                monthly_df = monthly_df.dropna(
                    subset=["date"]
                )

                monthly_demand = (

                    monthly_df
                    .groupby(
                        monthly_df["date"].dt.to_period("M")
                    )["demand"]
                    .sum()
                    .reset_index()

                )

                monthly_demand["date"] = (
                    monthly_demand["date"]
                    .astype(str)
                )

                monthly_demand.to_excel(
                    writer,
                    sheet_name="Monthly Demand",
                    index=False
                )


            # -----------------------------------------------------
            # 5. WEEKDAY DEMAND
            # -----------------------------------------------------

            if (
                "date" in df.columns
                and "demand" in df.columns
            ):

                weekday_df = df.copy()

                weekday_df["date"] = pd.to_datetime(
                    weekday_df["date"],
                    errors="coerce"
                )

                weekday_df["demand"] = pd.to_numeric(
                    weekday_df["demand"],
                    errors="coerce"
                )

                weekday_df = weekday_df.dropna(
                    subset=["date"]
                )

                weekday_demand = (

                    weekday_df
                    .groupby(
                        weekday_df["date"].dt.day_name()
                    )["demand"]
                    .sum()
                    .reset_index()

                )

                weekday_demand.columns = [
                    "Weekday",
                    "Demand"
                ]

                weekday_demand.to_excel(
                    writer,
                    sheet_name="Weekday Demand",
                    index=False
                )


            # -----------------------------------------------------
            # 6. INVENTORY ANALYSIS
            # -----------------------------------------------------

            if "inventory" in df.columns:

                inventory_df = df.copy()

                inventory_df["inventory"] = pd.to_numeric(
                    inventory_df["inventory"],
                    errors="coerce"
                )

                inventory_summary = pd.DataFrame({

                    "Metric": [
                        "Total Inventory",
                        "Average Inventory",
                        "Minimum Inventory",
                        "Maximum Inventory"
                    ],

                    "Value": [
                        inventory_df["inventory"].sum(),
                        inventory_df["inventory"].mean(),
                        inventory_df["inventory"].min(),
                        inventory_df["inventory"].max()
                    ]

                })

                inventory_summary.to_excel(
                    writer,
                    sheet_name="Inventory",
                    index=False
                )


            # -----------------------------------------------------
            # 7. FORECAST RESULTS
            # -----------------------------------------------------

            forecast_path = os.path.join(
                FORECAST_FOLDER,
                "forecast_results.csv"
            )

            if os.path.exists(forecast_path):

                try:

                    forecast_df = pd.read_csv(
                        forecast_path,
                        low_memory=False
                    )

                    forecast_df.to_excel(
                        writer,
                        sheet_name="Forecast Results",
                        index=False
                    )

                except Exception:
                    pass


        # ---------------------------------------------------------
        # PREPARE FILE FOR DOWNLOAD
        # ---------------------------------------------------------

        output.seek(0)


        return send_file(

            output,

            mimetype=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),

            as_attachment=True,

            download_name=(
                "retail_business_intelligence_report.xlsx"
            )

        )


    except Exception as e:

        return (
            f"Unable to generate the BI report: {str(e)}",
            500
        )

# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
@login_required
def reports():
    source_context = get_module_source_context("reports")
    dataset_info = get_shared_dataset_info()
    reports_dashboard = load_reports_dashboard()

    return render_template(
        "reports.html",
        module_name="reports",
        shared_dataset_available=dataset_info["available"],
        shared_dataset_rows=dataset_info["rows"],
        shared_dataset_columns=dataset_info["columns"],
        reports_dashboard=reports_dashboard,

        # Microsoft Power BI
        powerbi_report_url=POWERBI_REPORT_URL,
        powerbi_embed_url=POWERBI_EMBED_URL,
        powerbi_available=bool(POWERBI_EMBED_URL),

        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )
# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

with app.app_context():
    db.create_all()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
