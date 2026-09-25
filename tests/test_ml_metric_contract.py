import pandas as pd

from services.analytics_service import prepare_analytics_data, train_demand_classification_model, detect_demand_anomalies
from services.inventory_service import train_stockout_model


def test_ml_contract_metrics_exist_and_anomaly_is_not_80_percent():
    dataset = pd.read_csv('data/cleaned/cleaned_shared_dataset.csv')

    prep = prepare_analytics_data(dataset)
    analytics_df = prep['data']

    classification = train_demand_classification_model(analytics_df)
    assert classification['available'] is True
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        assert key in classification

    anomaly = detect_demand_anomalies(analytics_df)
    assert anomaly['available'] is True
    assert anomaly['anomaly_percentage'] < 60.0

    stockout = train_stockout_model(dataset)
    assert stockout['model_available'] is True
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        assert key in stockout
