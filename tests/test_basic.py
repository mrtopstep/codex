import pandas as pd
from war_data_loader import WarDataLoader
from war_prediction_model_improved import WarPredictionModel

def test_data_loader():
    loader = WarDataLoader(start_year=2000, end_year=2001, random_state=0)
    df = loader.load_all_data()
    assert not df.empty
    assert "gini" in df.columns

def test_model_training():
    loader = WarDataLoader(start_year=2000, end_year=2001, random_state=0)
    df = loader.load_all_data()
    model = WarPredictionModel(random_state=0)
    features = model.create_features(df)
    X, y_clf, y_reg = model.prepare_data(features)
    model.train_models(X, y_clf, y_reg)
    assert "logistic" in model.models
    assert "regressor" in model.models
