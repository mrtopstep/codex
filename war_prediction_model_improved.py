"""
war_prediction_model_improved.py

Provides simple machine learning models for predicting war
occurrence and intensity.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class WarPredictionModel:
    """Machine learning model for war prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for modeling."""
        features = df.copy()
        features["gdp_unemployment_ratio"] = features["gini"] / (1 + features["unemployment"])
        features["tension_trade_interaction"] = (
            features["geopolitical_tension"] * features["trade_connectivity"]
        )
        return features

    def prepare_data(self, df: pd.DataFrame):
        """Prepare numpy arrays for model training."""
        cols = [c for c in df.columns if c not in ["country", "year", "war_occurrence", "war_intensity"]]
        self.feature_names = cols
        X = df[cols].values
        y_clf = df["war_occurrence"].values
        y_reg = df["war_intensity"].values
        X_scaled = self.scale_features(X, fit=True)
        return X_scaled, y_clf, y_reg

    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def train_models(self, X: np.ndarray, y_clf: np.ndarray, y_reg: np.ndarray) -> None:
        """Train classification and regression models."""
        clf = LogisticRegression(max_iter=200, random_state=self.random_state)
        clf.fit(X, y_clf)

        reg = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        reg.fit(X, y_reg)

        self.models["logistic"] = clf
        self.models["regressor"] = reg

        self.results["classification"] = {"accuracy": clf.score(X, y_clf)}
        self.results["regression"] = {"r2": reg.score(X, y_reg)}
        self.results["feature_importance"] = reg.feature_importances_
