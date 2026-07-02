"""
Trains Linear Regression, Decision Tree and Random Forest regressors for
each numeric forecasting target (temperature, rainfall, humidity), evaluates
them with MAE/MSE/RMSE/R2, and saves the best performing model per target.
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from config import FEATURED_DATA_PATH, MODELS_DIR, RANDOM_STATE
from model_utils import get_features_for_target, regression_metrics, save_model

TARGETS = ["temperature", "rainfall", "humidity"]


def train_regression_for_target(df, target):
    features = get_features_for_target(target)
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_model, best_r2 = None, None, -float("inf")

    print(f"\n=== Training models for target: {target} ===")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        results[name] = metrics
        print(
            f"{name:15s} MAE={metrics['MAE']:.3f}  MSE={metrics['MSE']:.3f}  "
            f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}"
        )

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_name = name
            best_model = model

    print(f">>> Best model for '{target}': {best_name} (R2={best_r2:.3f})")

    save_model(best_model, os.path.join(MODELS_DIR, f"best_{target}_model.pkl"))
    save_model(features, os.path.join(MODELS_DIR, f"{target}_features.pkl"))

    return results, best_name


def train_all_regression_models():
    df = pd.read_csv(FEATURED_DATA_PATH)
    summary = {}
    for target in TARGETS:
        results, best_name = train_regression_for_target(df, target)
        summary[target] = {"results": results, "best_model": best_name}
    return summary


if __name__ == "__main__":
    train_all_regression_models()
