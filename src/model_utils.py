"""Shared helpers for feature selection, evaluation metrics and model I/O."""
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# Every regression/classification model draws from this same pool of columns.
# For a given target, that target itself is excluded from its own features.
BASE_FEATURES = [
    "month", "day_of_year", "season_code",
    "temperature", "humidity", "wind_speed", "rainfall",
]


def get_features_for_target(target_col):
    return [c for c in BASE_FEATURES if c != target_col]


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def classification_metrics(y_true, y_pred):
    """Accuracy, Precision, Recall and F1 (weighted across all classes)."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def save_model(obj, path):
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_model(path):
    return joblib.load(path)
