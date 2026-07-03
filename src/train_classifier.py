"""
Trains classifiers to predict the weather condition category
(Sunny, Cloudy, Rainy, Stormy) and saves the best performing one.
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from config import FEATURED_DATA_PATH, MODELS_DIR, RANDOM_STATE
from model_utils import classification_metrics, save_model

FEATURES = ["month", "day_of_year", "season_code", "temperature", "humidity", "wind_speed", "rainfall"]


def train_weather_condition_classifier():
    df = pd.read_csv(FEATURED_DATA_PATH)
    X = df[FEATURES]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["weather_condition"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
    }

    best_name, best_model, best_metrics = None, None, None
    best_acc = -1.0
    results = {}

    print("\n=== Training weather condition classifiers ===")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = classification_metrics(y_test, preds)
        results[name] = metrics

        print(
            f"\n{name:20s} Accuracy={metrics['Accuracy']:.3f}  "
            f"Precision={metrics['Precision']:.3f}  "
            f"Recall={metrics['Recall']:.3f}  "
            f"F1={metrics['F1']:.3f}"
        )
        print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

        if metrics["Accuracy"] > best_acc:
            best_acc = metrics["Accuracy"]
            best_name = name
            best_model = model
            best_metrics = metrics

    print(
        f">>> Best classifier: {best_name} "
        f"(Accuracy={best_metrics['Accuracy']:.3f}, Precision={best_metrics['Precision']:.3f}, "
        f"Recall={best_metrics['Recall']:.3f}, F1={best_metrics['F1']:.3f})"
    )

    save_model(best_model, os.path.join(MODELS_DIR, "best_weather_condition_model.pkl"))
    save_model(label_encoder, os.path.join(MODELS_DIR, "weather_condition_label_encoder.pkl"))
    save_model(FEATURES, os.path.join(MODELS_DIR, "weather_condition_features.pkl"))

    return best_name, best_metrics


if __name__ == "__main__":
    train_weather_condition_classifier()
