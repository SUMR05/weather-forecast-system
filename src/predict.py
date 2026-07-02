"""
Prediction script: loads the trained models and produces a forecast
(temperature, rainfall, humidity, weather condition) for new weather inputs.

Run with defaults:
    python predict.py

Run with custom inputs:
    python predict.py --date 2026-08-01 --temperature 31 --humidity 55 --wind_speed 10 --rainfall 0
"""
import argparse
import os

import pandas as pd

from config import MODELS_DIR
from feature_engineering import SEASON_CODE, SEASON_MAP
from model_utils import load_model


def _derive_date_features(date_str):
    date = pd.to_datetime(date_str)
    month = date.month
    day_of_year = date.dayofyear
    season = SEASON_MAP[month]
    season_code = SEASON_CODE[season]
    return month, day_of_year, season_code


def predict_weather(date_str, temperature, humidity, wind_speed, rainfall):
    month, day_of_year, season_code = _derive_date_features(date_str)

    raw = {
        "month": month,
        "day_of_year": day_of_year,
        "season_code": season_code,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rainfall": rainfall,
    }

    predictions = {}

    for target in ["temperature", "rainfall", "humidity"]:
        model = load_model(os.path.join(MODELS_DIR, f"best_{target}_model.pkl"))
        features = load_model(os.path.join(MODELS_DIR, f"{target}_features.pkl"))
        X = pd.DataFrame([{f: raw[f] for f in features}])
        predictions[target] = round(float(model.predict(X)[0]), 2)

    clf = load_model(os.path.join(MODELS_DIR, "best_weather_condition_model.pkl"))
    label_encoder = load_model(os.path.join(MODELS_DIR, "weather_condition_label_encoder.pkl"))
    clf_features = load_model(os.path.join(MODELS_DIR, "weather_condition_features.pkl"))
    X_clf = pd.DataFrame([{f: raw[f] for f in clf_features}])
    condition_code = clf.predict(X_clf)[0]
    predictions["weather_condition"] = label_encoder.inverse_transform([condition_code])[0]

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict weather for new inputs.")
    parser.add_argument("--date", default="2026-07-15", help="Date to forecast (YYYY-MM-DD)")
    parser.add_argument("--temperature", type=float, default=29.0, help="Current/estimated temperature (°C)")
    parser.add_argument("--humidity", type=float, default=68.0, help="Current/estimated humidity (%)")
    parser.add_argument("--wind_speed", type=float, default=14.0, help="Current/estimated wind speed (km/h)")
    parser.add_argument("--rainfall", type=float, default=2.0, help="Current/estimated rainfall (mm)")
    args = parser.parse_args()

    result = predict_weather(
        date_str=args.date,
        temperature=args.temperature,
        humidity=args.humidity,
        wind_speed=args.wind_speed,
        rainfall=args.rainfall,
    )

    print(f"\n=== Weather Forecast for {args.date} ===")
    print(f"Input  -> temperature: {args.temperature}°C, humidity: {args.humidity}%, "
          f"wind speed: {args.wind_speed} km/h, rainfall: {args.rainfall} mm")
    print("Predictions:")
    print(f"  Temperature      : {result['temperature']} °C")
    print(f"  Rainfall         : {result['rainfall']} mm")
    print(f"  Humidity         : {result['humidity']} %")
    print(f"  Weather condition: {result['weather_condition']}")


if __name__ == "__main__":
    main()
