"""
Runs the complete Weather Forecast System pipeline end-to-end:
generate data -> clean -> engineer features -> EDA charts -> train models
-> train classifier.

Usage:
    python main.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from generate_dataset import generate_weather_dataset
from data_cleaning import clean_weather_data
from feature_engineering import build_feature_dataset
from eda import run_eda
from train_models import train_all_regression_models
from train_classifier import train_weather_condition_classifier


def main():
    print("Step 1/6: Generating sample weather dataset...")
    generate_weather_dataset()

    print("\nStep 2/6: Cleaning weather data...")
    clean_weather_data()

    print("\nStep 3/6: Engineering features...")
    build_feature_dataset()

    print("\nStep 4/6: Running exploratory data analysis and saving charts...")
    run_eda()

    print("\nStep 5/6: Training regression models (temperature, rainfall, humidity)...")
    train_all_regression_models()

    print("\nStep 6/6: Training weather condition classifier...")
    train_weather_condition_classifier()

    print("\nPipeline complete!")
    print("Trained models are in 'models/', charts are in 'outputs/'.")
    print("Try a prediction with: python src/predict.py")


if __name__ == "__main__":
    main()
