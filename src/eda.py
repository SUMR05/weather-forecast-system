"""
Exploratory data analysis: prints summary statistics and saves charts for
temperature, rainfall, humidity, wind speed and monthly weather trends.
"""
import os

import matplotlib
matplotlib.use("Agg")  # render to file, no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FEATURED_DATA_PATH, OUTPUTS_DIR

sns.set_style("whitegrid")


def load_data():
    return pd.read_csv(FEATURED_DATA_PATH, parse_dates=["date"])


def plot_temperature_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["temperature"], color="tomato", linewidth=1)
    plt.title("Temperature Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "temperature_trend.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved chart: {path}")


def plot_rainfall_trend(df):
    plt.figure(figsize=(12, 5))
    plt.bar(df["date"], df["rainfall"], color="royalblue", width=1)
    plt.title("Rainfall Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "rainfall_trend.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved chart: {path}")


def plot_humidity_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["humidity"], color="seagreen", linewidth=1)
    plt.title("Humidity Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "humidity_trend.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved chart: {path}")


def plot_wind_speed_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["wind_speed"], bins=30, kde=True, ax=axes[0], color="slateblue")
    axes[0].set_title("Wind Speed Distribution")
    axes[0].set_xlabel("Wind Speed (km/h)")

    sns.scatterplot(x="wind_speed", y="temperature", hue="weather_condition", data=df, ax=axes[1])
    axes[1].set_title("Wind Speed vs Temperature")
    axes[1].set_xlabel("Wind Speed (km/h)")
    axes[1].set_ylabel("Temperature (°C)")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "wind_speed_analysis.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved chart: {path}")


def plot_monthly_weather_changes(df):
    monthly_avg = df.groupby("month")[["temperature", "humidity", "rainfall", "wind_speed"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Monthly Weather Changes")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Value")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "monthly_weather_changes.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved chart: {path}")

    plt.figure(figsize=(10, 6))
    sns.countplot(x="month", hue="weather_condition", data=df)
    plt.title("Weather Condition Counts by Month")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    path2 = os.path.join(OUTPUTS_DIR, "monthly_condition_counts.png")
    plt.savefig(path2)
    plt.close()
    print(f"Saved chart: {path2}")


def run_eda():
    df = load_data()

    print("\n=== Dataset summary ===")
    print(df.describe())

    print("\n=== Weather condition counts ===")
    print(df["weather_condition"].value_counts())

    plot_temperature_trend(df)
    plot_rainfall_trend(df)
    plot_humidity_trend(df)
    plot_wind_speed_analysis(df)
    plot_monthly_weather_changes(df)


if __name__ == "__main__":
    run_eda()
