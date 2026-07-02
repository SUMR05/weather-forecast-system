"""Data cleaning step: handles missing values, duplicates and invalid ranges."""
import pandas as pd

from config import RAW_DATA_PATH, CLEANED_DATA_PATH

NUMERIC_COLUMNS = ["temperature", "humidity", "wind_speed", "rainfall"]


def clean_weather_data():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    print(f"Raw data shape: {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum())

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    df = df.sort_values("date").reset_index(drop=True)

    # Fill missing numeric values with the column median (robust to outliers)
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with the most frequent condition
    df["weather_condition"] = df["weather_condition"].fillna(
        df["weather_condition"].mode()[0]
    )

    # Clip values to physically valid ranges
    df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)
    df["rainfall"] = df["rainfall"].clip(lower=0)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH} (shape={df.shape})")
    return df


if __name__ == "__main__":
    clean_weather_data()
