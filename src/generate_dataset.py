"""
Generates a synthetic historical weather dataset with realistic seasonal
patterns for temperature, humidity, wind speed and rainfall.

Some missing values and duplicate rows are intentionally injected so the
data cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from config import RAW_DATA_PATH, RANDOM_STATE


def _weather_condition(rainfall, wind_speed, humidity):
    if rainfall > 20 and wind_speed > 30:
        return "Stormy"
    if rainfall > 5:
        return "Rainy"
    if humidity > 70:
        return "Cloudy"
    return "Sunny"


def generate_weather_dataset(start_date="2022-01-01", num_days=1095, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    day_of_year = dates.dayofyear.values

    # Seasonal temperature curve (peaks around day ~200, dips in winter)
    seasonal_temp = 18 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    temperature = seasonal_temp + rng.normal(0, 3, num_days)

    # Humidity tends to be higher when it's cooler
    seasonal_humidity = 65 - 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    humidity = np.clip(seasonal_humidity + rng.normal(0, 8, num_days), 20, 100)

    wind_speed = np.clip(rng.normal(15, 6, num_days), 0, None)

    rain_chance = np.clip((humidity - 40) / 60, 0, 1)
    rainfall = np.where(
        rng.random(num_days) < rain_chance * 0.5,
        rng.gamma(2, 5, num_days),
        0.0,
    )
    wind_speed = wind_speed + rainfall * 0.3  # storms bring stronger wind

    temperature = np.round(temperature, 1)
    humidity = np.round(humidity, 1)
    wind_speed = np.round(wind_speed, 1)
    rainfall = np.round(rainfall, 1)

    conditions = [
        _weather_condition(r, w, h)
        for r, w, h in zip(rainfall, wind_speed, humidity)
    ]

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
            "weather_condition": conditions,
        }
    )

    # Inject missing values (~2% per numeric column) to simulate real data
    for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * num_days), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject duplicate rows (~1%) to simulate messy raw data
    dup_rows = df.sample(n=int(0.01 * num_days), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Sample weather dataset generated: {RAW_DATA_PATH} (shape={df.shape})")
    return df


if __name__ == "__main__":
    generate_weather_dataset()
