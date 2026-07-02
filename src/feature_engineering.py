"""Feature engineering: derives date, month, season and calendar features."""
import pandas as pd

from config import CLEANED_DATA_PATH, FEATURED_DATA_PATH

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_CODE = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}


def add_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].map(SEASON_MAP)
    df["season_code"] = df["season"].map(SEASON_CODE)
    return df


def build_feature_dataset():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = add_features(df)
    df.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Featured data saved to {FEATURED_DATA_PATH} (shape={df.shape})")
    return df


if __name__ == "__main__":
    build_feature_dataset()
