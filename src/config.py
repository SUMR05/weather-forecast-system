"""Central paths and constants used across the project."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_weather_data.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_weather_data.csv")
FEATURED_DATA_PATH = os.path.join(DATA_DIR, "featured_weather_data.csv")

RANDOM_STATE = 42

for _directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_directory, exist_ok=True)
