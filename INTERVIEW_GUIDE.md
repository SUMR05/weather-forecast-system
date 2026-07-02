# Weather Forecast System — Complete Project Guide & Interview Prep

This document explains the whole project in simple words: what each folder
and file does, how data flows through the pipeline, and includes sample
interview questions with easy answers so you can confidently explain this
project in an interview.

---

## 1. What This Project Does (One-Minute Summary)

This is a Machine Learning project that:
1. Takes historical daily weather data (date, temperature, humidity, wind
   speed, rainfall, weather condition).
2. Cleans the data (fixes missing values, removes duplicates).
3. Creates new useful columns from the date (month, season, etc.) — this is
   called **feature engineering**.
4. Trains 3 ML models (Linear Regression, Decision Tree, Random Forest) to
   predict temperature, rainfall, and humidity.
5. Trains a classification model to predict the weather condition
   (Sunny / Cloudy / Rainy / Stormy).
6. Compares all models using standard metrics and automatically picks the
   **best one**.
7. Saves the best models to disk so they can be reused later without
   retraining (using Joblib).
8. Lets you type in new weather values and get an instant forecast.

Everything runs locally with plain Python — no internet, no external
weather API.

---

## 2. Folder-by-Folder Explanation (Interview-Friendly)

```
weather-forecast-system/
├── data/
├── models/
├── outputs/
├── src/
├── main.py
├── requirements.txt
└── README.md
```

### 📁 `data/` — Where all datasets live
| File | What it is |
|---|---|
| `raw_weather_data.csv` | The original sample dataset. Has some missing values and duplicate rows on purpose, to simulate real-world messy data. |
| `cleaned_weather_data.csv` | Same data, after removing duplicates and filling missing values. |
| `featured_weather_data.csv` | Same data, after adding new columns like `month`, `season`, `day_of_year`. This is the final table used to train the models. |

**Interview line:** *"I keep raw, cleaned, and feature-engineered data as
separate files so every stage of the pipeline is visible and debuggable —
you can literally open each CSV and see what changed at that step."*

### 📁 `models/` — Where trained models are saved
Contains `.pkl` files created by **Joblib**. A `.pkl` file is just a trained
model saved to disk so we don't need to retrain it every time.

| File | What it is |
|---|---|
| `best_temperature_model.pkl` | The winning model for predicting temperature |
| `best_rainfall_model.pkl` | The winning model for predicting rainfall |
| `best_humidity_model.pkl` | The winning model for predicting humidity |
| `best_weather_condition_model.pkl` | The winning classifier for weather condition |
| `weather_condition_label_encoder.pkl` | Converts text labels (Sunny/Rainy/...) to numbers and back |
| `*_features.pkl` | Remembers exactly which columns each model expects as input |

**Interview line:** *"I save the model AND the list of feature names it was
trained on, together. This avoids a common bug where the prediction script
sends columns in the wrong order or a missing column."*

### 📁 `outputs/` — Where charts (visualizations) are saved
| File | What it shows |
|---|---|
| `temperature_trend.png` | Temperature over the full 3-year history |
| `rainfall_trend.png` | Rainfall over the full 3-year history |
| `humidity_trend.png` | Humidity over the full 3-year history |
| `wind_speed_analysis.png` | Wind speed distribution + wind speed vs temperature |
| `monthly_weather_changes.png` | Average weather values per month (bar chart) |
| `monthly_condition_counts.png` | How many Sunny/Rainy/Cloudy/Stormy days per month |

**Interview line:** *"These charts are generated automatically by the EDA
script — they help me understand seasonal patterns before I even train a
model."*

### 📁 `src/` — All the Python source code
This is the "engine room" of the project. Each file has ONE clear job:

| File | Job (in plain English) |
|---|---|
| `config.py` | Stores all file paths and settings in one place, so nothing is hardcoded everywhere else. |
| `generate_dataset.py` | Creates the sample weather dataset (since we don't use a live weather API). |
| `data_cleaning.py` | Fixes missing values, removes duplicate rows, fixes invalid values (e.g., humidity can't be negative). |
| `feature_engineering.py` | Adds new columns derived from the date: `month`, `day_of_year`, `season`. |
| `model_utils.py` | Shared helper code: which columns to use as features, how to calculate MAE/MSE/RMSE/R², how to save/load models. |
| `train_models.py` | Trains Linear Regression, Decision Tree, and Random Forest for temperature, rainfall, and humidity. Picks the best one per target. |
| `train_classifier.py` | Trains classifiers to predict the weather condition category (Sunny/Cloudy/Rainy/Stormy). |
| `eda.py` | Exploratory Data Analysis — prints statistics and draws all the charts saved in `outputs/`. |
| `predict.py` | Loads the saved models and gives a forecast for new weather inputs you type in. |

**Interview line:** *"I split the code by responsibility — one file per
step of the pipeline. This is called separation of concerns; it makes the
code easy to test, debug, and explain."*

### 📄 `main.py` — The "run everything" button
Runs all steps in order: generate data → clean → feature engineer → EDA →
train regression models → train classifier. One command, full pipeline.

### 📄 `requirements.txt`
The list of Python libraries needed (pandas, numpy, scikit-learn,
matplotlib, seaborn, joblib) so anyone can install them with one command:
`pip install -r requirements.txt`.

### 📄 `README.md`
Setup and usage instructions for someone cloning/using the project.

---

## 3. How Data Flows Through the Project (Pipeline Diagram)

```
generate_dataset.py
        │  raw_weather_data.csv (messy, has NaNs & duplicates)
        ▼
data_cleaning.py
        │  cleaned_weather_data.csv (no NaNs, no duplicates)
        ▼
feature_engineering.py
        │  featured_weather_data.csv (+ month, season, day_of_year)
        ▼
   ┌────┴─────┐
   ▼          ▼
eda.py    train_models.py / train_classifier.py
   │              │
charts in     best models saved
outputs/      in models/ (via Joblib)
                   │
                   ▼
              predict.py
        (loads saved models, predicts
         temperature/rainfall/humidity/
         weather condition for new input)
```

---

## 4. Key Machine Learning Concepts Used (Simple Explanations)

| Term | Simple meaning |
|---|---|
| **Feature** | An input column used to make a prediction (e.g., humidity, month). |
| **Target** | The value we are trying to predict (e.g., temperature). |
| **Linear Regression** | Draws a straight-line relationship between inputs and output. Fast, simple, but can't capture complex patterns. |
| **Decision Tree Regressor** | Splits data into branches using yes/no questions (e.g., "is humidity > 70?") to reach a prediction. Captures non-linear patterns. |
| **Random Forest Regressor** | Builds many decision trees and averages their answers. Usually more accurate and stable than a single tree. |
| **Classification** | Predicting a category (Sunny/Cloudy/Rainy/Stormy) instead of a number. |
| **MAE (Mean Absolute Error)** | Average of how far off predictions are, in original units (e.g., °C). Easy to interpret. |
| **MSE (Mean Squared Error)** | Like MAE but squares the errors first — punishes big mistakes more. |
| **RMSE (Root Mean Squared Error)** | Square root of MSE — brings the error back to the original unit (e.g., °C), easier to read than MSE. |
| **R² Score** | How much of the pattern in the data the model explains, from 0 to 1 (closer to 1 = better). |
| **Train/Test Split** | We train the model on 80% of the data and test it on the other 20% it has never seen, to check it generalizes. |
| **Joblib** | A library to save a trained model to a file (`.pkl`) and load it back later without retraining. |
| **Label Encoder** | Converts text categories (like "Sunny") into numbers (like 0), because ML models need numbers. |

---

## 5. Sample Interview Questions & Simple Answers

**Q1: What does this project do?**
> It predicts future temperature, rainfall, humidity, and weather condition
> (Sunny/Cloudy/Rainy/Stormy) using machine learning models trained on
> historical weather data.

**Q2: Why did you use three different regression models instead of just one?**
> To compare them fairly. Linear Regression is simple and fast but assumes a
> straight-line relationship. Decision Tree and Random Forest can capture
> more complex, non-linear patterns. I evaluate all three with the same
> metrics and automatically keep whichever one performs best.

**Q3: How do you decide which model is "best"?**
> I split the data into training and test sets, train each model on the
> training set, then compare their R² score on the unseen test set. The
> model with the highest R² is saved as the best model for that target.

**Q4: What is the difference between MAE, MSE, RMSE, and R²?**
> MAE and RMSE tell you the average prediction error in the same units as
> the target (e.g., degrees Celsius) — RMSE punishes large errors more than
> MAE. MSE is the squared version before taking the square root. R² tells
> you, on a scale of 0 to 1, how much of the variation in the data the model
> explains.

**Q5: Why is the rainfall prediction less accurate than temperature?**
> Rainfall is naturally noisy and close to random — it doesn't follow a
> smooth seasonal curve the way temperature does. This is expected and
> mirrors real-world weather forecasting, where rainfall is genuinely harder
> to predict than temperature.

**Q6: What is feature engineering, and what features did you create?**
> Feature engineering means creating new, useful input columns from existing
> data. From the `date` column, I derived `month`, `day_of_year`, and
> `season` (Winter/Spring/Summer/Autumn), because weather strongly depends
> on the time of year.

**Q7: How do you handle missing data and duplicates?**
> Missing numeric values (temperature, humidity, wind speed, rainfall) are
> filled with the column's median value, which is robust to outliers.
> Missing weather condition values are filled with the most frequent
> category. Duplicate rows are removed with `drop_duplicates()`.

**Q8: Why use the median instead of the mean to fill missing values?**
> The median is less affected by extreme outliers (like one freak storm day)
> than the mean, so it gives a more realistic "typical" value to fill gaps
> with.

**Q9: What is Joblib and why do you use it?**
> Joblib saves a trained model object to a file on disk. Once a model is
> saved, we can load it instantly and make predictions without retraining it
> every time the program runs — this is essential for deploying models in
> real applications.

**Q10: How does your prediction script work?**
> `predict.py` loads the saved best model (and its expected feature list)
> for each target, builds a small table of input features from what the
> user provides (date, temperature, humidity, wind speed, rainfall), and
> calls `model.predict()` on it to get the forecast.

**Q11: What's the difference between regression and classification in this project?**
> Regression predicts a continuous number, like temperature or rainfall
> amount. Classification predicts a category, like whether the day is
> Sunny, Cloudy, Rainy, or Stormy.

**Q12: Why do you exclude the target column from its own feature list?**
> Because a model can't use the value it's trying to predict as one of its
> own inputs — that would be cheating (called data leakage) and wouldn't
> work on new, real data where that value is unknown.

**Q13: What Python libraries did you use, and why?**
> - **Pandas** — loading, cleaning, and transforming tabular data.
> - **NumPy** — fast numerical operations, generating synthetic data.
> - **Scikit-learn** — the machine learning models, train/test split, and
>   evaluation metrics.
> - **Matplotlib & Seaborn** — drawing charts for exploratory data analysis.
> - **Joblib** — saving and loading trained models.

**Q14: How would you improve this project further?**
> I could add more real historical weather data instead of synthetic data,
> try more advanced models like Gradient Boosting or XGBoost, add
> hyperparameter tuning (e.g., GridSearchCV), and build a simple web
> interface (like Flask or Streamlit) so users can get forecasts through a
> browser instead of the command line.

**Q15: Is this project using real-time weather data?**
> No — it uses a locally generated synthetic dataset that mimics realistic
> seasonal weather patterns, so the whole project can run offline without
> needing an API key. The same pipeline would work unchanged on real
> historical weather data.

---

## 6. Quick One-Line Answers (For Fast-Paced Interview Rounds)

- **What language/tools?** → Python, Pandas, NumPy, Scikit-learn, Matplotlib,
  Seaborn, Joblib.
- **What's predicted?** → Temperature, rainfall, humidity (regression) and
  weather condition (classification).
- **Models used?** → Linear Regression, Decision Tree Regressor, Random
  Forest Regressor, plus classifiers (Logistic Regression, Decision Tree,
  Random Forest) for weather condition.
- **How is the best model chosen?** → Highest R² on the test set for
  regression; highest accuracy for classification.
- **How are models reused?** → Saved with Joblib as `.pkl` files, loaded
  later for prediction without retraining.
- **How do you test the model on new data?** → `python src/predict.py
  --date ... --temperature ... --humidity ... --wind_speed ... --rainfall ...`
