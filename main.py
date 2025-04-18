import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

# Load data
df = pd.read_csv("covid_19_india.csv")

# Parse date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Filter only national total data (we'll use "India" level total)
df = df.groupby('Date')[['Confirmed', 'Cured', 'Deaths']].sum().reset_index()

# Engineer daily new cases
df['New_Cases'] = df['Confirmed'].diff().fillna(0)
df['New_Cases'] = df['New_Cases'].apply(lambda x: x if x > 0 else 0)  # remove negative diffs

# Feature engineering (lag features)
df['Lag_1'] = df['New_Cases'].shift(1)
df['Lag_7'] = df['New_Cases'].shift(7)
df['Lag_14'] = df['New_Cases'].shift(14)
df = df.dropna()

# Train-test split
X = df[['Lag_1', 'Lag_7', 'Lag_14']]
y = df['New_Cases']
dates = df['Date']

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False
)

# === 1. Linear Regression ===
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# === 2. LightGBM ===
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

# === 3. XGBoost ===
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# === Evaluation ===
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }

results = [
    evaluate_model(y_test, y_pred_lr, "Linear Regression"),
    evaluate_model(y_test, y_pred_lgb, "LightGBM"),
    evaluate_model(y_test, y_pred_xgb, "XGBoost"),
]

results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:\n", results_df)

# === Plot Actual vs Predicted ===
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test.values, label="Actual Cases", color='black', linewidth=2)
plt.plot(dates_test, y_pred_lr, label="Linear Regression", linestyle='--')
plt.plot(dates_test, y_pred_lgb, label="LightGBM", linestyle='--')
plt.plot(dates_test, y_pred_xgb, label="XGBoost", linestyle='--')
plt.xlabel("Date")
plt.ylabel("New Daily Cases")
plt.title("Actual vs Predicted COVID-19 Cases in India")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
