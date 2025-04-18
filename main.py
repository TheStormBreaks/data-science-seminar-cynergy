# -*- coding: utf-8 -*-
"""
Bangalore City Traffic Prediction
Modified version integrating Kaggle dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import logging
import os
import zipfile

# Configuration
DATA_PATH = 'data/banglore-city-traffic-dataset.zip'  # Update this path
TIME_STEP = 24  # Using 24 hours as time step for traffic data
EPOCHS = 30  
BATCH_SIZE = 32
N_SPLITS = 5 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_kaggle_data(filepath):
    """Load and extract Kaggle dataset"""
    try:
        # Extract zip file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('data/')
        
        # Find the CSV file in extracted contents
        for file in os.listdir('data'):
            if file.endswith('.csv'):
                csv_path = os.path.join('data', file)
                df = pd.read_csv(csv_path)
                logging.info(f"Successfully loaded data from {csv_path}")
                return df
        
        raise FileNotFoundError("No CSV file found in the extracted dataset")
    except Exception as e:
        logging.error(f"Error loading Kaggle data: {e}")
        exit()

def preprocess_traffic_data(df):
    """Preprocess traffic data"""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Lag features
    for lag in [1, 2, 3, 24]:  # 1h, 2h, 3h, and 24h lags
        df[f'lag_{lag}'] = df['traffic'].shift(lag)
    
    # Rolling statistics
    df['rolling_3h_mean'] = df['traffic'].shift(1).rolling(window=3).mean()
    df['rolling_24h_mean'] = df['traffic'].shift(1).rolling(window=24).mean()
    
    df.dropna(inplace=True)
    return df

def evaluate_model(model, X_test, Y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, predictions)
    mape = mean_absolute_percentage_error(Y_test, predictions)
    logging.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}%")
    return predictions, mae, rmse, r2, mape

def tune_random_forest(X_train, Y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_xgboost(X_train, Y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    model = XGBRegressor(random_state=42)
    grid_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10)
    grid_search.fit(X_train, Y_train)
    logging.info(f"Best parameters for XGBoost: {grid_search.best_params_}")
    return grid_search.best_estimator_

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold):
    plt.figure(figsize=(15, 6))
    plt.plot(Y_test, label='Actual Traffic', color='blue', linewidth=2)
    plt.plot(predictions_lr, label='Linear Regression', color='orange', linestyle='--')
    plt.plot(predictions_rf, label='Random Forest', color='green', linestyle='-.')
    plt.plot(predictions_xgb, label='XGBoost', color='red', linestyle=':')
    plt.plot(predictions_ensemble, label='Ensemble', color='purple', linestyle='-')
    plt.title(f'Bangalore Traffic Prediction (Fold {fold + 1})')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)

def main():
    # Load and preprocess data
    df = load_kaggle_data(DATA_PATH)
    df = preprocess_traffic_data(df)
    
    # Prepare features - adjust based on your actual dataset columns
    feature_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 
                   'lag_1', 'lag_2', 'lag_3', 'lag_24', 
                   'rolling_3h_mean', 'rolling_24h_mean']
    
    X = df[feature_cols].values
    Y = df['traffic'].values  # Assuming 'traffic' is the target column

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    plot_figures = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logging.info(f"Fold {fold + 1}: Train samples: {len(train_index)}, Test samples: {len(test_index)}")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Linear Regression
        model_lr = LinearRegression()
        model_lr.fit(X_train, Y_train)
        predictions_lr, mae_lr, rmse_lr, r2_lr, mape_lr = evaluate_model(model_lr, X_test, Y_test, "Linear Regression")

        # Random Forest
        model_rf = tune_random_forest(X_train, Y_train)
        predictions_rf, mae_rf, rmse_rf, r2_rf, mape_rf = evaluate_model(model_rf, X_test, Y_test, "Random Forest")

        # XGBoost
        model_xgb = tune_xgboost(X_train, Y_train)
        predictions_xgb, mae_xgb, rmse_xgb, r2_xgb, mape_xgb = evaluate_model(model_xgb, X_test, Y_test, "XGBoost")

        # Ensemble model
        ensemble_model = VotingRegressor([
            ('lr', model_lr),
            ('rf', model_rf),
            ('xgb', model_xgb)
        ])
        ensemble_model.fit(X_train, Y_train)
        predictions_ensemble, mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble = evaluate_model(
            ensemble_model, X_test, Y_test, "Ensemble")

        results.append({
            'fold': fold + 1,
            'Linear Regression': (mae_lr, rmse_lr, r2_lr, mape_lr),
            'Random Forest': (mae_rf, rmse_rf, r2_rf, mape_rf),
            'XGBoost': (mae_xgb, rmse_xgb, r2_xgb, mape_xgb),
            'Ensemble': (mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble)
        })

        plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold)
        plot_figures.append(plt.gcf())

    # LSTM Model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['traffic']].values)

    X_lstm, Y_lstm = create_dataset(scaled_data, TIME_STEP)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    model_lstm = create_lstm_model((X_lstm.shape[1], 1))
    history = model_lstm.fit(X_lstm, Y_lstm, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.title('LSTM Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Make predictions
    predictions_lstm = model_lstm.predict(X_lstm)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)
    Y_lstm_original = scaler.inverse_transform(Y_lstm.reshape(-1, 1))

    # Plot LSTM results
    plt.figure(figsize=(15, 6))
    plt.plot(Y_lstm_original, label='Actual Traffic', color='blue', linewidth=2)
    plt.plot(predictions_lstm, label='LSTM Predictions', color='red', linestyle='--')
    plt.title('Bangalore Traffic Prediction (LSTM Model)')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show all results
    results_df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(results_df)

if __name__ == "__main__":
    main()