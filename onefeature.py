# === Import necessary libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === Load the dataset ===
df = pd.read_csv("covid_19_india.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# === Aggregate data ===
df = df.groupby('Date')[['Confirmed', 'Cured', 'Deaths']].sum().reset_index()

# === Create new cases column ===
df['New_Cases'] = df['Confirmed'].diff().fillna(0)
df['New_Cases'] = df['New_Cases'].apply(lambda x: x if x > 0 else 0)


# === Create lag features ===
df['Lag_1'] = df['New_Cases'].shift(1)
df['Lag_7'] = df['New_Cases'].shift(7)
df['Lag_14'] = df['New_Cases'].shift(14)
df = df.dropna()

# === Time range and splitting into two halves ===
min_date = df['Date'].min()
max_date = df['Date'].max()

# Find the midpoint date
mid_date = min_date + (max_date - min_date) / 2

# Split the data into two halves
df_first_half = df[df['Date'] <= mid_date]
df_second_half = df[df['Date'] > mid_date]

# Filter data for the first half (2020-02 to 2020-10)
df_first_half = df[(df['Date'] >= '2020-02-01') & (df['Date'] <= '2020-10-31')]


# Scatter plot for Lag_1 vs Time in the first half
plt.figure(figsize=(14, 6))
plt.scatter(df_first_half['Date'], df_first_half['Lag_1'], color='blue', alpha=0.6, label="Lag_1 (Previous Day Cases)")

# Fit a linear trend line for visualization
trend_model = LinearRegression()
date_ordinal = np.array(df_first_half['Date'].map(pd.Timestamp.toordinal)).reshape(-1, 1)
trend_model.fit(date_ordinal, df_first_half['Lag_1'])
line_trend = trend_model.predict(date_ordinal)
# Plot regression line
plt.plot(df_first_half['Date'], line_trend, color='red', linewidth=2, label="Trend Line")
# Labels and aesthetics
plt.xlabel("Date")
plt.ylabel("Lag_1 (Previous Day Cases)")
plt.title("Scatter Plot of Lag_1 vs Time (2020-02 to 2020-10) with Trend Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. Actual vs Predicted Plot (Second Half) ===
# Define features (X) and target for second half
X_second_half = df_second_half[['Lag_1']]  # Only one feature as per your request
y_second_half = df_second_half['New_Cases']
dates_second_half = df_second_half['Date']

# Split the second half into training and testing sets
X_train_second_half, X_test_second_half, y_train_second_half, y_test_second_half, dates_train_second_half, dates_test_second_half = train_test_split(
    X_second_half, y_second_half, dates_second_half, test_size=0.2, shuffle=False
)


# Split the second half into X and y (features and target)
X_second_half = df_second_half[['Lag_1']]  # Only using Lag_1
y_second_half = df_second_half['New_Cases']

# Train the Linear Regression model on the second half
lr_model_second_half = LinearRegression()
lr_model_second_half.fit(X_train_second_half, y_train_second_half)
y_pred_lr_second_half = lr_model_second_half.predict(X_test_second_half)
# Filter data for the second half (2020-11 to 2021-08)
df_second_half = df[(df['Date'] >= '2020-11-01') & (df['Date'] <= '2021-08-11')]


# Get the intercept and coefficient from the trained model
intercept = lr_model_second_half.intercept_
coefficient = lr_model_second_half.coef_[0]  # Since we have one feature (Lag_1)

# Print the equation of the linear regression model
print(f"Linear Regression Equation: y = {612.17:.2f} + ({0.99:.2f}) * Lag_1")

# Use the given equation to predict values
df_second_half['Predicted_New_Cases'] = 612.17 + 0.99 * df_second_half['Lag_1']

# Create a table with Date, Lag_1 and Predicted_New_Cases
table = df_second_half[['Date', 'Lag_1', 'Predicted_New_Cases']]

# Display the first 20 rows of the table
print(table.head(40))


# === Plot the predicted values ===
plt.figure(figsize=(14, 6))
plt.plot(df_second_half['Date'], df_second_half['Predicted_New_Cases'], label="Predicted New Cases", color='blue', linewidth=2)

# Labels and aesthetics
plt.xlabel("Date")
plt.ylabel("Predicted New Daily Cases")
plt.title("Predicted New COVID-19 Cases (2020-11 to 2021-08) Using Linear Regression Equation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Use the same Linear Regression model for prediction
y_pred_second_half = lr_model_second_half.predict(X_second_half)

# === Actual vs Predicted Plot for Second Half ===
plt.figure(figsize=(14, 6))
plt.plot(df_second_half['Date'], y_second_half.values, label="Actual Cases", color='black', linewidth=2)
plt.plot(df_second_half['Date'], y_pred_second_half, label="Linear Regression Prediction", linestyle='--', color='orange')

# Labels and aesthetics
plt.xlabel("Date")
plt.ylabel("New Daily Cases")
plt.title("Actual vs Predicted COVID-19 Cases (2020-11 to 2021-08)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
