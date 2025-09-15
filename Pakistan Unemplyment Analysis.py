
# 📌 Unemployment Analysis with Python

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Step 2: Load dataset
df = pd.read_csv("Pakistan_Poverty_Dataset_2000_2023.csv")

# Step 3: Explore data
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Step 4: Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Step 5: Descriptive statistics
print("\nSummary statistics:\n", df.describe())

# Step 6: Unemployment trend
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="Year", y="Unemployment Rate (%)", marker="o", color="red")
plt.title("Unemployment Rate Trend (2000-2023)", fontsize=14)
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# Step 7: Compare with other indicators
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="Year", y="Poverty Headcount Ratio (%)", label="Poverty Rate")
sns.lineplot(data=df, x="Year", y="Unemployment Rate (%)", label="Unemployment Rate")
plt.title("Poverty vs Unemployment Trend", fontsize=14)
plt.ylabel("Percentage (%)")
plt.legend()
plt.show()

# Step 8: Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=14)
plt.show()

# Step 9: Relationship plots
sns.pairplot(df[["Unemployment Rate (%)", "GDP Growth Rate (%)", "Inflation Rate (%)", "Poverty Headcount Ratio (%)"]])
plt.show()

# ----------------------------
# 📌 Machine Learning - Linear Regression
# ----------------------------

# Predict Unemployment based on GDP Growth, Inflation, and Poverty
X = df[["GDP Growth Rate (%)", "Inflation Rate (%)", "Poverty Headcount Ratio (%)"]]
y = df["Unemployment Rate (%)"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("\nLinear Regression Performance:")
print("R2 Score:", r2_score(y, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y, y_pred))

# Plot predictions
plt.figure(figsize=(10,5))
plt.plot(df["Year"], y, label="Actual", marker="o")
plt.plot(df["Year"], y_pred, label="Predicted", linestyle="--")
plt.title("Unemployment Prediction (Linear Regression)", fontsize=14)
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# ----------------------------
# 📌 Time Series Forecasting (ARIMA)
# ----------------------------
unemp_series = df.set_index("Year")["Unemployment Rate (%)"]

# Fit ARIMA model
model_arima = ARIMA(unemp_series, order=(2,1,2))
arima_result = model_arima.fit()

# Forecast next 5 years
forecast = arima_result.forecast(steps=5)
print("\nARIMA Forecast (next 5 years):")
print(forecast)

# Plot ARIMA forecast
plt.figure(figsize=(10,5))
plt.plot(unemp_series, label="Historical Data")
plt.plot(range(2024, 2029), forecast, label="Forecast", marker="o", linestyle="--", color="red")
plt.title("Unemployment Forecast (ARIMA)", fontsize=14)
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()