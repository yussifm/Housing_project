import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")

# Data Cleaning
df = df.dropna()  # Remove missing values

# Keep only rows where 'total_sqft' is numeric
df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '').isdigit())]
df['total_sqft'] = df['total_sqft'].astype(float)  # Convert to float

# If needed, convert 'size' to numeric by extracting the number (e.g., "2 BHK" or "2 Bedroom" -> 2)
if 'size' in df.columns:
    df['size'] = df['size'].str.extract(r"(\d+)").astype(float)

# Updated feature list (using 'size' instead of 'bhk')
features = ['total_sqft', 'bath', 'balcony', 'size']
target = 'price'

# Filter the dataframe to contain only the required columns
df = df[features + [target]]

# Train-Test Split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (XGBoost can work without scaling but we keep it for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model: XGBoost Regressor
model = xgb.XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Model Performance
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"XGBoost Regressor - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"XGBoost Regressor - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

# Save Model & Scaler
joblib.dump(model, "house_price_xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("XGBoost model training complete! Saved as 'house_price_xgboost_model.pkl'.")

# -------- PLOTTING CHARTS -------- #

# 1️⃣ Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# 2️⃣ Feature vs Price Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.scatterplot(x=df[feature], y=df['price'], ax=axes[row, col], color='red')
    axes[row, col].set_title(f"{feature} vs Price")
plt.tight_layout()
plt.show()

# 3️⃣ Training vs Test Predictions (using a subset of training data for clarity)
plt.figure(figsize=(8, 5))
plt.plot(y_train.values[:100], label="Actual Prices", marker='o')
plt.plot(y_train_pred[:100], label="Predicted Prices", linestyle="dashed", marker='s')
plt.title("Actual vs Predicted Prices (Train Data) - XGBoost")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.legend()
plt.show()
