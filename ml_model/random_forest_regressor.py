import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report

# ================================
# Data Loading and Preprocessing
# ================================

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")

# Data Cleaning: Remove missing values
df = df.dropna()

# Keep only rows where 'total_sqft' is numeric
df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '').isdigit())]
df['total_sqft'] = df['total_sqft'].astype(float)

# Convert 'size' to numeric by extracting the number (e.g., "2 BHK" or "2 Bedroom" -> 2)
if 'size' in df.columns:
    df['size'] = df['size'].str.extract(r"(\d+)").astype(float)

# Updated feature list and target
features = ['total_sqft', 'bath', 'balcony', 'size']
target = 'price'
df = df[features + [target]]

# Train-Test Split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (optional for tree-based methods, but included for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Train Random Forest Regressor
# ================================

model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Compute Performance Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Random Forest Regressor - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Random Forest Regressor - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Random Forest Regressor - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")

# Save Model & Scaler
joblib.dump(model, "house_price_random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Random Forest model training complete! Saved as 'house_price_random_forest_model.pkl'.")

# ================================
# Additional: Confusion Metrics for Regression
# ================================
# Convert continuous prices into binary classes using the median as threshold
median_price_threshold = df['price'].median()
y_test_class = (y_test > median_price_threshold).astype(int)
y_test_pred_class = (y_test_pred > median_price_threshold).astype(int)

conf_matrix = confusion_matrix(y_test_class, y_test_pred_class)
print("\nConfusion Matrix (Test Data):")
print(conf_matrix)
print("\nClassification Report (Test Data):")
print(classification_report(y_test_class, y_test_pred_class))

# Visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Data)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# ================================
# Visualization
# ================================

# 1️⃣ Price Distribution with Mean and Median Annotations
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True, color='blue', ax=ax)
ax.set_title("Distribution of House Prices")
ax.set_xlabel("Price")
ax.set_ylabel("Count")
mean_price = df['price'].mean()
median_price = df['price'].median()
xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.text(xlims[1]*0.6, ylims[1]*0.8, f"Mean: {mean_price:.2f}\nMedian: {median_price:.2f}",
        bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# 2️⃣ Feature vs. Price Scatter Plots with Correlation Coefficients
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.scatterplot(x=df[feature], y=df['price'], ax=axes[row, col], color='red')
    axes[row, col].set_title(f"{feature} vs Price")
    # Compute Pearson correlation coefficient
    corr_value = df[feature].corr(df['price'])
    axes[row, col].text(0.05, 0.95, f"Corr: {corr_value:.2f}", transform=axes[row, col].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()

# 3️⃣ Actual vs. Predicted Prices (Train Data) with Performance Metrics Annotation
plt.figure(figsize=(8, 5))
plt.plot(y_train.values[:100], label="Actual Prices", marker='o')
plt.plot(y_train_pred[:100], label="Predicted Prices", linestyle="dashed", marker='s')
plt.title("Actual vs. Predicted Prices (Train Data) - Random Forest")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.legend()
plt.text(0.05, 0.95, f"Train MAE: {train_mae:.2f}\nTrain RMSE: {train_rmse:.2f}\nTrain R²: {train_r2:.2f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()
