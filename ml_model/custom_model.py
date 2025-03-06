import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import joblib

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

# Convert 'size' to numeric by extracting the number (e.g., "2 BHK" -> 2)
if 'size' in df.columns:
    df['size'] = df['size'].str.extract(r"(\d+)").astype(float)

# Define features and target
features = ['total_sqft', 'bath', 'balcony', 'size']
target = 'price'

# Ensure dataframe contains only the desired columns
df = df[features + [target]]

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# ================================
# Custom Neural Network Model
# ================================

# Hyperparameters (feel free to modify these)
n_hidden_layers = 3           # Number of hidden layers
neurons_per_layer = 64        # Number of neurons in each hidden layer
dropout_rate = 0.2            # Dropout rate (set to 0 to disable dropout)
learning_rate = 0.001         # Learning rate for the optimizer
epochs = 100                  # Maximum number of training epochs
batch_size = 32               # Batch size
patience = 10                 # Early stopping patience

# Build the model
model = Sequential()
# Input layer
model.add(Dense(neurons_per_layer, activation='relu', input_dim=X_train_scaled.shape[1]))
if dropout_rate > 0:
    model.add(Dropout(dropout_rate))

# Additional hidden layers
for _ in range(n_hidden_layers - 1):
    model.add(Dense(neurons_per_layer, activation='relu'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

# Output layer
model.add(Dense(1))  # Regression output

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

# Callback for early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# ================================
# Evaluation
# ================================

# Predictions
y_train_pred = model.predict(X_train_scaled).flatten()
y_test_pred = model.predict(X_test_scaled).flatten()

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Custom NN Model - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Custom NN Model - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Custom NN Model - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")

# Save the Keras model
model.save("custom_nn_house_price_model.h5")
print("Custom Neural Network model training complete! Saved as 'custom_nn_house_price_model.h5'.")

# ================================
# Additional: Confusion Metrics for Regression
# ================================
# Note: Confusion matrices are for classification. Here we demonstrate an example by converting
# the continuous house prices into binary classes using the median price as a threshold.

# Define binary classes based on the median price
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

# 1️⃣ Price Distribution with Mean and Median annotations
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

# 2️⃣ Feature vs. Price Scatter Plots with correlation coefficients
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.scatterplot(x=df[feature], y=df['price'], ax=axes[row, col], color='red')
    axes[row, col].set_title(f"{feature} vs Price")
    # Compute Pearson correlation
    corr_value = df[feature].corr(df['price'])
    axes[row, col].text(0.05, 0.95, f"Corr: {corr_value:.2f}", transform=axes[row, col].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()

# 3️⃣ Actual vs. Predicted Prices (Train Data) with performance metrics annotation
plt.figure(figsize=(8, 5))
plt.plot(y_train.values[:100], label="Actual Prices", marker='o')
plt.plot(y_train_pred[:100], label="Predicted Prices", linestyle="dashed", marker='s')
plt.title("Actual vs Predicted Prices (Train Data) - Custom NN")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.legend()
plt.text(0.05, 0.95, f"Train MAE: {train_mae:.2f}\nTrain RMSE: {train_rmse:.2f}\nTrain R²: {train_r2:.2f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()
