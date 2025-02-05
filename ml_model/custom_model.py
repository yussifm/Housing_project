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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
n_hidden_layers = 2           # Number of hidden layers
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

print(f"Custom NN Model - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Custom NN Model - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

# Save the Keras model (HDF5 format or SavedModel format)
model.save("custom_nn_house_price_model.h5")
print("Custom Neural Network model training complete! Saved as 'custom_nn_house_price_model.h5'.")

# ================================
# Visualization
# ================================

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

# 3️⃣ Training vs Test Predictions (using a subset for clarity)
plt.figure(figsize=(8, 5))
plt.plot(y_train.values[:100], label="Actual Prices", marker='o')
plt.plot(y_train_pred[:100], label="Predicted Prices", linestyle="dashed", marker='s')
plt.title("Actual vs Predicted Prices (Train Data) - Custom NN")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.legend()
plt.show()
