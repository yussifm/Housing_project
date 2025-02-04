import joblib
import numpy as np
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load trained model & scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input type
input_features = [("float_input", FloatTensorType([None, 4]))]  # 4 features: sqft, bath, balcony, bhk

# Convert to ONNX format with explicit target opset
onnx_model = convert_sklearn(
    model, 
    initial_types=input_features,
    target_opset=9  # Explicitly set to opset 9
)

# Save ONNX model
with open("house_price_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Optional: Validate the ONNX model
try:
    onnx.checker.check_model(onnx_model)
    print("Model successfully converted to ONNX and validated!")
except onnx.checker.ValidationError as e:
    print(f"Validation error: {e}")

# Additional check for IR version
print(f"ONNX Model IR Version: {onnx_model.ir_version}")