import joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load trained model & scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input type
input_features = [("float_input", FloatTensorType([None, 4]))]  # 4 features: sqft, bath, balcony, bhk

# Convert to ONNX format
onnx_model = convert_sklearn(model, initial_types=input_features)

# Save ONNX model
with open("house_price_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX!")
