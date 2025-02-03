import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the ONNX model
onnx_model = onnx.load("house_price_model.onnx")

# Convert the ONNX model to a TensorFlow model
tf_rep = prepare(onnx_model)
tf_rep.export_graph("house_price_model.pb")

# Convert the TensorFlow model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("house_price_model.pb")
tflite_model = converter.convert()

# Save the TFLite model to file
with open("house_price_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model conversion to TFLite successful!")
