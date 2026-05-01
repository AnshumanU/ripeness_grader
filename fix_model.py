import onnx
import os

base = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base, "models", "banana_model.onnx")

model = onnx.load(model_path)  # removed load_external_data=False so it reads the .data file too
onnx.save_model(model, model_path, save_as_external_data=False)
print("Model fixed and saved!")