from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import gdown

app = Flask(__name__)

# ===== CONFIG =====
MODEL_DIR = "models"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_FILES = {
    "banana": "1Rs9A2qSELBoz7qC2bcL1vqYk0noAHrlG",
}

sessions = {}

# ===== DOWNLOAD MODEL =====
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fruit, fid in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, f"{fruit}.onnx")
        if not os.path.exists(path):
            gdown.download(f"https://drive.google.com/uc?id={fid}", path, quiet=False)

# ===== LOAD MODEL =====
def load_models():
    for fruit in MODEL_FILES.keys():
        path = os.path.join(MODEL_DIR, f"{fruit}.onnx")
        sessions[fruit] = ort.InferenceSession(path)

# ===== PREPROCESS =====
def preprocess(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis, :]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ===== PREDICT =====
def predict(image, fruit):
    session = sessions[fruit]
    logits = session.run(None, {session.get_inputs()[0].name: preprocess(image)})[0][0]
    probs = softmax(logits)

    names = ["overripe", "ripe"] if len(probs) == 2 else ["unripe", "ripe", "overripe"]
    idx = int(np.argmax(probs))

    return {
        "prediction": names[idx],
        "confidence": float(probs[idx]) * 100,
        "all_probs": {names[i]: float(probs[i]) for i in range(len(names))}
    }

# ===== ROUTES =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running 🚀"})

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    fruit = request.form.get("fruit", "banana")

    image = Image.open(file)

    result = predict(image, fruit)

    return jsonify(result)

# ===== START =====
if __name__ == "__main__":
    download_models()
    load_models()
    app.run(host="0.0.0.0", port=10000)