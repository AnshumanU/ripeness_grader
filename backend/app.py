from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

app = Flask(__name__)

# ===== CORS =====
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://ripeness-grader.vercel.app",  # ← update with your Vercel URL
    "*"  # remove this in production
])

# ===== CONFIG =====
MODEL_DIR = "models"
IMG_SIZE  = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ===== LOAD ALL MODELS FROM models/ FOLDER =====
sessions = {}

def load_models():
    if not os.path.exists(MODEL_DIR):
        print(f"WARNING: models folder not found at '{MODEL_DIR}'")
        return
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "").replace(".onnx", "")
            path  = os.path.join(MODEL_DIR, fname)
            sessions[fruit] = ort.InferenceSession(path)
            print(f"✅ Loaded model: {fruit}")

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
    input_name = session.get_inputs()[0].name  # auto-detect input name
    logits = session.run(None, {input_name: preprocess(image)})[0][0]
    probs  = softmax(logits)

    names = ["overripe", "ripe"] if len(probs) == 2 else ["unripe", "ripe", "overripe"]
    idx   = int(np.argmax(probs))

    return {
        "fruit":      fruit,
        "grade":      names[idx],           # ← changed from "prediction" to "grade"
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probs":  {names[i]: round(float(probs[i]) * 100, 2) for i in range(len(names))}
    }

# ===== ROUTES =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":        "FruitSense API running 🚀",
        "models_loaded": list(sessions.keys())
    })

@app.route("/fruits", methods=["GET"])
def get_fruits():
    return jsonify({"available": list(sessions.keys())})

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file  = request.files["file"]
    fruit = request.args.get("fruit") or request.form.get("fruit", "banana")
    fruit = fruit.lower().strip()

    if fruit not in sessions:
        return jsonify({
            "error":     f"No model found for '{fruit}'.",
            "available": list(sessions.keys())
        }), 400

    try:
        image  = Image.open(file)
        result = predict(image, fruit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== START =====
load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)