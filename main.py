from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR   = "models"
CLASS_NAMES = ["overripe", "ripe"]
IMG_SIZE    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONFIDENCE_THRESHOLD = 65.0

sessions = {}
if os.path.exists(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            sessions[fruit] = ort.InferenceSession(
                os.path.join(MODEL_DIR, fname),
                providers=["CPUExecutionProvider"]
            )
            print(f"Loaded model: {fruit}")
else:
    print(f"WARNING: models folder not found at {MODEL_DIR}")

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

@app.get("/")
def root():
    return {"message": "Ripeness Grading API is running"}

@app.get("/fruits")
def get_fruits():
    return {"available": list(sessions.keys())}

@app.post("/predict")
async def predict(fruit: str, file: UploadFile = File(...)):

    # Safe content type check
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return {"error": "Uploaded file is not an image. Please upload a JPG or PNG."}

    fruit = fruit.lower()
    if fruit not in sessions:
        return {
            "error": f"No model found for '{fruit}'.",
            "available": list(sessions.keys())
        }

    image_bytes = await file.read()

    # Reject empty or corrupted files
    if len(image_bytes) < 1000:
        return {"error": "Image is too small or corrupted. Please try another photo."}

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}

    try:
        logits = sessions[fruit].run(None, {"image": tensor})[0][0]
    except Exception as e:
        return {"error": f"Model inference failed: {str(e)}"}

    probs      = softmax(logits)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx]) * 100

    # Reject low confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "error": f"Image does not appear to be a {fruit}. "
                     f"Confidence too low ({confidence:.1f}%). "
                     f"Please upload a clear photo of a {fruit}.",
            "confidence": round(confidence, 2),
            "all_probs": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(CLASS_NAMES, probs)
            }
        }

    pred_label = CLASS_NAMES[pred_idx]
    return {
        "fruit":      fruit,
        "grade":      pred_label,
        "confidence": round(confidence, 2),
        "all_probs":  {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASS_NAMES, probs)
        }
    }