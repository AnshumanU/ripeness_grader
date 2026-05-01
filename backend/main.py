from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, os, uuid, time
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
import json

# ── Optional JWT auth (install: pip install python-jose[cryptography] passlib[bcrypt]) ──
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("WARNING: Install python-jose and passlib for user auth features")

app = FastAPI(title="FruitSense AI API", version="2.0.0")

# ── CORS — update these origins for your Vercel domain ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Constants ───────────────────────────────────────────────────────────────────
MODEL_DIR            = "models"
IMG_SIZE             = 224
MEAN                 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD                  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 65.0

# Per-fruit class names (extend as you add more models)
FRUIT_CLASSES = {
    "banana":     ["overripe", "ripe"],
    "apple":      ["unripe", "ripe", "overripe"],
    "mango":      ["unripe", "ripe", "overripe"],
    "orange":     ["unripe", "ripe", "overripe"],
    "tomato":     ["unripe", "ripe", "overripe"],
    "strawberry": ["unripe", "ripe", "overripe"],
    "grape":      ["unripe", "ripe", "overripe"],
    "peach":      ["unripe", "ripe", "overripe"],
    "kiwi":       ["unripe", "ripe", "overripe"],
    "pear":       ["unripe", "ripe", "overripe"],
    "avocado":    ["unripe", "ripe", "overripe"],
}

# ── Auth config (change SECRET_KEY in production!) ─────────────────────────────
SECRET_KEY    = os.environ.get("SECRET_KEY", "change-me-in-production-use-env-var")
ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ── Simple in-memory stores (replace with a real DB like PostgreSQL/SQLite) ────
# Structure: { user_id: { "email": ..., "hashed_password": ..., "name": ... } }
users_db: dict = {}
# Structure: { user_id: [ scan_record, ... ] }
history_db: dict = {}

# ── Load ONNX models ────────────────────────────────────────────────────────────
sessions: dict = {}
if os.path.exists(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            sessions[fruit] = ort.InferenceSession(
                os.path.join(MODEL_DIR, fname),
                providers=["CPUExecutionProvider"]
            )
            print(f"✅ Loaded model: {fruit}")
else:
    print(f"⚠️  WARNING: models folder not found at '{MODEL_DIR}'")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

if AUTH_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    security    = HTTPBearer(auto_error=False)

    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)

    def create_access_token(data: dict) -> str:
        payload = data.copy()
        payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[str]:
        """Returns user_id if token is valid, else None (anonymous allowed)."""
        if not credentials:
            return None
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub")
        except JWTError:
            return None
else:
    # Stubs so the rest of the code doesn't break when jose/passlib are missing
    security = HTTPBearer(auto_error=False)
    def get_current_user(credentials=None) -> Optional[str]:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


# ═══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)

def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def run_inference(fruit: str, image_bytes: bytes) -> dict:
    """Core inference logic shared by /predict and /detect."""
    class_names = FRUIT_CLASSES.get(fruit, ["overripe", "ripe"])

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

    all_probs = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(class_names, probs)
    }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "error": (
                f"Image does not appear to be a {fruit}. "
                f"Confidence too low ({confidence:.1f}%). "
                f"Please upload a clear photo of a {fruit}."
            ),
            "confidence": round(confidence, 2),
            "all_probs":  all_probs,
        }

    return {
        "fruit":      fruit,
        "grade":      class_names[pred_idx],
        "confidence": round(confidence, 2),
        "all_probs":  all_probs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "FruitSense AI API v2.0 is running",
        "endpoints": ["/fruits", "/predict", "/detect", "/register", "/login", "/history"],
        "models_loaded": list(sessions.keys()),
    }

@app.get("/fruits")
def get_fruits():
    return {"available": list(sessions.keys())}


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — manual fruit selection (original endpoint, unchanged behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(
    fruit: str,
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(get_current_user),
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return {"error": "Uploaded file is not an image. Please upload a JPG or PNG."}

    fruit = fruit.lower()
    if fruit not in sessions:
        return {"error": f"No model found for '{fruit}'.", "available": list(sessions.keys())}

    image_bytes = await file.read()
    if len(image_bytes) < 1000:
        return {"error": "Image is too small or corrupted. Please try another photo."}

    result = run_inference(fruit, image_bytes)

    # Save to history if user is logged in
    if user_id and not result.get("error"):
        _save_history(user_id, result)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT — no fruit dropdown needed
# Tries every loaded model and returns the best confident match.
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/detect")
async def auto_detect(
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Automatically detects the fruit type AND ripeness grade.
    No fruit parameter required — the frontend just uploads the image.

    Strategy:
      1. Run the image through every loaded ONNX model.
      2. Keep results whose confidence ≥ CONFIDENCE_THRESHOLD.
      3. Return the highest-confidence match.
      4. If nothing clears the threshold, return the best guess with a warning.
    """
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return {"error": "Uploaded file is not an image. Please upload a JPG or PNG."}

    if not sessions:
        return {"error": "No models are loaded. Add .onnx files to the models/ folder."}

    image_bytes = await file.read()
    if len(image_bytes) < 1000:
        return {"error": "Image is too small or corrupted. Please try another photo."}

    candidates = []
    all_results = {}

    for fruit_name, session in sessions.items():
        class_names = FRUIT_CLASSES.get(fruit_name, ["overripe", "ripe"])
        try:
            tensor     = preprocess(image_bytes)
            logits     = session.run(None, {"image": tensor})[0][0]
            probs      = softmax(logits)
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100
            all_results[fruit_name] = {
                "grade":      class_names[pred_idx],
                "confidence": round(confidence, 2),
                "all_probs":  {c: round(float(p)*100, 2) for c, p in zip(class_names, probs)},
            }
            if confidence >= CONFIDENCE_THRESHOLD:
                candidates.append((fruit_name, confidence, class_names[pred_idx], all_results[fruit_name]["all_probs"]))
        except Exception:
            continue  # skip broken models silently

    if not candidates:
        # Return best guess even below threshold
        best = max(all_results.items(), key=lambda x: x[1]["confidence"])
        return {
            "warning": "No fruit detected with high confidence. Showing best guess.",
            "fruit":      best[0],
            "grade":      best[1]["grade"],
            "confidence": best[1]["confidence"],
            "all_probs":  best[1]["all_probs"],
            "auto_detected": True,
        }

    # Pick the winner
    winner_fruit, winner_conf, winner_grade, winner_probs = max(candidates, key=lambda x: x[1])

    result = {
        "fruit":         winner_fruit,
        "grade":         winner_grade,
        "confidence":    round(winner_conf, 2),
        "all_probs":     winner_probs,
        "auto_detected": True,
        "all_scores":    {f: all_results[f]["confidence"] for f in all_results},
    }

    if user_id:
        _save_history(user_id, result)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# USER AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/register")
def register(req: RegisterRequest):
    if not AUTH_AVAILABLE:
        return {"error": "Auth not available. Run: pip install python-jose[cryptography] passlib[bcrypt]"}

    email = req.email.lower().strip()
    if any(u["email"] == email for u in users_db.values()):
        raise HTTPException(status_code=400, detail="Email already registered.")

    user_id = str(uuid.uuid4())
    users_db[user_id] = {
        "id":              user_id,
        "name":            req.name,
        "email":           email,
        "hashed_password": hash_password(req.password),
        "created_at":      datetime.utcnow().isoformat(),
    }
    history_db[user_id] = []

    token = create_access_token({"sub": user_id})
    return {
        "token":   token,
        "user": {"id": user_id, "name": req.name, "email": email},
    }


@app.post("/login")
def login(req: LoginRequest):
    if not AUTH_AVAILABLE:
        return {"error": "Auth not available. Run: pip install python-jose[cryptography] passlib[bcrypt]"}

    email = req.email.lower().strip()
    user  = next((u for u in users_db.values() if u["email"] == email), None)

    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token({"sub": user["id"]})
    return {
        "token": token,
        "user":  {"id": user["id"], "name": user["name"], "email": user["email"]},
    }


@app.get("/me")
def get_me(user_id: Optional[str] = Depends(get_current_user)):
    if not user_id or user_id not in users_db:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    u = users_db[user_id]
    return {"id": u["id"], "name": u["name"], "email": u["email"]}


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def _save_history(user_id: str, result: dict):
    if user_id not in history_db:
        history_db[user_id] = []
    history_db[user_id].insert(0, {
        **result,
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    })
    # Keep last 200 scans per user
    history_db[user_id] = history_db[user_id][:200]


@app.get("/history")
def get_history(
    limit: int = 20,
    user_id: Optional[str] = Depends(get_current_user),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login to view scan history.")
    return {
        "scans": history_db.get(user_id, [])[:limit],
        "total": len(history_db.get(user_id, [])),
    }


@app.delete("/history")
def clear_history(user_id: Optional[str] = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required.")
    history_db[user_id] = []
    return {"message": "Scan history cleared."}
