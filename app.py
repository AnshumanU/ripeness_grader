import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gdown
import os

st.set_page_config(
    page_title="Fruit Ripeness Grader",
    page_icon="🍌",
    layout="centered"
)

MODEL_DIR   = "models"
CLASS_NAMES = ["overripe", "ripe"]
IMG_SIZE    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 60.0
COLORS = {"ripe": "#2ecc71", "overripe": "#e74c3c"}

VALID_FRUITS = {
    "banana": ["banana"],
    "apple":  ["apple"],
    "orange": ["orange"],
}

# ---- Download models from Google Drive ----
@st.cache_resource
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_files = {
        "banana_model.onnx": "1Rs9A2qSELBoz7qC2bcL1vqYk0noAHrlG",
    }

    for filename, file_id in model_files.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            st.info(f"Downloading {filename} from Google Drive...")
            try:
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                output = gdown.download(url, dest, quiet=False, fuzzy=True)
                if output is None:
                    st.error(
                        f"Download returned None — make sure the file is shared "
                        f"as 'Anyone with the link' on Google Drive."
                    )
                elif not os.path.exists(dest):
                    st.error(f"File was not saved to {dest}")
                else:
                    size = os.path.getsize(dest)
                    st.success(f"Downloaded {filename} ({size:,} bytes)")
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
        else:
            size = os.path.getsize(dest)
            st.info(f"Model already cached: {filename} ({size:,} bytes)")

    files = os.listdir(MODEL_DIR)
    st.write(f"Files in models/ folder: {files}")
    return True

# ---- Load ripeness models ----
@st.cache_resource
def load_ripeness_models(_download_done):
    sessions = {}
    if not os.path.exists(MODEL_DIR):
        st.error(f"models/ folder does not exist")
        return sessions

    onnx_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
    if not onnx_files:
        st.error(f"No .onnx files found in models/ folder")
        return sessions

    for fname in onnx_files:
        fruit = fname.replace("_model.onnx", "")
        fpath = os.path.join(MODEL_DIR, fname)
        try:
            sessions[fruit] = ort.InferenceSession(
                fpath,
                providers=["CPUExecutionProvider"]
            )
            st.success(f"Loaded model: {fruit}")
        except Exception as e:
            st.error(f"Failed to load {fname}: {str(e)}")

    return sessions

# ---- Load YOLO detector ----
@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")

# ---- Run startup ----
download_done = download_models()
sessions      = load_ripeness_models(download_done)
detector      = load_detector()

# ---- Helper functions ----
def detect_fruit(image: Image.Image, fruit: str):
    results  = detector(image, verbose=False)
    detected = []
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls)].lower()
            detected.append(cls_name)
    valid_labels = VALID_FRUITS.get(fruit, [fruit])
    for label in detected:
        if any(v in label for v in valid_labels):
            return True, detected
    return False, detected

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def predict_ripeness(image: Image.Image, fruit: str):
    tensor     = preprocess(image)
    logits     = sessions[fruit].run(None, {"image": tensor})[0][0]
    probs      = softmax(logits)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx]) * 100
    return CLASS_NAMES[pred_idx], confidence, probs

# ---- UI ----
st.title("🍌 Fruit Ripeness Grader")
st.markdown("Upload or capture a fruit image to detect its ripeness using AI.")

if not sessions:
    st.error(
        "No models loaded. Check the errors above — "
        "make sure your Google Drive file is shared as 'Anyone with the link'."
    )
    st.stop()

# Fruit selector
fruit = st.selectbox(
    "Select fruit",
    options=list(sessions.keys()),
    format_func=lambda x: x.capitalize()
)

# Input mode
mode = st.radio("Input method", ["Upload Image", "Use Camera"], horizontal=True)

image = None

if mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a fruit image", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

elif mode == "Use Camera":
    camera_photo = st.camera_input("Take a photo of the fruit")
    if camera_photo:
        image = Image.open(camera_photo)
        st.image(image, caption="Captured image", use_container_width=True)

# ---- Predict ----
if image is not None:
    if st.button("Predict Ripeness", type="primary", use_container_width=True):

        with st.spinner("Detecting fruit in image..."):
            fruit_found, detected_labels = detect_fruit(image, fruit)

        if not fruit_found:
            if detected_labels:
                st.error(
                    f"No {fruit} detected in the image. "
                    f"Detected instead: {', '.join(set(detected_labels))}. "
                    f"Please upload a clear photo of a {fruit}."
                )
            else:
                st.error(
                    f"No fruit detected in the image at all. "
                    f"Please upload a clear photo of a {fruit}."
                )
        else:
            with st.spinner("Analysing ripeness..."):
                grade, confidence, probs = predict_ripeness(image, fruit)

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"{fruit.capitalize()} detected but ripeness is unclear. "
                    f"Confidence: {confidence:.1f}%. "
                    f"Try a clearer or better lit photo."
                )
            else:
                st.success("Prediction complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Grade", grade.capitalize())
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")

                st.markdown("#### Confidence breakdown")
                for cls, prob in zip(CLASS_NAMES, probs):
                    pct = float(prob) * 100
                    color = COLORS.get(cls, "#888")
                    st.markdown(f"""
                    <div style='margin-bottom:10px'>
                      <div style='display:flex; justify-content:space-between;
                                  font-size:14px; margin-bottom:4px'>
                        <span style='text-transform:capitalize'>{cls}</span>
                        <span>{pct:.1f}%</span>
                      </div>
                      <div style='background:#f0f0f0; border-radius:4px; height:12px'>
                        <div style='width:{pct}%; height:100%; border-radius:4px;
                                    background:{color}'>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.caption(
                    f"Detected: {', '.join(set(detected_labels))} — "
                    f"Grade: {grade} — "
                    f"Confidence: {confidence:.1f}%"
                )