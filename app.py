import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
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
            st.info(f"Downloading {filename}...")
            try:
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                output = gdown.download(url, dest, quiet=False, fuzzy=True)
                if output is None or not os.path.exists(dest):
                    st.error("Download failed. Check Google Drive sharing settings.")
                else:
                    st.success(f"Downloaded {filename} ({os.path.getsize(dest):,} bytes)")
            except Exception as e:
                st.error(f"Download error: {str(e)}")
        else:
            st.info(f"Model cached: {filename} ({os.path.getsize(dest):,} bytes)")
    return True

# ---- Load ripeness models ----
@st.cache_resource
def load_ripeness_models(_done):
    sessions = {}
    if not os.path.exists(MODEL_DIR):
        return sessions
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            fpath = os.path.join(MODEL_DIR, fname)
            try:
                sessions[fruit] = ort.InferenceSession(
                    fpath, providers=["CPUExecutionProvider"]
                )
                st.success(f"Loaded model: {fruit}")
            except Exception as e:
                st.error(f"Failed to load {fname}: {str(e)}")
    return sessions

# ---- Fruit validator (color based, no YOLO needed) ----
def validate_fruit(image: Image.Image, fruit: str) -> tuple[bool, str]:
    """
    Simple color based validator.
    Checks if the image contains colors consistent with the selected fruit.
    """
    img   = image.convert("RGB").resize((100, 100))
    arr   = np.array(img, dtype=np.float32)
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    # Check if image is too gray (likely not a fruit)
    r_std = arr[:, :, 0].std()
    g_std = arr[:, :, 1].std()
    b_std = arr[:, :, 2].std()
    avg_std = (r_std + g_std + b_std) / 3

    if avg_std < 15:
        return False, "Image appears to be blank or too uniform."

    if fruit == "banana":
        # Bananas are yellow: high R, high G, low B
        is_yellowish = (r_mean > 120) and (g_mean > 100) and (b_mean < 120)
        # Or dark brown/black for overripe
        is_dark = (r_mean < 80) and (g_mean < 70) and (b_mean < 60)
        if not (is_yellowish or is_dark):
            return False, (
                f"Image colors (R:{r_mean:.0f} G:{g_mean:.0f} B:{b_mean:.0f}) "
                f"don't match a banana. Please upload a clear banana photo."
            )

    elif fruit == "apple":
        # Apples are red or green: high R or high G
        is_reddish = (r_mean > 130) and (r_mean > g_mean * 1.2)
        is_greenish = (g_mean > 100) and (g_mean > r_mean * 0.9)
        if not (is_reddish or is_greenish):
            return False, (
                f"Image colors don't match an apple. "
                f"Please upload a clear apple photo."
            )

    elif fruit == "orange":
        # Oranges are orange: high R, medium G, low B
        is_orangish = (r_mean > 150) and (g_mean > 80) and (b_mean < 100)
        if not is_orangish:
            return False, (
                f"Image colors don't match an orange. "
                f"Please upload a clear orange photo."
            )

    return True, "Fruit detected."

# ---- Preprocessing ----
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

# ---- Startup ----
done     = download_models()
sessions = load_ripeness_models(done)

# ---- UI ----
st.title("🍌 Fruit Ripeness Grader")
st.markdown("Upload or capture a fruit image to detect its ripeness using AI.")

if not sessions:
    st.error(
        "No models loaded. Check errors above and make sure "
        "your Google Drive file is shared as 'Anyone with the link'."
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

# ---- Predict button ----
if image is not None:
    if st.button("Predict Ripeness", type="primary", use_container_width=True):

        # Step 1 — validate fruit
        with st.spinner("Checking image..."):
            is_valid, message = validate_fruit(image, fruit)

        if not is_valid:
            st.error(f"Invalid image: {message}")

        else:
            # Step 2 — predict ripeness
            with st.spinner("Analysing ripeness..."):
                grade, confidence, probs = predict_ripeness(image, fruit)

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"{fruit.capitalize()} detected but ripeness unclear. "
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
                    pct   = float(prob) * 100
                    color = COLORS.get(cls, "#888")
                    st.markdown(f"""
                    <div style='margin-bottom:10px'>
                      <div style='display:flex;justify-content:space-between;
                                  font-size:14px;margin-bottom:4px'>
                        <span style='text-transform:capitalize'>{cls}</span>
                        <span>{pct:.1f}%</span>
                      </div>
                      <div style='background:#f0f0f0;border-radius:4px;height:12px'>
                        <div style='width:{pct}%;height:100%;border-radius:4px;
                                    background:{color}'>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.caption(f"Fruit: {fruit} — Grade: {grade} — Confidence: {confidence:.1f}%")