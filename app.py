import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# ---- Page config ----
st.set_page_config(
    page_title="Fruit Ripeness Grader",
    page_icon="🍌",
    layout="centered"
)

# ---- Constants ----
MODEL_DIR   = "models"
CLASS_NAMES = ["overripe", "ripe"]
IMG_SIZE    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 65.0
COLORS = {"ripe": "#2ecc71", "overripe": "#e74c3c", "unripe": "#f39c12"}

# ---- Load models ----
@st.cache_resource
def load_models():
    sessions = {}
    if not os.path.exists(MODEL_DIR):
        return sessions
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            sessions[fruit] = ort.InferenceSession(
                os.path.join(MODEL_DIR, fname),
                providers=["CPUExecutionProvider"]
            )
    return sessions

sessions = load_models()

# ---- Preprocess ----
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def predict(image: Image.Image, fruit: str):
    tensor     = preprocess(image)
    logits     = sessions[fruit].run(None, {"image": tensor})[0][0]
    probs      = softmax(logits)
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx]) * 100
    return CLASS_NAMES[pred_idx], confidence, probs

# ---- UI ----
st.title("Fruit Ripeness Grader")
st.markdown("Detect whether your fruit is **ripe** or **overripe** using AI.")

if not sessions:
    st.error("No models found in the models/ folder. Please add your .onnx files.")
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

# Predict button
if image is not None:
    if st.button("Predict Ripeness", type="primary", use_container_width=True):
        with st.spinner("Analysing..."):
            try:
                grade, confidence, probs = predict(image, fruit)

                if confidence < CONFIDENCE_THRESHOLD:
                    st.error(
                        f"Image does not appear to be a {fruit}. "
                        f"Confidence too low ({confidence:.1f}%). "
                        f"Please upload a clear photo of a {fruit}."
                    )
                else:
                    # Result
                    st.success(f"Prediction complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Grade", grade.capitalize())
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")

                    # Confidence bars
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
                                        background:{color}; transition:width 0.5s'>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")