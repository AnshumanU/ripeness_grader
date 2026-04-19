import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(
    page_title="Fruit Ripeness Grader",
    page_icon="🍎",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 60.0

COLORS = {
    "ripe": "#2ecc71",
    "overripe": "#e74c3c",
    "unripe": "#f39c12"
}

MODEL_FILES = {
    "banana": "1Rs9A2qSELBoz7qC2bcL1vqYk0noAHrlG",
}

def validate_fruit(image: Image.Image, fruit: str):
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)

    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()

    avg_std = np.array([arr[:, :, i].std() for i in range(3)]).mean()

    if avg_std < 15:
        return False, "Image appears blank or too uniform."

    checks = {
        "banana": (r > 120 and g > 100 and b < 120) or (r < 80 and g < 70),
        "apple": (r > 130 and r > g * 1.2) or (g > 100 and g > r * 0.9),
        "mango": (r > 150 and g > 100 and b < 100),
        "orange": (r > 150 and g > 80 and b < 100),
        "grape": (r > 80 and b > 60) or (r < 120 and g < 100 and b < 120),
        "strawberry": (r > 150 and r > g * 1.5),
        "tomato": (r > 140 and r > g * 1.3),
        "avocado": (g > 80 and r < 180),
        "peach": (r > 180 and g > 120 and b > 80),
        "pear": (g > 100 and r > 100),
        "kiwi": (g > 90 and r < 160),
    }

    passed = checks.get(fruit, True)

    if not passed:
        return False, (
            f"Colors (R:{r:.0f} G:{g:.0f} B:{b:.0f}) don't match a {fruit}. "
            f"Please upload a clear {fruit} photo."
        )

    return True, "OK"

@st.cache_resource
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for fruit, file_id in MODEL_FILES.items():
        dest = os.path.join(MODEL_DIR, f"{fruit}_model.onnx")

        if not os.path.exists(dest):
            try:
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                gdown.download(url, dest, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download error for {fruit}: {e}")

    return True

@st.cache_resource
def load_models(_done):
    sessions = {}

    if not os.path.exists(MODEL_DIR):
        return sessions

    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            path = os.path.join(MODEL_DIR, fname)

            try:
                sessions[fruit] = ort.InferenceSession(
                    path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            except Exception as e:
                st.error(f"Could not load {fname}: {e}")

    return sessions

def preprocess(image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_ripeness(image: Image.Image, fruit: str, session):
    tensor = preprocess(image)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits)

    num_classes = len(probs)

    if num_classes == 2:
        class_names = ["overripe", "ripe"]
    else:
        class_names = ["unripe", "ripe", "overripe"]

    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return predicted_class, confidence, probs, class_names

done = download_models()
sessions = load_models(done)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
    }

    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px;
        height: 100%;
    }

    .result-card {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        padding: 20px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-top: 20px;
        font-size: 18px;
        font-weight: 600;
    }

    .warning-card {
        background: linear-gradient(135deg, #dc2626, #f97316);
        padding: 18px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-top: 20px;
        font-size: 16px;
        font-weight: 500;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 16px;
        color: white;
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: -5px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 14px;
        font-size: 16px;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="font-size:48px; margin-bottom:8px;">🍎 Fruit Ripeness Grader</h1>
        <p class="subtitle">
            Upload or capture a fruit image and detect its ripeness using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not sessions:
    st.error("No models loaded. Please check your model files.")
    st.stop()

left_col, right_col = st.columns([1, 1.1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Fruit Selection</div>', unsafe_allow_html=True)

    fruit = st.selectbox(
        "Select Fruit",
        options=sorted(sessions.keys()),
        format_func=lambda x: x.capitalize()
    )

    mode = st.radio(
        "Choose Input Method",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    image = None

    if mode == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a fruit image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded)

    else:
        photo = st.camera_input("Take a photo of the fruit")

        if photo:
            image = Image.open(photo)

    predict_btn = st.button("Predict Ripeness", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Preview & Result</div>', unsafe_allow_html=True)

    if image is not None:
        st.image(image, caption="Selected Fruit", use_container_width=True)

        if predict_btn:
            with st.spinner("Checking image..."):
                is_valid, message = validate_fruit(image, fruit)

            if not is_valid:
                st.markdown(
                    f"""
                    <div class="warning-card">
                        ⚠️ {message}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                with st.spinner("Analyzing ripeness..."):
                    grade, confidence, probs, class_names = predict_ripeness(
                        image,
                        fruit,
                        sessions[fruit]
                    )

                if confidence < CONFIDENCE_THRESHOLD:
                    st.markdown(
                        f"""
                        <div class="warning-card">
                            Detected {fruit}, but confidence is low ({confidence:.1f}%).
                            Try a clearer image.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            {fruit.capitalize()} is <b>{grade.capitalize()}</b><br>
                            Confidence: {confidence:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(confidence / 100)

                    st.markdown("### Confidence Breakdown")

                    for cls, prob in zip(class_names, probs):
                        pct = float(prob) * 100
                        color = COLORS.get(cls, "#888888")

                        st.markdown(
                            f'''
                            <div style="margin-bottom:14px;">
                                <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:6px;">
                                    <span style="text-transform:capitalize;">{cls}</span>
                                    <span>{pct:.1f}%</span>
                                </div>
                                <div style="background:#334155;border-radius:999px;height:12px;overflow:hidden;">
                                    <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;"></div>
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

    else:
        st.info("Upload or capture a fruit image to preview it here.")

    st.markdown('</div>', unsafe_allow_html=True)