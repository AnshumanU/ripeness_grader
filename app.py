import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import gdown
import os
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FruitSense · Ripeness AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 60.0

COLORS = {
    "ripe":     "#5cb85c",
    "overripe": "#d9534f",
    "unripe":   "#e6a817",
}

MODEL_FILES = {
    "banana": "1Rs9A2qSELBoz7qC2bcL1vqYk0noAHrlG",
}

# ── Fruit info database ────────────────────────────────────────────────────────
FRUIT_INFO = {
    "banana": {
        "emoji": "🍌",
        "unripe":   {"msg": "Leave at room temperature for 2–3 days to ripen.",        "tip": "Place near other ripe fruits to speed up ripening.",         "icon": "⏳"},
        "ripe":     {"msg": "Best time to eat! Stores 2–3 days at room temperature.",  "tip": "Refrigerate to slow further ripening; skin may darken.",     "icon": "✅"},
        "overripe": {"msg": "Use for smoothies or baking. Don't store further.",       "tip": "Freeze peeled bananas for future smoothies or banana bread.", "icon": "🧁"},
    },
    "apple": {
        "emoji": "🍎",
        "unripe":   {"msg": "Needs more time. Store at room temp for a few days.",     "tip": "Avoid refrigerating — cold slows ripening significantly.",   "icon": "⏳"},
        "ripe":     {"msg": "Crisp and ready! Refrigerate to keep fresh up to 2 weeks.","tip": "Store away from other produce — apples release ethylene.",  "icon": "✅"},
        "overripe": {"msg": "Best for sauces, pies, or juicing.",                      "tip": "Cut away soft spots; the rest is still flavourful.",         "icon": "🥧"},
    },
    "mango": {
        "emoji": "🥭",
        "unripe":   {"msg": "Leave at room temp for 2–5 days until it yields to touch.","tip": "Great for pickles or raw chutneys at this stage.",          "icon": "⏳"},
        "ripe":     {"msg": "Perfect sweetness! Refrigerate and consume within 3 days.","tip": "A ripe mango smells fruity near the stem end.",             "icon": "✅"},
        "overripe": {"msg": "Use immediately for smoothies, lassi, or desserts.",      "tip": "Freeze pulp in bags for later use in shakes.",              "icon": "🥤"},
    },
    "orange": {
        "emoji": "🍊",
        "unripe":   {"msg": "Too tart now. Allow 3–5 more days at room temperature.",  "tip": "Colour alone doesn't indicate ripeness — feel the weight.",  "icon": "⏳"},
        "ripe":     {"msg": "Juicy and ready! Keep refrigerated for up to 2 weeks.",   "tip": "Roll on a counter before cutting to maximise juice yield.",  "icon": "✅"},
        "overripe": {"msg": "Juice it now before it dries out. Avoid eating raw.",     "tip": "Zest the peel before juicing — it stores well frozen.",      "icon": "🍹"},
    },
    "tomato": {
        "emoji": "🍅",
        "unripe":   {"msg": "Store stem-down at room temp; ripe in 3–7 days.",         "tip": "Never refrigerate unripe tomatoes — it kills the flavour.",  "icon": "⏳"},
        "ripe":     {"msg": "Peak flavour! Use within 2 days or refrigerate.",         "tip": "Room temp always tastes better than fridge-cold.",          "icon": "✅"},
        "overripe": {"msg": "Ideal for sauces, soups, or roasting.",                   "tip": "Roast with olive oil and garlic for a quick pasta sauce.",   "icon": "🍝"},
    },
    "strawberry": {
        "emoji": "🍓",
        "unripe":   {"msg": "White/pink — needs 1–2 more days.",                       "tip": "Strawberries don't ripen further once refrigerated.",        "icon": "⏳"},
        "ripe":     {"msg": "Eat today or tomorrow for best flavour!",                 "tip": "Store unwashed in the fridge; wash just before eating.",    "icon": "✅"},
        "overripe": {"msg": "Use in jam, smoothies, or dessert toppings right away.",  "tip": "Blend with yogurt and honey for an instant compote.",       "icon": "🍰"},
    },
    "avocado": {
        "emoji": "🥑",
        "unripe":   {"msg": "Hard — leave at room temp for 3–5 days.",                 "tip": "Put in a paper bag with a banana to ripen faster.",         "icon": "⏳"},
        "ripe":     {"msg": "Yields to gentle pressure — eat within 24 hours!",        "tip": "Refrigerate a ripe avocado to buy 1–2 extra days.",         "icon": "✅"},
        "overripe": {"msg": "Check inside — brown flesh? Compost. Slightly dark? Guac!","tip": "Lemon juice slows browning once cut.",                     "icon": "🥗"},
    },
    "grape": {
        "emoji": "🍇",
        "unripe":   {"msg": "Very tart — wait 2–4 more days on the vine.",             "tip": "Clusters ripen unevenly; taste a few before picking all.",  "icon": "⏳"},
        "ripe":     {"msg": "Sweet and juicy! Refrigerate and eat within 1 week.",     "tip": "Rinse only just before eating to prevent mould.",           "icon": "✅"},
        "overripe": {"msg": "Wrinkled — still edible but best for juicing or raisins.","tip": "Dehydrate in oven at low heat to make your own raisins.",   "icon": "🍷"},
    },
    "peach": {
        "emoji": "🍑",
        "unripe":   {"msg": "Firm — leave at room temp for 2–3 days.",                 "tip": "Check near the stem for a sweet smell as it ripens.",       "icon": "⏳"},
        "ripe":     {"msg": "Fragrant and soft — eat today or refrigerate.",           "tip": "Store in a single layer to avoid bruising.",                "icon": "✅"},
        "overripe": {"msg": "Very soft — blend into smoothies or bake into a crumble.","tip": "Skin peels easily at this stage — great for jams.",        "icon": "🫙"},
    },
    "kiwi": {
        "emoji": "🥝",
        "unripe":   {"msg": "Rock-hard — ripen 3–5 days at room temperature.",         "tip": "Store next to apples or bananas to speed it up.",           "icon": "⏳"},
        "ripe":     {"msg": "Yields to thumb pressure — eat within 2 days!",           "tip": "Once ripe, move to the fridge to extend life by a week.",   "icon": "✅"},
        "overripe": {"msg": "Very mushy — best blended into juices or smoothies.",     "tip": "The flavour is still great; texture just won't hold slices.","icon": "🥤"},
    },
    "pear": {
        "emoji": "🍐",
        "unripe":   {"msg": "Firm — ripen at room temp, checking daily.",              "tip": "Pears ripen from the inside out — check near the neck.",    "icon": "⏳"},
        "ripe":     {"msg": "Gentle give near the stem — eat now or refrigerate.",     "tip": "Unlike most fruits, pears ripen best off the tree.",        "icon": "✅"},
        "overripe": {"msg": "Grainy texture — use in smoothies, poaching, or sauce.",  "tip": "Poached in spiced red wine for an elegant dessert.",        "icon": "🍮"},
    },
}

DEFAULT_FRUIT_INFO = {
    "unripe":   {"msg": "Not ready yet. Leave at room temperature to ripen.",          "tip": "Check daily and avoid refrigerating during ripening.",      "icon": "⏳"},
    "ripe":     {"msg": "Ready to eat! Best consumed within a few days.",              "tip": "Refrigerate to slow further ripening.",                     "icon": "✅"},
    "overripe": {"msg": "Past peak — use soon in cooking, smoothies, or baking.",     "tip": "Don't discard — overripe fruit is flavour-packed!",        "icon": "🧑‍🍳"},
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0c1a0e;
  --bg2:       #111f14;
  --bg3:       #162019;
  --border:    #1f3325;
  --green:     #5cb85c;
  --amber:     #e6a817;
  --red:       #d9534f;
  --text:      #dde8d5;
  --muted:     #6b8f6b;
  --accent:    #a8d878;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1160px !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Hero ── */
.hero {
  display: flex; align-items: center; gap: 1.2rem;
  padding: 2rem 2.5rem; margin-bottom: 2rem;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 20px;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(92,184,92,0.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-emoji { font-size: 3.2rem; filter: drop-shadow(0 0 20px rgba(168,216,120,0.5)); }
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem; font-weight: 700;
  color: var(--accent); line-height: 1; margin: 0;
}
.hero-sub {
  font-size: 0.82rem; color: var(--muted);
  letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.3rem;
}

/* ── Cards ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.6rem;
  margin-bottom: 1.2rem;
}
.card-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.15rem; color: var(--accent);
  margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
}

/* ── Result badge ── */
.result-badge {
  display: inline-block;
  padding: 0.35rem 1rem;
  border-radius: 999px;
  font-size: 0.8rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-ripe     { background: rgba(92,184,92,0.18);  color: #5cb85c;  border: 1px solid rgba(92,184,92,0.4);  }
.badge-overripe { background: rgba(217,83,79,0.18);  color: #e87370;  border: 1px solid rgba(217,83,79,0.4);  }
.badge-unripe   { background: rgba(230,168,23,0.18); color: #e6a817;  border: 1px solid rgba(230,168,23,0.4); }

/* ── Animated confidence bar ── */
.conf-wrap { margin-bottom: 1rem; }
.conf-label {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; color: var(--muted); margin-bottom: 0.4rem;
  text-transform: capitalize;
}
.conf-track {
  background: rgba(255,255,255,0.06);
  border-radius: 999px; height: 10px; overflow: hidden;
}
.conf-fill {
  height: 100%; border-radius: 999px;
  animation: fillBar 1s cubic-bezier(0.4,0,0.2,1) forwards;
}
@keyframes fillBar {
  from { width: 0%; }
  to   { width: var(--target); }
}

/* ── Ripeness timeline ── */
.timeline {
  display: flex; align-items: center;
  gap: 0; margin: 0.5rem 0 1.2rem;
}
.tl-stage {
  flex: 1; text-align: center;
  padding: 0.6rem 0.3rem;
  border-radius: 10px;
  font-size: 0.78rem; font-weight: 500;
  color: var(--muted);
  border: 1px solid transparent;
  transition: all 0.3s;
}
.tl-stage.active-ripe     { background: rgba(92,184,92,0.18);  color: #5cb85c;  border-color: rgba(92,184,92,0.5);  box-shadow: 0 0 14px rgba(92,184,92,0.2); }
.tl-stage.active-overripe { background: rgba(217,83,79,0.18);  color: #e87370;  border-color: rgba(217,83,79,0.5);  box-shadow: 0 0 14px rgba(217,83,79,0.2); }
.tl-stage.active-unripe   { background: rgba(230,168,23,0.18); color: #e6a817;  border-color: rgba(230,168,23,0.5); box-shadow: 0 0 14px rgba(230,168,23,0.2); }
.tl-arrow { color: var(--border); font-size: 1rem; padding: 0 0.2rem; flex-shrink: 0; }

/* ── Info card ── */
.info-card {
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-top: 0.8rem;
}
.info-card.ripe     { background: rgba(92,184,92,0.1);  border: 1px solid rgba(92,184,92,0.3);  }
.info-card.overripe { background: rgba(217,83,79,0.1);  border: 1px solid rgba(217,83,79,0.3);  }
.info-card.unripe   { background: rgba(230,168,23,0.1); border: 1px solid rgba(230,168,23,0.3); }
.info-main { font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem; }
.info-tip  { font-size: 0.82rem; color: var(--muted); font-style: italic; }

/* ── History table ── */
.hist-row {
  display: grid;
  grid-template-columns: 2fr 2fr 1.5fr 2fr;
  gap: 0.5rem;
  padding: 0.7rem 0.5rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.84rem;
  align-items: center;
}
.hist-row:last-child { border-bottom: none; }
.hist-header {
  color: var(--muted); font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.1em;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.hist-conf-bar {
  height: 6px; border-radius: 999px;
  display: inline-block; vertical-align: middle;
  margin-right: 6px;
}

/* ── Buttons ── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #2d5a30, #3d7a40) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(92,184,92,0.4) !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.5rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important; font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #3d7a40, #4d9a50) !important;
  box-shadow: 0 4px 20px rgba(92,184,92,0.25) !important;
  transform: translateY(-1px) !important;
}

/* ── Selectbox / radio ── */
.stSelectbox > div > div, .stRadio > div {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
label, .stRadio label { color: var(--text) !important; font-size: 0.88rem !important; }
.stFileUploader > div {
  border: 2px dashed var(--border) !important;
  border-radius: 14px !important;
  background: var(--bg3) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green) !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 12px !important; font-size: 0.86rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Model helpers ──────────────────────────────────────────────────────────────
def validate_fruit(image: Image.Image, fruit: str):
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    avg_std = np.array([arr[:,:,i].std() for i in range(3)]).mean()
    if avg_std < 15:
        return False, "Image appears blank or too uniform."
    checks = {
        "banana":     (r > 120 and g > 100 and b < 120) or (r < 80 and g < 70),
        "apple":      (r > 130 and r > g * 1.2) or (g > 100 and g > r * 0.9),
        "mango":      (r > 150 and g > 100 and b < 100),
        "orange":     (r > 150 and g > 80 and b < 100),
        "grape":      (r > 80 and b > 60) or (r < 120 and g < 100 and b < 120),
        "strawberry": (r > 150 and r > g * 1.5),
        "tomato":     (r > 140 and r > g * 1.3),
        "avocado":    (g > 80 and r < 180),
        "peach":      (r > 180 and g > 120 and b > 80),
        "pear":       (g > 100 and r > 100),
        "kiwi":       (g > 90 and r < 160),
    }
    passed = checks.get(fruit, True)
    if not passed:
        return False, (f"Colors (R:{r:.0f} G:{g:.0f} B:{b:.0f}) don't match a {fruit}. "
                       f"Please upload a clear {fruit} photo.")
    return True, "OK"

@st.cache_resource
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fruit, file_id in MODEL_FILES.items():
        dest = os.path.join(MODEL_DIR, f"{fruit}_model.onnx")
        if not os.path.exists(dest):
            try:
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                gdown.download(url, dest, quiet=False)
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
                    path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as e:
                st.error(f"Could not load {fname}: {e}")
    return sessions

def preprocess(image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]
    return arr.astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_ripeness(image: Image.Image, fruit: str, session):
    tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits)
    class_names = ["overripe", "ripe"] if len(probs) == 2 else ["unripe", "ripe", "overripe"]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]) * 100, probs, class_names

# ── UI helpers ─────────────────────────────────────────────────────────────────
def confidence_bars_html(class_names, probs):
    html = ""
    for cls, prob in zip(class_names, probs):
        pct = float(prob) * 100
        color = COLORS.get(cls, "#888")
        html += f"""
        <div class="conf-wrap">
          <div class="conf-label">
            <span>{cls}</span><span>{pct:.1f}%</span>
          </div>
          <div class="conf-track">
            <div class="conf-fill" style="--target:{pct:.1f}%; background:{color};"></div>
          </div>
        </div>"""
    return html

def timeline_html(grade):
    stages = ["unripe", "ripe", "overripe"]
    labels = {"unripe": "🌱 Unripe", "ripe": "🍃 Ripe", "overripe": "🍂 Overripe"}
    html = '<div class="timeline">'
    for i, s in enumerate(stages):
        cls = f"active-{s}" if s == grade else ""
        html += f'<div class="tl-stage {cls}">{labels[s]}</div>'
        if i < len(stages) - 1:
            html += '<span class="tl-arrow">›</span>'
    html += '</div>'
    return html

def info_card_html(fruit, grade):
    info_db = FRUIT_INFO.get(fruit, {})
    info    = info_db.get(grade, DEFAULT_FRUIT_INFO.get(grade, {}))
    emoji   = FRUIT_INFO.get(fruit, {}).get("emoji", "🍑")
    icon    = info.get("icon", "")
    msg     = info.get("msg", "")
    tip     = info.get("tip", "")
    return f"""
    <div class="info-card {grade}">
      <div class="info-main">{icon} {emoji} {msg}</div>
      <div class="info-tip">💡 {tip}</div>
    </div>"""

def history_table_html(history):
    if not history:
        return "<p style='color:var(--muted); font-size:0.85rem;'>No scans yet — results will appear here.</p>"
    html = """
    <div class="hist-row hist-header">
      <div>Fruit</div><div>Grade</div><div>Confidence</div><div>Time</div>
    </div>"""
    for entry in reversed(history[-10:]):
        color = COLORS.get(entry["grade"], "#888")
        badge_cls = f"badge-{entry['grade']}"
        bar_w = min(entry["confidence"], 100)
        html += f"""
        <div class="hist-row">
          <div>{entry.get('emoji','🍑')} {entry['fruit'].capitalize()}</div>
          <div><span class="result-badge {badge_cls}">{entry['grade']}</span></div>
          <div>
            <span class="hist-conf-bar" style="width:{bar_w*0.6:.0f}px; background:{color};"></span>
            {entry['confidence']:.1f}%
          </div>
          <div style="color:var(--muted);">{entry['timestamp']}</div>
        </div>"""
    return html

# ── Load models ────────────────────────────────────────────────────────────────
done     = download_models()
sessions = load_models(done)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-emoji">🍃</span>
  <div>
    <div class="hero-title">FruitSense</div>
    <div class="hero-sub">Ripeness AI · Scan · Analyse · Act</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not sessions:
    st.error("No models loaded. Please check your model files.")
    st.stop()

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.15], gap="large")

# ─── LEFT COLUMN ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Scan a Fruit</div>', unsafe_allow_html=True)

    fruit = st.selectbox(
        "Select fruit",
        options=sorted(sessions.keys()),
        format_func=lambda x: f"{FRUIT_INFO.get(x,{}).get('emoji','🍑')}  {x.capitalize()}"
    )
    mode = st.radio("Input method", ["Upload Image", "Use Camera"], horizontal=True)

    image = None
    if mode == "Upload Image":
        uploaded = st.file_uploader("Drop a fruit image here", type=["jpg","jpeg","png"])
        if uploaded:
            image = Image.open(uploaded)
    else:
        photo = st.camera_input("Take a photo")
        if photo:
            image = Image.open(photo)

    predict_btn = st.button("✦  Analyse Ripeness", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── History ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🕘 Scan History</div>', unsafe_allow_html=True)
    st.markdown(history_table_html(st.session_state.history), unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─── RIGHT COLUMN ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Preview & Result</div>', unsafe_allow_html=True)

    if image is not None:
        st.image(image, caption=f"{fruit.capitalize()} — uploaded", use_container_width=True)

        if predict_btn:
            with st.spinner("Checking image…"):
                is_valid, message = validate_fruit(image, fruit)

            if not is_valid:
                st.warning(f"⚠️  {message}")
            else:
                with st.spinner("Analysing ripeness…"):
                    grade, confidence, probs, class_names = predict_ripeness(
                        image, fruit, sessions[fruit])

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"Detected {fruit}, but confidence is low ({confidence:.1f}%). Try a clearer image.")
                else:
                    # Save to history
                    st.session_state.history.append({
                        "fruit":      fruit,
                        "emoji":      FRUIT_INFO.get(fruit, {}).get("emoji", "🍑"),
                        "grade":      grade,
                        "confidence": round(confidence, 1),
                        "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    })

                    # ── Result header ──
                    badge_cls = f"badge-{grade}"
                    color = COLORS.get(grade, "#888")
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:1rem; margin:0.8rem 0;">
                      <span style="font-family:'Playfair Display',serif; font-size:1.6rem; color:{color};">
                        {fruit.capitalize()} is {grade.capitalize()}
                      </span>
                      <span class="result-badge {badge_cls}">{confidence:.1f}% confident</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Ripeness Timeline ──
                    st.markdown('<div class="card-title" style="margin-top:1rem;">📍 Ripeness Stage</div>', unsafe_allow_html=True)
                    st.markdown(timeline_html(grade), unsafe_allow_html=True)

                    # ── Fruit Info Card ──
                    st.markdown('<div class="card-title">📋 What to do</div>', unsafe_allow_html=True)
                    st.markdown(info_card_html(fruit, grade), unsafe_allow_html=True)

                    # ── Confidence Bars ──
                    st.markdown('<div class="card-title" style="margin-top:1.2rem;">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
                    st.markdown(confidence_bars_html(class_names, probs), unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color:var(--muted);">
          <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">🍃</div>
          <div style="font-size:0.9rem;">Upload or capture a fruit image<br>to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)