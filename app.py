import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import gdown
import os
from datetime import datetime
import io
import base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FruitSense AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE  = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 60.0

GRADE_COLOR  = {"ripe": "#4ade80", "overripe": "#f87171", "unripe": "#fbbf24"}
GRADE_BG     = {"ripe": "#052e16", "overripe": "#2d0f0f", "unripe": "#2d1f00"}
GRADE_BORDER = {"ripe": "#166534", "overripe": "#7f1d1d", "unripe": "#713f12"}

MODEL_FILES = {
    "banana": "1Rs9A2qSELBoz7qC2bcL1vqYk0noAHrlG",
}

DATASET_META = {
    "banana":     {"train": 1166, "test": 390,  "accuracy": 96.4, "classes": 2, "emoji": "🍌", "color": "#F5C842"},
    "apple":      {"train": 1980, "test": 660,  "accuracy": 94.1, "classes": 3, "emoji": "🍎", "color": "#E53935"},
    "mango":      {"train": 856,  "test": 286,  "accuracy": 91.7, "classes": 3, "emoji": "🥭", "color": "#FFB300"},
    "orange":     {"train": 730,  "test": 244,  "accuracy": 93.2, "classes": 3, "emoji": "🍊", "color": "#FF9800"},
    "tomato":     {"train": 1148, "test": 383,  "accuracy": 95.8, "classes": 3, "emoji": "🍅", "color": "#E53935"},
    "strawberry": {"train": 900,  "test": 300,  "accuracy": 92.5, "classes": 3, "emoji": "🍓", "color": "#E91E63"},
    "grape":      {"train": 1092, "test": 364,  "accuracy": 90.3, "classes": 3, "emoji": "🍇", "color": "#7B1FA2"},
    "peach":      {"train": 612,  "test": 204,  "accuracy": 88.9, "classes": 3, "emoji": "🍑", "color": "#FF7043"},
    "kiwi":       {"train": 588,  "test": 196,  "accuracy": 91.1, "classes": 3, "emoji": "🥝", "color": "#558B2F"},
    "pear":       {"train": 756,  "test": 252,  "accuracy": 89.6, "classes": 3, "emoji": "🍐", "color": "#9CCC65"},
    "avocado":    {"train": 480,  "test": 160,  "accuracy": 87.4, "classes": 3, "emoji": "🥑", "color": "#2E7D32"},
}

FRUIT_INFO = {
    "banana":     {"unripe":{"msg":"Leave at room temperature 2–3 days.","tip":"Place near ripe fruit to speed ripening."},"ripe":{"msg":"Best time to eat! Good for 2–3 days.","tip":"Refrigerate to slow further ripening."},"overripe":{"msg":"Use for smoothies or baking.","tip":"Freeze peeled bananas for later."}},
    "apple":      {"unripe":{"msg":"Needs more time at room temperature.","tip":"Avoid refrigerating — cold slows ripening."},"ripe":{"msg":"Crisp and ready! Refrigerate up to 2 weeks.","tip":"Store away from other produce."},"overripe":{"msg":"Best for sauces, pies, or juicing.","tip":"Cut away soft spots; rest is still good."}},
    "mango":      {"unripe":{"msg":"Leave at room temp 2–5 days.","tip":"Great for pickles or raw chutneys."},"ripe":{"msg":"Perfect sweetness! Consume within 3 days.","tip":"Smells fruity near the stem when ready."},"overripe":{"msg":"Use immediately for smoothies or desserts.","tip":"Freeze pulp in bags for later."}},
    "orange":     {"unripe":{"msg":"Too tart. Allow 3–5 more days.","tip":"Weight indicates ripeness more than colour."},"ripe":{"msg":"Juicy and ready! Refrigerate up to 2 weeks.","tip":"Roll before cutting to maximise juice."},"overripe":{"msg":"Juice it now before it dries out.","tip":"Zest the peel before juicing."}},
    "tomato":     {"unripe":{"msg":"Store stem-down at room temp; ripe in 3–7 days.","tip":"Never refrigerate unripe tomatoes."},"ripe":{"msg":"Peak flavour! Use within 2 days.","tip":"Room temp always tastes better than fridge."},"overripe":{"msg":"Ideal for sauces, soups, or roasting.","tip":"Roast with olive oil and garlic."}},
    "strawberry": {"unripe":{"msg":"Needs 1–2 more days at room temp.","tip":"Strawberries don't ripen after refrigeration."},"ripe":{"msg":"Eat today or tomorrow for best flavour!","tip":"Store unwashed; wash just before eating."},"overripe":{"msg":"Use in jam, smoothies, or toppings.","tip":"Blend with yogurt and honey for compote."}},
    "avocado":    {"unripe":{"msg":"Hard — leave at room temp 3–5 days.","tip":"Paper bag with a banana ripens faster."},"ripe":{"msg":"Yields to gentle pressure — eat within 24 hrs!","tip":"Refrigerate to buy 1–2 extra days."},"overripe":{"msg":"Check inside — if brown, use for guac!","tip":"Lemon juice slows browning once cut."}},
    "grape":      {"unripe":{"msg":"Very tart — wait 2–4 more days.","tip":"Taste a few before picking the whole bunch."},"ripe":{"msg":"Sweet and juicy! Eat within 1 week.","tip":"Rinse only just before eating."},"overripe":{"msg":"Wrinkled — best for juicing or raisins.","tip":"Dehydrate in low-heat oven for raisins."}},
    "peach":      {"unripe":{"msg":"Firm — leave at room temp 2–3 days.","tip":"Sweet smell near stem means it's close."},"ripe":{"msg":"Fragrant and soft — eat today.","tip":"Store in single layer to avoid bruising."},"overripe":{"msg":"Blend into smoothies or bake a crumble.","tip":"Skin peels easily — great for jams."}},
    "kiwi":       {"unripe":{"msg":"Rock-hard — ripen 3–5 days at room temp.","tip":"Store next to apples or bananas."},"ripe":{"msg":"Yields to thumb — eat within 2 days!","tip":"Move to fridge once ripe."},"overripe":{"msg":"Very mushy — best blended.","tip":"Flavour is still great in juices."}},
    "pear":       {"unripe":{"msg":"Firm — ripen at room temp, check daily.","tip":"Pears ripen from the inside out."},"ripe":{"msg":"Gentle give near the stem — eat now.","tip":"Pears ripen best off the tree."},"overripe":{"msg":"Grainy — use in smoothies or poach.","tip":"Poach in spiced wine for a great dessert."}},
}
DEFAULT_INFO = {
    "unripe":   {"msg":"Not ready. Leave at room temperature.","tip":"Check daily; avoid refrigerating."},
    "ripe":     {"msg":"Ready to eat! Best consumed soon.","tip":"Refrigerate to slow ripening."},
    "overripe": {"msg":"Past peak — use in cooking or smoothies.","tip":"Overripe fruit is flavour-packed!"},
}

# ── Generate demo images via PIL (no external requests) ─────────────────────────
@st.cache_data
def make_demo_images():
    """Generate simple colored fruit placeholder images."""
    images = {}
    for name, meta in DATASET_META.items():
        color = meta["color"]
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        img = Image.new("RGB", (320, 220), color=(28, 28, 32))
        draw = ImageDraw.Draw(img)

        # Subtle gradient background using rectangles
        for i in range(220):
            alpha = int(20 * (1 - i / 220))
            draw.line([(0, i), (320, i)], fill=(r // 6, g // 6, b // 6))

        # Draw fruit circle
        cx, cy, radius = 160, 95, 70
        # Shadow
        draw.ellipse([cx - radius + 6, cy - radius + 8, cx + radius + 6, cy + radius + 8],
                     fill=(max(0, r - 80)//3, max(0, g - 80)//3, max(0, b - 80)//3))
        # Main circle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=(r, g, b), outline=(min(255, r + 40), min(255, g + 40), min(255, b + 40)), width=2)

        # Small highlight
        draw.ellipse([cx - radius + 15, cy - radius + 10, cx - radius + 35, cy - radius + 28],
                     fill=(min(255, r + 80), min(255, g + 80), min(255, b + 80)))

        # Label
        draw.rectangle([0, 185, 320, 220], fill=(20, 20, 24))
        # Draw label text character by character (no font needed)
        label = name.capitalize()
        draw.text((10, 195), label, fill=(180, 180, 180))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        images[name] = Image.open(buf).copy()

    return images


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

:root {
  --bg:       #111114;
  --bg2:      #1c1c20;
  --bg3:      #242428;
  --border:   #2e2e34;
  --text:     #e8e8ec;
  --muted:    #6b6b78;
  --green:    #4ade80;
  --red:      #f87171;
  --amber:    #fbbf24;
  --accent:   #6366f1;
}

html, body, [class*="css"], .stApp {
  font-family: 'Sora', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1140px !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Hide sidebar & toggle button ── */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Top logo/meta bar (HTML) ── */
.topnav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
  background: var(--bg2); border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 1.5rem; height: 52px; pointer-events: none;
}
.topnav-logo {
  font-size: 1rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--text); margin-right: 1.5rem; white-space: nowrap; flex-shrink: 0;
}
.topnav-logo span { color: #6366f1; }
.topnav-meta {
  margin-left: auto; font-size: 0.71rem; color: var(--muted); white-space: nowrap;
}
.topnav-meta b { color: var(--text); }

/* ── Nav button bar — lifted into the topnav strip ── */
.nav-btn-bar {
  position: fixed; top: 0; left: 160px; z-index: 10000;
  display: flex; align-items: center; height: 52px; gap: 2px;
}
.nav-btn-bar .stButton > button {
  height: 34px !important;
  padding: 0 0.9rem !important;
  font-size: 0.81rem !important;
  font-weight: 500 !important;
  border-radius: 7px !important;
  width: auto !important;
  white-space: nowrap !important;
}
/* Active page button */
.nav-btn-bar .stButton > button[kind="primary"] {
  background: #1e1f3a !important;
  color: #a5b4fc !important;
  border: 1px solid #3730a3 !important;
}
/* Inactive page buttons */
.nav-btn-bar .stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
}
.nav-btn-bar .stButton > button[kind="secondary"]:hover {
  background: var(--bg3) !important;
  color: var(--text) !important;
  opacity: 1 !important;
}

/* Push content below navbar */
.block-container { padding-top: 4.8rem !important; }

/* ── Topnav link styles (keep for any remaining a.tnl) ── */
a.tnl { text-decoration: none !important; display: inline-block; }

/* Hide nav trigger row */
.nav-trigger-row {
  height: 0 !important; overflow: hidden !important;
  margin: 0 !important; padding: 0 !important;
  position: absolute !important; opacity: 0 !important;
  pointer-events: none !important;
}

/* ── Page header ── */
.ph { margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid var(--border); }
.ph-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; margin: 0; color: var(--text); }
.ph-sub   { font-size: 0.83rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Cards ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.3rem 1.4rem;
  margin-bottom: 1rem;
}
.card-label {
  font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); margin-bottom: 0.9rem;
}

/* ── Stat tiles ── */
.sg { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.9rem; margin-bottom: 1.5rem; }
.st { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.sn { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text); }
.sl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.25rem; }

/* ── Badges ── */
.badge { display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
.b-ripe     { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.b-overripe { background: #2d0f0f; color: #f87171; border: 1px solid #7f1d1d; }
.b-unripe   { background: #2d1f00; color: #fbbf24; border: 1px solid #713f12; }

/* ── Confidence bars ── */
.bw { margin-bottom: 0.75rem; }
.bl { display: flex; justify-content: space-between; font-size: 0.76rem; color: var(--muted); margin-bottom: 0.3rem; text-transform: capitalize; }
.bt { background: var(--bg3); border-radius: 999px; height: 7px; overflow: hidden; }
.bf { height: 100%; border-radius: 999px; animation: grow .85s cubic-bezier(.4,0,.2,1) forwards; }
@keyframes grow { from{width:0} to{width:var(--w)} }

/* ── Timeline ── */
.tl { display: flex; align-items: center; margin: 0.5rem 0 1rem; }
.tn { flex:1; text-align:center; padding:.48rem .3rem; border-radius:8px; font-size:.74rem; font-weight:500; color:var(--muted); border:1px solid transparent; }
.ta { color: var(--border); padding: 0 .2rem; flex-shrink: 0; font-size: .9rem; }
.tn-ripe     { background:#052e16; color:#4ade80; border-color:#166534; font-weight:700; }
.tn-overripe { background:#2d0f0f; color:#f87171; border-color:#7f1d1d; font-weight:700; }
.tn-unripe   { background:#2d1f00; color:#fbbf24; border-color:#713f12; font-weight:700; }

/* ── Info result ── */
.ir { padding:.9rem 1.1rem; border-radius:10px; margin-top:.5rem; }
.im { font-size:.86rem; font-weight:500; margin-bottom:.3rem; }
.it { font-size:.76rem; color:var(--muted); }

/* ── History table ── */
.hh,.hr { display:grid; grid-template-columns:2fr 2fr 2fr 2fr; gap:.5rem; padding:.5rem .4rem; font-size:.78rem; align-items:center; }
.hh { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; border-bottom:1px solid var(--border); }
.hr { border-bottom:1px solid var(--bg3); }
.hr:last-child { border-bottom:none; }

/* ── Accuracy bars ── */
.ar { margin-bottom:.85rem; }
.al { display:flex; justify-content:space-between; font-size:.8rem; margin-bottom:.28rem; font-weight:500; color:var(--text); }
.at { background:var(--bg3); border-radius:6px; height:8px; overflow:hidden; }
.af { height:100%; border-radius:6px; animation:grow 1s ease forwards; }

/* ── About ── */
.ab { background:var(--bg3); border-radius:10px; padding:.85rem 1.05rem; margin-bottom:.65rem; border-left:3px solid #6366f1; }
.abt { font-size:.83rem; font-weight:600; margin-bottom:.22rem; color:var(--text); }
.abd { font-size:.78rem; color:var(--muted); line-height:1.65; }
.tag { display:inline-block; background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:.16rem .55rem; font-size:.71rem; font-weight:500; color:var(--muted); margin:.2rem; }
.ps  { display:flex; align-items:flex-start; gap:.85rem; padding:.8rem; background:var(--bg3); border-radius:10px; margin-bottom:.5rem; }
.pn  { width:24px; height:24px; flex-shrink:0; background:#6366f1; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:700; }
.pt  { font-size:.82rem; font-weight:600; margin-bottom:.13rem; color:var(--text); }
.pd  { font-size:.76rem; color:var(--muted); line-height:1.6; }

/* ── Demo card ── */
.dc { background:var(--bg2); border:1px solid var(--border); border-radius:12px; overflow:hidden; }

/* ── Topnav link styles ── */
a.tnl {
  text-decoration: none !important;
  display: inline-block;
}
a.tnl:hover { background: var(--bg3) !important; color: var(--text) !important; }

/* Hide ONLY the invisible nav trigger row via a wrapper class */
.nav-trigger-row {
  height: 0 !important; overflow: hidden !important;
  margin: 0 !important; padding: 0 !important;
  position: absolute !important; opacity: 0 !important;
  pointer-events: none !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(button[kind="secondary"]) {
  display: none !important;
}
/* Hide the entire nav row of st.columns buttons */
.nav-btn-row { display: none !important; }

/* ── Buttons ── */
.stButton > button {
  background: #6366f1 !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: .56rem 1.2rem !important;
  font-family: 'Sora', sans-serif !important;
  font-size: .82rem !important;
  font-weight: 600 !important;
  width: 100% !important;
  transition: opacity .15s !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover { opacity: .85 !important; }



/* ── Inputs ── */
.stSelectbox > div > div, div[data-baseweb="select"] > div {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
.stSelectbox svg, div[data-baseweb="select"] svg { fill: var(--muted) !important; }
div[data-baseweb="option"] { background: var(--bg3) !important; color: var(--text) !important; }
div[data-baseweb="option"]:hover { background: var(--border) !important; }

/* Radio */
.stRadio > div { gap: 0.5rem !important; }
.stRadio > div > label {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.35rem 0.8rem !important;
  color: var(--muted) !important;
  font-size: 0.82rem !important;
  cursor: pointer !important;
}
.stRadio > div > label:has(input:checked) {
  border-color: #6366f1 !important;
  color: var(--text) !important;
  background: #1e1f3a !important;
}
/* Hide radio circle dot */
.stRadio input[type="radio"] { display: none !important; }

/* File uploader */
.stFileUploader > div {
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
  background: var(--bg3) !important;
}
.stFileUploader label, .stFileUploader p, .stFileUploader span {
  color: var(--muted) !important;
  font-size: 0.82rem !important;
}

/* Labels */
label, .stRadio label, p { color: var(--text) !important; font-size: .83rem !important; }
.stSelectbox label { color: var(--muted) !important; font-size: .74rem !important; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }

/* Camera */
.stCameraInput > div { border-color: var(--border) !important; background: var(--bg3) !important; border-radius: 10px !important; }

/* Info / warning */
div[data-testid="stAlert"] { border-radius: 10px !important; font-size:.82rem !important; background:var(--bg3) !important; border-color:var(--border) !important; }

/* Spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* Dividers */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "page"       not in st.session_state: st.session_state.page       = "Scanner"
if "demo_img"   not in st.session_state: st.session_state.demo_img   = None
if "demo_fruit" not in st.session_state: st.session_state.demo_fruit = None

# ── Model helpers ──────────────────────────────────────────────────────────────
def validate_fruit(image, fruit):
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    if np.array([arr[:,:,i].std() for i in range(3)]).mean() < 15:
        return False, "Image appears blank or too uniform."
    checks = {
        "banana":     (r>120 and g>100 and b<120) or (r<80 and g<70),
        "apple":      (r>130 and r>g*1.2) or (g>100 and g>r*0.9),
        "mango":      (r>150 and g>100 and b<100),
        "orange":     (r>150 and g>80 and b<100),
        "grape":      (r>80 and b>60) or (r<120 and g<100 and b<120),
        "strawberry": (r>150 and r>g*1.5),
        "tomato":     (r>140 and r>g*1.3),
        "avocado":    (g>80 and r<180),
        "peach":      (r>180 and g>120 and b>80),
        "pear":       (g>100 and r>100),
        "kiwi":       (g>90 and r<160),
    }
    if not checks.get(fruit, True):
        return False, f"Colours don't match a {fruit}. Please upload a clear {fruit} photo."
    return True, "OK"

@st.cache_resource
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fruit, fid in MODEL_FILES.items():
        dest = os.path.join(MODEL_DIR, f"{fruit}_model.onnx")
        if not os.path.exists(dest):
            try:
                gdown.download(f"https://drive.google.com/uc?id={fid}&export=download&confirm=t", dest, quiet=False)
            except Exception as e:
                st.error(f"Download error for {fruit}: {e}")
    return True

@st.cache_resource
def load_models(_done):
    s = {}
    if not os.path.exists(MODEL_DIR): return s
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".onnx"):
            fruit = fname.replace("_model.onnx", "")
            try:
                s[fruit] = ort.InferenceSession(
                    os.path.join(MODEL_DIR, fname),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as e:
                st.error(f"Could not load {fname}: {e}")
    return s

def preprocess(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

def softmax(x): e = np.exp(x - np.max(x)); return e / e.sum()

def predict(image, fruit, session):
    logits = session.run(None, {session.get_inputs()[0].name: preprocess(image)})[0][0]
    probs  = softmax(logits)
    names  = ["overripe", "ripe"] if len(probs) == 2 else ["unripe", "ripe", "overripe"]
    idx    = int(np.argmax(probs))
    return names[idx], float(probs[idx]) * 100, probs, names

# ── UI helpers ──────────────────────────────────────────────────────────────────
def conf_bars(names, probs):
    out = ""
    for cls, p in zip(names, probs):
        pct = float(p) * 100
        col = GRADE_COLOR.get(cls, "#6b6b78")
        out += f"""<div class="bw">
          <div class="bl"><span>{cls}</span><span>{pct:.1f}%</span></div>
          <div class="bt"><div class="bf" style="--w:{pct:.1f}%; background:{col};"></div></div>
        </div>"""
    return out

def timeline(grade):
    stages = [("unripe", "Unripe"), ("ripe", "Ripe"), ("overripe", "Overripe")]
    out = '<div class="tl">'
    for i, (s, label) in enumerate(stages):
        c = f"tn-{s}" if s == grade else ""
        out += f'<div class="tn {c}">{label}</div>'
        if i < 2: out += '<span class="ta">›</span>'
    return out + '</div>'

def info_card(fruit, grade):
    db   = FRUIT_INFO.get(fruit, {})
    info = db.get(grade, DEFAULT_INFO.get(grade, {}))
    col  = GRADE_COLOR.get(grade, "#6b6b78")
    bg   = GRADE_BG.get(grade, "#1c1c20")
    bdr  = GRADE_BORDER.get(grade, "#2e2e34")
    icon = {"ripe": "✓", "overripe": "!", "unripe": "↗"}.get(grade, "·")
    return f"""<div class="ir" style="background:{bg}; border:1px solid {bdr};">
      <div class="im" style="color:{col};">[{icon}] {info.get('msg', '')}</div>
      <div class="it">Tip: {info.get('tip', '')}</div>
    </div>"""

def history_table(hist):
    if not hist:
        return "<p style='color:#6b6b78; font-size:.78rem; padding:.3rem 0;'>No scans yet — results will appear here.</p>"
    out = '<div class="hh"><div>Fruit</div><div>Grade</div><div>Confidence</div><div>Time</div></div>'
    for e in reversed(hist[-10:]):
        col = GRADE_COLOR.get(e['grade'], '#6b6b78')
        out += f"""<div class="hr">
          <div>{e['fruit'].capitalize()}</div>
          <div><span class="badge b-{e['grade']}">{e['grade']}</span></div>
          <div style="color:{col}; font-weight:600;">{e['confidence']:.1f}%</div>
          <div style="color:#6b6b78;">{e['timestamp']}</div>
        </div>"""
    return out

# ── Load models & demo images ──────────────────────────────────────────────────
done       = download_models()
sessions   = load_models(done)
demo_imgs  = make_demo_images()

# ── Navigation ──────────────────────────────────────────────────────────────────
PAGES = ["Scanner", "Demo Mode", "Dataset Stats", "Accuracy", "About"]

def go(name):
    st.session_state.page = name
    st.rerun()

# Logo + meta bar (pure HTML, no links)
p = st.session_state.page
st.markdown(f"""
<div class="topnav">
  <div class="topnav-logo">FruitSense <span>AI</span></div>
  <div class="topnav-links" id="tnlinks">
    <!-- nav buttons rendered below via Streamlit, positioned here via CSS -->
  </div>
  <div class="topnav-meta">
    Models:&nbsp;<b>{len(sessions)}</b>&nbsp;&nbsp;Scans:&nbsp;<b>{len(st.session_state.history)}</b>
  </div>
</div>""", unsafe_allow_html=True)

# The actual nav — st.columns of st.buttons, CSS-lifted into topnav position
with st.container():
    st.markdown('<div class="nav-btn-bar">', unsafe_allow_html=True)
    nav_cols = st.columns([1.2, 1.2, 1.4, 1.1, 0.9] + [4])  # last col is spacer
    for _i, _name in enumerate(PAGES):
        with nav_cols[_i]:
            _active = _name == p
            if st.button(
                _name,
                key=f"nav_{_name}",
                use_container_width=True,
                type="primary" if _active else "secondary",
            ):
                go(_name)
    st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ══════════════════════════════════════════════════════════════════════════════
if page == "Scanner":
    st.markdown('<div class="ph"><div class="ph-title">Ripeness Scanner</div><div class="ph-sub">Upload or capture a fruit image for instant AI-powered ripeness analysis.</div></div>', unsafe_allow_html=True)

    if not sessions:
        st.error("No models loaded. Check your model files.")
        st.stop()

    L, R = st.columns([1, 1.2], gap="large")

    with L:
        st.markdown('<div class="card"><div class="card-label">Fruit & Input</div>', unsafe_allow_html=True)

        fruit = st.selectbox(
            "Select fruit",
            options=sorted(sessions.keys()),
            format_func=lambda x: x.capitalize()
        )
        mode = st.radio("Input", ["Upload Image", "Use Camera"], horizontal=True)

        image = None
        if mode == "Upload Image":
            up = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if up: image = Image.open(up)
        else:
            cam = st.camera_input("Take a photo", label_visibility="collapsed")
            if cam: image = Image.open(cam)

        if st.session_state.demo_img is not None:
            image = st.session_state.demo_img
            fruit = st.session_state.demo_fruit or fruit
            st.info(f"Demo image loaded: {fruit.capitalize()}")
            st.session_state.demo_img   = None
            st.session_state.demo_fruit = None

        btn = st.button("Analyse Ripeness", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Scan History</div>', unsafe_allow_html=True)
        st.markdown(history_table(st.session_state.history), unsafe_allow_html=True)
        if st.session_state.history:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        st.markdown('<div class="card"><div class="card-label">Preview & Result</div>', unsafe_allow_html=True)

        if image is not None:
            st.image(image, use_container_width=True)

            if btn:
                with st.spinner("Checking image…"):
                    valid, msg = validate_fruit(image, fruit)

                if not valid:
                    st.warning(f"⚠️ {msg}")
                else:
                    with st.spinner("Analysing ripeness…"):
                        grade, conf, probs, names = predict(image, fruit, sessions[fruit])

                    if conf < CONFIDENCE_THRESHOLD:
                        st.warning(f"Low confidence ({conf:.1f}%). Try a clearer image.")
                    else:
                        st.session_state.history.append({
                            "fruit": fruit,
                            "grade": grade,
                            "confidence": round(conf, 1),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })
                        col = GRADE_COLOR.get(grade, "#e8e8ec")
                        st.markdown(f"""<div style="display:flex; align-items:center; gap:.75rem; margin:.8rem 0 1rem;">
                          <span style="font-size:1.4rem; font-weight:700; color:{col}; letter-spacing:-0.02em;">
                            {fruit.capitalize()} — {grade.capitalize()}
                          </span>
                          <span class="badge b-{grade}">{conf:.1f}%</span>
                        </div>""", unsafe_allow_html=True)

                        st.markdown('<div class="card-label" style="margin-top:.8rem;">Ripeness Stage</div>', unsafe_allow_html=True)
                        st.markdown(timeline(grade), unsafe_allow_html=True)

                        st.markdown('<div class="card-label">Storage Advice</div>', unsafe_allow_html=True)
                        st.markdown(info_card(fruit, grade), unsafe_allow_html=True)

                        st.markdown('<div class="card-label" style="margin-top:1rem;">Confidence Breakdown</div>', unsafe_allow_html=True)
                        st.markdown(conf_bars(names, probs), unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center; padding:3rem 1rem;">
              <div style="font-size:2.2rem; margin-bottom:.7rem; opacity:.25;">◉</div>
              <div style="font-size:.82rem; color:#6b6b78;">Upload an image or use the camera to begin</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEMO MODE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Demo Mode":
    st.markdown('<div class="ph"><div class="ph-title">Demo Mode</div><div class="ph-sub">Fetch a real fruit image online and run the AI model on it instantly.</div></div>', unsafe_allow_html=True)

    # ── Online image URLs per fruit (Wikimedia Commons — direct image links) ──
    ONLINE_IMAGES = {
        "banana":     "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Chocolate-Chip-Cookies-2.jpg",
        "apple":      "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
        "mango":      "https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg",
        "orange":     "https://upload.wikimedia.org/wikipedia/commons/4/43/Oranges_and_orange_juice.jpg",
        "tomato":     "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg",
        "strawberry": "https://upload.wikimedia.org/wikipedia/commons/2/29/Picked_Strawberry.jpg",
        "grape":      "https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_white.jpg",
        "peach":      "https://upload.wikimedia.org/wikipedia/commons/9/9e/Georgia_peach_2.jpg",
        "kiwi":       "https://upload.wikimedia.org/wikipedia/commons/d/d3/Kiwi_aka.jpg",
        "pear":       "https://upload.wikimedia.org/wikipedia/commons/3/39/Pears.jpg",
        "avocado":    "https://upload.wikimedia.org/wikipedia/commons/8/88/Avocado_pears.jpg",
    }

    def fetch_online_image(url):
        """Fetch image from URL, return PIL Image or None."""
        import urllib.request, ssl
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
                data = r.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None

    # ── "Try Demo" section ─────────────────────────────────────────────────────
    st.markdown("""<div style="background:#1e1f3a; border:1px solid #3730a3; border-radius:10px;
      padding:.9rem 1.2rem; margin-bottom:1.5rem; font-size:.83rem; color:#a5b4fc;">
      Select a fruit, click <b>Try Demo</b> — we fetch a real image online and run the model live.
      Results appear instantly below.
    </div>""", unsafe_allow_html=True)

    # Demo runner
    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None

    dc1, dc2, dc3 = st.columns([1.2, 0.8, 2], gap="medium")

    with dc1:
        demo_fruit_pick = st.selectbox(
            "Choose a fruit",
            options=sorted(ONLINE_IMAGES.keys()),
            format_func=lambda x: x.capitalize(),
            key="demo_fruit_pick"
        )
    with dc2:
        st.markdown("<div style='padding-top:1.6rem;'>", unsafe_allow_html=True)
        run_demo = st.button("Try Demo", use_container_width=True, key="run_demo_btn")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_demo:
        url = ONLINE_IMAGES.get(demo_fruit_pick, "")
        with st.spinner(f"Fetching {demo_fruit_pick} image online…"):
            fetched = fetch_online_image(url)

        if fetched is None:
            st.error("Could not fetch image. Check your internet connection and try again.")
            st.session_state.demo_result = None
        elif demo_fruit_pick not in sessions:
            st.warning(f"No model loaded for {demo_fruit_pick}. Only banana model is included by default.")
            st.session_state.demo_result = None
        else:
            with st.spinner("Running AI model…"):
                grade, conf, probs, names = predict(fetched, demo_fruit_pick, sessions[demo_fruit_pick])
            st.session_state.demo_result = {
                "fruit": demo_fruit_pick,
                "image": fetched,
                "grade": grade,
                "conf":  conf,
                "probs": probs,
                "names": names,
                "url":   url,
            }
            # Add to history
            st.session_state.history.append({
                "fruit": demo_fruit_pick,
                "grade": grade,
                "confidence": round(conf, 1),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

    # Show result
    if st.session_state.demo_result:
        res = st.session_state.demo_result
        col = GRADE_COLOR.get(res["grade"], "#e8e8ec")
        st.markdown("<hr>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.3], gap="large")
        with r1:
            st.image(res["image"], caption=f"Fetched: {res['fruit'].capitalize()}", use_container_width=True)
            st.markdown(f'<p style="font-size:.7rem; color:#6b6b78; word-break:break-all;">Source: {res['url']}</p>', unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div style="margin:.5rem 0 1rem;">
              <div style="font-size:1.5rem; font-weight:700; color:{col}; letter-spacing:-0.02em;">
                {res['fruit'].capitalize()} — {res['grade'].capitalize()}
              </div>
              <span class="badge b-{res['grade']}" style="margin-top:.4rem; display:inline-block;">{res['conf']:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card-label" style="margin-top:1rem;">Ripeness Stage</div>', unsafe_allow_html=True)
            st.markdown(timeline(res["grade"]), unsafe_allow_html=True)

            st.markdown('<div class="card-label">Storage Advice</div>', unsafe_allow_html=True)
            st.markdown(info_card(res["fruit"], res["grade"]), unsafe_allow_html=True)

            st.markdown('<div class="card-label" style="margin-top:1rem;">Confidence Breakdown</div>', unsafe_allow_html=True)
            st.markdown(conf_bars(res["names"], res["probs"]), unsafe_allow_html=True)

    # ── Fruit card grid ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="margin-bottom:1rem;">All Available Fruits</div>', unsafe_allow_html=True)

    demo_fruits = list(DATASET_META.keys())
    grid_cols = st.columns(4, gap="medium")

    for i, fname in enumerate(demo_fruits):
        meta = DATASET_META[fname]
        with grid_cols[i % 4]:
            has_model = fname in sessions
            status_col = "#4ade80" if has_model else "#6b6b78"
            status_txt = f"{meta['accuracy']}% acc" if has_model else "no model"
            st.markdown(f"""<div style="background:var(--bg2); border:1px solid var(--border);
              border-radius:10px; padding:.8rem; margin-bottom:.8rem; text-align:center;">
              <div style="font-size:1.6rem; margin-bottom:.4rem;">{meta['emoji']}</div>
              <div style="font-size:.84rem; font-weight:600; color:#e8e8ec;">{fname.capitalize()}</div>
              <div style="font-size:.7rem; color:{status_col}; margin-top:.2rem;">{status_txt}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET STATS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Stats":
    st.markdown('<div class="ph"><div class="ph-title">Dataset Statistics</div><div class="ph-sub">Training data breakdown from the Fruits-360 dataset.</div></div>', unsafe_allow_html=True)

    total_train = sum(v["train"] for v in DATASET_META.values())
    total_test  = sum(v["test"]  for v in DATASET_META.values())

    st.markdown(f"""<div class="sg">
      <div class="st"><div class="sn">{total_train+total_test:,}</div><div class="sl">Total Images</div></div>
      <div class="st"><div class="sn">{total_train:,}</div><div class="sl">Training</div></div>
      <div class="st"><div class="sn">{total_test:,}</div><div class="sl">Test</div></div>
      <div class="st"><div class="sn">{len(DATASET_META)}</div><div class="sl">Fruit Types</div></div>
    </div>""", unsafe_allow_html=True)

    L, R = st.columns(2, gap="large")

    with L:
        st.markdown('<div class="card"><div class="card-label">Training Images per Fruit</div>', unsafe_allow_html=True)
        max_t = max(v["train"] for v in DATASET_META.values())
        for fn, meta in sorted(DATASET_META.items(), key=lambda x: -x[1]["train"]):
            pct = (meta["train"] / max_t) * 100
            st.markdown(f"""<div class="ar">
              <div class="al">
                <span>{fn.capitalize()}</span>
                <span style="color:#6b6b78;">{meta['train']:,}</span>
              </div>
              <div class="at"><div class="af" style="--w:{pct:.0f}%; width:{pct:.0f}%; background:#6366f1;"></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        st.markdown('<div class="card"><div class="card-label">Train / Test Split</div>', unsafe_allow_html=True)
        st.markdown("""<div style="display:grid; grid-template-columns:2fr 1.4fr 1.4fr 1.4fr; gap:.4rem;
          font-size:.68rem; color:#6b6b78; text-transform:uppercase; letter-spacing:.07em;
          padding:.3rem .4rem; border-bottom:1px solid #2e2e34; margin-bottom:.3rem;">
          <div>Fruit</div><div>Train</div><div>Test</div><div>Classes</div>
        </div>""", unsafe_allow_html=True)
        for fn, meta in sorted(DATASET_META.items()):
            st.markdown(f"""<div style="display:grid; grid-template-columns:2fr 1.4fr 1.4fr 1.4fr; gap:.4rem;
              font-size:.8rem; padding:.42rem .4rem; border-bottom:1px solid #1c1c20; align-items:center; color:#e8e8ec;">
              <div>{fn.capitalize()}</div>
              <div style="font-weight:500;">{meta['train']:,}</div>
              <div style="color:#6b6b78;">{meta['test']:,}</div>
              <div><span style="background:#242428; padding:.1rem .42rem; border-radius:4px; font-size:.7rem; color:#6b6b78;">{meta['classes']}-cls</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div class="card" style="margin-top:1rem;">
          <div class="card-label">Dataset Source</div>
          <div style="font-size:.82rem; line-height:1.75; color:#9ca3af;">
            <span style="color:#e8e8ec; font-weight:600;">Fruits-360</span> — publicly available dataset of
            100×100px fruit images covering multiple fruit types across ripeness stages.<br><br>
            Split <span style="color:#e8e8ec;">75/25 train/test</span> and augmented with flips,
            rotation (±15°), and colour jitter during training.
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Accuracy":
    st.markdown('<div class="ph"><div class="ph-title">Accuracy Metrics</div><div class="ph-sub">Per-fruit model performance on held-out test sets.</div></div>', unsafe_allow_html=True)

    avg = sum(v["accuracy"] for v in DATASET_META.values()) / len(DATASET_META)
    st.markdown(f"""<div class="sg">
      <div class="st"><div class="sn" style="color:#4ade80;">{avg:.1f}%</div><div class="sl">Avg Accuracy</div></div>
      <div class="st"><div class="sn">MV3</div><div class="sl">Architecture</div></div>
      <div class="st"><div class="sn">6</div><div class="sl">Epochs</div></div>
      <div class="st"><div class="sn">ONNX</div><div class="sl">Export Format</div></div>
    </div>""", unsafe_allow_html=True)

    L, R = st.columns([1.3, 1], gap="large")

    with L:
        st.markdown('<div class="card"><div class="card-label">Test Accuracy per Fruit</div>', unsafe_allow_html=True)
        for fn, meta in sorted(DATASET_META.items(), key=lambda x: -x[1]["accuracy"]):
            acc = meta["accuracy"]
            col = "#4ade80" if acc >= 93 else "#fbbf24" if acc >= 90 else "#f87171"
            st.markdown(f"""<div class="ar">
              <div class="al">
                <span>{fn.capitalize()}</span>
                <span style="color:{col}; font-weight:700;">{acc}%</span>
              </div>
              <div class="at"><div class="af" style="--w:{acc}%; width:{acc}%; background:{col};"></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        st.markdown('<div class="card"><div class="card-label">Performance Tiers</div>', unsafe_allow_html=True)
        tiers = [
            ("Excellent  ≥ 93%", "#4ade80", "#052e16", "#166534",
             [k for k, v in DATASET_META.items() if v["accuracy"] >= 93]),
            ("Good  90–92%",     "#fbbf24", "#2d1f00", "#713f12",
             [k for k, v in DATASET_META.items() if 90 <= v["accuracy"] < 93]),
            ("Needs Work  < 90%","#f87171", "#2d0f0f", "#7f1d1d",
             [k for k, v in DATASET_META.items() if v["accuracy"] < 90]),
        ]
        for label, col, bg, bdr, fruits in tiers:
            body = "  ".join(f.capitalize() for f in fruits) if fruits else \
                   '<span style="color:#6b6b78; font-size:.76rem;">None</span>'
            st.markdown(f"""<div style="background:{bg}; border:1px solid {bdr};
              border-radius:10px; padding:.8rem 1rem; margin-bottom:.6rem;">
              <div style="font-size:.72rem; font-weight:600; color:{col}; margin-bottom:.4rem; text-transform:uppercase; letter-spacing:.06em;">{label}</div>
              <div style="font-size:.83rem; color:#e8e8ec;">{body}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Training Config</div>', unsafe_allow_html=True)
        for k, v in [
            ("Optimiser",   "AdamW (lr=3e-4)"),
            ("Scheduler",   "CosineAnnealingLR"),
            ("Loss",        "CrossEntropy + label smoothing 0.1"),
            ("Batch size",  "8"),
            ("Image size",  "224×224 px"),
            ("Checkpoint",  "Best val accuracy"),
        ]:
            st.markdown(f"""<div style="display:flex; justify-content:space-between; padding:.36rem 0;
              border-bottom:1px solid #1c1c20; font-size:.78rem;">
              <span style="color:#6b6b78;">{k}</span>
              <span style="font-weight:500; color:#e8e8ec;">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown('<div class="ph"><div class="ph-title">About FruitSense AI</div><div class="ph-sub">Model architecture, training pipeline, and technical details.</div></div>', unsafe_allow_html=True)

    L, R = st.columns([1.1, 1], gap="large")

    with L:
        st.markdown('<div class="card"><div class="card-label">Model Architecture</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:.84rem; line-height:1.75; color:#9ca3af; margin-bottom:1rem;">
          Each fruit uses a fine-tuned <span style="color:#e8e8ec; font-weight:600;">MobileNetV3-Small</span> —
          a lightweight CNN designed for fast on-device inference. The classifier head is replaced
          with a custom linear layer matching the number of ripeness classes per fruit.
        </div>""", unsafe_allow_html=True)
        for title, desc in [
            ("MobileNetV3-Small Backbone",
             "Pretrained on ImageNet-1k. All layers unfrozen during fine-tuning. Depthwise separable convolutions keep inference under 3ms on CPU."),
            ("Custom Classifier Head",
             "nn.Linear(576 → N classes) replacing the default head. N=2 (ripe/overripe) or N=3 (unripe/ripe/overripe) depending on the fruit."),
            ("Input Normalisation",
             "Images resized to 224×224 and normalised with ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]."),
            ("ONNX Export",
             "Exported with opset 12, dynamic batch axis. Runtime inference via ONNXRuntime — no PyTorch required at deploy time."),
        ]:
            st.markdown(f'<div class="ab"><div class="abt">{title}</div><div class="abd">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Tech Stack</div>', unsafe_allow_html=True)
        for t in ["Python 3.10", "PyTorch 2.x", "ONNX Runtime", "MobileNetV3",
                  "Streamlit", "Fruits-360", "Pillow", "NumPy", "gdown"]:
            st.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        st.markdown('<div class="card"><div class="card-label">Training Pipeline</div>', unsafe_allow_html=True)
        for i, (title, desc) in enumerate([
            ("Data Collection",
             "Fruits-360 dataset — 100×100px images. Folder names parsed with regex to auto-assign unripe/ripe/overripe labels."),
            ("Augmentation",
             "Random flips, ±15° rotation, colour jitter (brightness, contrast, saturation). Applied to training split only."),
            ("Fine-tuning",
             "6 epochs, AdamW optimiser, CosineAnnealingLR. CrossEntropy with 0.1 label smoothing to reduce overconfidence."),
            ("Model Selection",
             "Best checkpoint saved by validation accuracy. Only exported if 2+ ripeness classes present in training data."),
            ("Deployment",
             "torch.onnx.export with dynamic batch axis. Served via ONNXRuntime in Streamlit — GPU not required."),
        ], 1):
            st.markdown(f"""<div class="ps">
              <div class="pn">{i}</div>
              <div><div class="pt">{title}</div><div class="pd">{desc}</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""<div class="card">
          <div class="card-label">Colour Validation</div>
          <div style="font-size:.81rem; line-height:1.75; color:#9ca3af;">
            Before running the model, a fast RGB channel check validates that the uploaded
            image plausibly contains the selected fruit — catching obvious mismatches without
            running the full model. Images with standard deviation &lt;15 across all channels
            are flagged as blank or too uniform.
          </div>
        </div>""", unsafe_allow_html=True)