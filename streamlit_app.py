"""
CardioCheck - Heart Disease Prediction (Streamlit version)
==========================================================
Same trained model as the Flask version, shown with Streamlit so it can be
deployed on Streamlit Community Cloud.

Run locally:   streamlit run streamlit_app.py
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "svc_trained_model.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

# The same 4 attributes (and order) used for training
FEATURES = ["sex", "cp", "exang", "oldpeak"]

CHEST_PAIN = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "No symptoms": 4,
}

st.set_page_config(page_title="CardioCheck | Heart disease prediction",
                   page_icon="🫀", layout="wide")


# ----------------------------------------------------------------------
# Model (loaded once, retrained automatically if it cannot be loaded)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


@st.cache_data
def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return {}


model = load_model()
metrics = load_metrics()
accuracy = metrics.get("accuracy", 0.83)
total_patients = metrics.get("total_patients", 857)


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: "Plus Jakarta Sans", system-ui, sans-serif; }
.stApp { background: #f4f6f9; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding: 0 0 2rem 0 !important; max-width: 100% !important; }

.nav { background: #fff; border-bottom: 1px solid #e1e8ef; padding: 0 max(20px, calc((100% - 1160px) / 2)); height: 68px; display: flex; align-items: center; justify-content: space-between; }
.brand { display: flex; align-items: center; gap: 12px; font-weight: 800; font-size: 1.2rem; color: #0f2a43; letter-spacing: -.01em; }
.nav-note { color: #6b8196; font-size: .92rem; font-weight: 500; }

.hero { background: radial-gradient(900px 340px at 85% -10%, rgba(15,111,255,.35), transparent 70%), linear-gradient(180deg, #0b1f33, #12304b); color: #fff; padding: 52px max(20px, calc((100% - 1160px) / 2)) 110px; }
.hero h1 { margin: 0 0 16px; max-width: 21ch; font-size: clamp(2.1rem, 5vw, 3.6rem); line-height: 1.05; font-weight: 800; letter-spacing: -.03em; color: #fff; padding: 0; }
.hero p { margin: 0; max-width: 54ch; font-size: 1.1rem; color: #b9cbdb; }
.stats { display: flex; gap: 14px; margin-top: 30px; flex-wrap: wrap; }
.stat { min-width: 150px; padding: 14px 18px; border: 1px solid rgba(255,255,255,.14); border-radius: 12px; background: rgba(255,255,255,.06); }
.stat b { display: block; font-size: 1.6rem; line-height: 1.15; font-weight: 800; }
.stat span { font-size: .88rem; color: #a6bccf; }

/* the two columns: centred, and pulled up over the hero */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"],
div[data-testid="stHorizontalBlock"] { padding: 0 max(20px, calc((100% - 1160px) / 2)); }
div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] { margin-top: -70px; position: relative; z-index: 5; }

/* form card */
div[data-testid="stForm"] { background: #fff; border: 1px solid #e1e8ef; border-radius: 18px; padding: 30px; box-shadow: 0 1px 2px rgba(15,42,67,.06), 0 8px 24px rgba(15,42,67,.06); }
.card-title { margin: 0 0 4px; font-size: 1.3rem; font-weight: 800; color: #0f2a43; letter-spacing: -.02em; }
.card-sub { margin: 0 0 18px; color: #6b8196; font-size: .96rem; }
div[data-testid="stForm"] label p { font-weight: 700; color: #0f2a43; font-size: .98rem; }
div[data-testid="stFormSubmitButton"] button { width: 100%; padding: 14px 20px; font-weight: 700; font-size: 1.04rem; color: #fff; background: #0f6fff; border: 0; border-radius: 12px; box-shadow: 0 6px 16px rgba(15,111,255,.28); }
div[data-testid="stFormSubmitButton"] button:hover { background: #0a55c4; color: #fff; }
div[data-testid="stFormSubmitButton"] button p { color: #fff; font-size: 1.04rem; font-weight: 700; }

/* about card */
.about { background: #fff; border: 1px solid #e1e8ef; border-radius: 18px; padding: 22px 26px; margin-top: 20px; box-shadow: 0 1px 2px rgba(15,42,67,.06), 0 8px 24px rgba(15,42,67,.06); }
.about h2 { margin: 0 0 14px; font-size: 1.05rem; font-weight: 800; color: #0f2a43; padding: 0; }
.row { display: flex; justify-content: space-between; gap: 16px; padding: 6px 0; font-size: .92rem; }
.row span { color: #6b8196; }
.row b { color: #0f2a43; text-align: right; }

.foot { margin-top: 28px; padding: 18px max(20px, calc((100% - 1160px) / 2)); border-top: 1px solid #e1e8ef; background: #fff; color: #6b8196; font-size: .86rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px 24px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
LOGO = ('<svg width="34" height="34" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#0f2a43"/>'
        '<path d="M4 17h6l3-7 5 13 3-6h7" fill="none" stroke="#45e0b4" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')

st.markdown(f"""
<div class="nav">
  <div class="brand">{LOGO} CardioCheck</div>
  <div class="nav-note">Heart disease risk prediction</div>
</div>
<div class="hero">
  <h1>Predict heart disease from four clinical inputs</h1>
  <p>Enter a patient's sex, chest pain type, exercise response and ST depression. A trained Support Vector Classifier returns a prediction in under a second.</p>
  <div class="stats">
    <div class="stat"><b>{round(accuracy * 100)}%</b><span>Test accuracy</span></div>
    <div class="stat"><b>{total_patients}</b><span>Patients in the data</span></div>
    <div class="stat"><b>4</b><span>Clinical inputs</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Result monitor (an animated ECG drawn as inline SVG)
# ----------------------------------------------------------------------
def ecg_path(abnormal):
    if abnormal:
        beat = [(0,80),(14,80),(20,73),(26,80),(34,80),(40,88),(46,26),(52,104),(58,94),(78,94),(88,104),(98,94),(110,82),(120,80)]
    else:
        beat = [(0,80),(14,80),(20,73),(26,80),(34,80),(40,88),(46,26),(52,104),(58,80),(76,80),(86,66),(96,80),(120,80)]
    pts = [(x + i * 120, y) for i in range(5) for x, y in beat]
    return "M" + " L".join(f"{x} {y}" for x, y in pts)


def monitor_html(state, label, detail, chips):
    colors = {"idle": "#4b6a7f", "likely": "#ff6f61", "unlikely": "#45e0b4", "error": "#ffc65c"}
    color = colors[state]
    path = "M0 80 H600" if state in ("idle", "error") else ecg_path(state == "likely")
    anim = "animation: draw 1.6s cubic-bezier(.3,.1,.3,1) forwards;" if state in ("likely", "unlikely") else ""
    dash = "stroke-dasharray:1; stroke-dashoffset:1;" if anim else ""
    glow = f"filter: drop-shadow(0 0 5px {color}88);" if state in ("likely", "unlikely") else ""
    status_text = {"idle": "Ready", "likely": "Complete", "unlikely": "Complete", "error": "Needs attention"}[state]
    chips_html = ""
    if chips:
        chips_html = ('<div class="summary"><h3>Inputs used</h3><div class="chips">'
                      + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div></div>")
    return f"""
<style>
  html, body {{ background: #f4f6f9; }}
  body {{ margin: 0; font-family: "Plus Jakarta Sans", system-ui, sans-serif; }}
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
  .monitor {{ background: #0b1f33; color: #e8f1f8; border-radius: 18px; padding: 18px 18px 26px; box-shadow: 0 8px 24px rgba(15,42,67,.12); }}
  .head {{ display: flex; justify-content: space-between; padding: 2px 4px 14px; font-size: .86rem; color: #8fa7bc; font-weight: 600; }}
  .live {{ display: inline-flex; align-items: center; gap: 8px; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {color}; box-shadow: 0 0 0 4px {color}2e; }}
  .screen {{ border-radius: 12px; overflow: hidden; border: 1px solid #1a3a55; background-color: #07131f;
    background-image: linear-gradient(rgba(69,224,180,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(69,224,180,.07) 1px, transparent 1px);
    background-size: 24px 24px; }}
  svg {{ display: block; width: 100%; height: 170px; }}
  path {{ fill: none; stroke: {color}; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; {dash} {anim} {glow} }}
  @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
  .readout {{ padding: 20px 6px 0; }}
  .status {{ margin: 0 0 8px; font-size: 2rem; line-height: 1.15; font-weight: 800; letter-spacing: -.02em; color: {color if state != "idle" else "#e8f1f8"}; }}
  .detail {{ margin: 0; color: #b4c6d6; font-size: .97rem; max-width: 46ch; line-height: 1.5; }}
  .summary {{ margin: 20px 6px 0; padding-top: 16px; border-top: 1px solid #1d3a52; }}
  .summary h3 {{ margin: 0 0 10px; font-size: .84rem; font-weight: 600; color: #8fa7bc; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{ padding: 6px 12px; font-size: .86rem; font-weight: 600; color: #dbe8f3; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 999px; }}
  @media (prefers-reduced-motion: reduce) {{ path {{ animation: none !important; stroke-dashoffset: 0 !important; }} }}
</style>
<div class="monitor">
  <div class="head"><span class="live"><span class="dot"></span>{status_text}</span><span>Prediction</span></div>
  <div class="screen"><svg viewBox="0 0 600 160" preserveAspectRatio="none"><path d="{path}" pathLength="1"/></svg></div>
  <div class="readout"><p class="status">{label}</p><p class="detail">{detail}</p></div>
  {chips_html}
</div>"""


# ----------------------------------------------------------------------
# Page body: form (left) and result (right)
# ----------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.form("patient_form"):
        st.markdown('<p class="card-title">Patient details</p><p class="card-sub">All four answers are required.</p>',
                    unsafe_allow_html=True)
        sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
        cp_label = st.selectbox("Chest pain type", list(CHEST_PAIN.keys()), index=3,
                                help="1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 no symptoms")
        exang_label = st.radio("Chest pain during exercise", ["No", "Yes"], horizontal=True)
        oldpeak = st.slider("ST depression", min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
                            help="How far the ECG line drops below its baseline after exercise. Typical values are 0 to 3.")
        submitted = st.form_submit_button("Run prediction", use_container_width=True)

with right:

    if submitted:
        values = {
            "sex": 1 if sex_label == "Male" else 0,
            "cp": CHEST_PAIN[cp_label],
            "exang": 1 if exang_label == "Yes" else 0,
            "oldpeak": float(oldpeak),
        }
        result = int(model.predict(pd.DataFrame([values])[FEATURES])[0])
        chips = [sex_label, cp_label if cp_label != "No symptoms" else "No chest pain",
                 f"Exercise chest pain: {exang_label.lower()}", f"ST depression {oldpeak:.1f}"]
        if result == 1:
            html = monitor_html("likely", "Heart disease likely",
                                "These inputs match patients who were found to have heart disease. A clinician can confirm with further tests.",
                                chips)
        else:
            html = monitor_html("unlikely", "Heart disease not likely",
                                "These inputs match patients who were found to be healthy. Regular check-ups are still recommended.",
                                chips)
    else:
        html = monitor_html("idle", "Awaiting input",
                            "Complete the form and run the prediction. The result appears here.", [])

    # st.iframe is the current way to show custom HTML; older Streamlit
    # versions only have components.html, so fall back to it.
    if hasattr(st, "iframe"):
        st.iframe(html, height=500)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=500)

    st.markdown(f"""
    <div class="about">
      <h2>About the model</h2>
      <div class="row"><span>Algorithm</span><b>Support Vector Classifier</b></div>
      <div class="row"><span>Training data</span><b>UCI Heart Disease, 4 hospitals</b></div>
      <div class="row"><span>Patients used</span><b>{total_patients}</b></div>
      <div class="row"><span>Held-out test accuracy</span><b>{accuracy * 100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="foot">
  <span>CardioCheck. Predictions are informational and are not a medical diagnosis.</span>
  <span>Data: UCI Heart Disease Databases</span>
</div>
""", unsafe_allow_html=True)
