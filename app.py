"""
CardioCheck - Heart Disease Prediction (Flask web app)
======================================================
Everything for the website lives in this one file: the server, the model
loading and the full page design (HTML, CSS and JavaScript).

  GET  /          -> the web page
  POST /predict   -> takes the 4 inputs, returns the prediction (JSON)
  GET  /health    -> health check used by the cloud host

Run locally:   python app.py
Run in cloud:  gunicorn app:app
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "svc_trained_model.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

# The same 4 attributes (and order) used for training
FEATURES = ["sex", "cp", "exang", "oldpeak"]

app = Flask(__name__)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def load_model():
    """Load the saved model; retrain automatically if it is missing or was
    saved with a different scikit-learn version."""
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


model = load_model()


def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return {}


def parse_input(data):
    """Validate the 4 answers. Returns (values, error_message)."""
    try:
        sex = int(data.get("sex"))
        cp = int(data.get("cp"))
        exang = int(data.get("exang"))
        oldpeak = float(data.get("oldpeak"))
    except (TypeError, ValueError):
        return None, "Please answer all four questions."

    if sex not in (0, 1):
        return None, "Choose male or female."
    if cp not in (1, 2, 3, 4):
        return None, "Choose one of the four chest pain types."
    if exang not in (0, 1):
        return None, "Choose yes or no for exercise-induced chest pain."
    if not -3.0 <= oldpeak <= 7.0:
        return None, "ST depression must be between -3 and 7."

    return {"sex": sex, "cp": cp, "exang": exang, "oldpeak": oldpeak}, None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE, metrics=load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    values, error = parse_input(request.get_json(silent=True) or {})
    if error:
        return jsonify({"error": error}), 400

    user_input = pd.DataFrame([values])[FEATURES]
    result = int(model.predict(user_input)[0])

    return jsonify({
        "prediction": result,
        "label": "Heart disease likely" if result == 1 else "Heart disease not likely",
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ----------------------------------------------------------------------
# Front end (HTML + CSS + JavaScript)
# ----------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CardioCheck | Heart disease prediction</title>
<meta name="description" content="Predict heart disease likelihood from four clinical inputs using a trained Support Vector Classifier.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230f2a43'/%3E%3Cpath d='M4 17h6l3-7 5 13 3-6h7' fill='none' stroke='%2345e0b4' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #f4f6f9;
  --surface: #ffffff;
  --ink: #0f2a43;
  --ink-2: #3d566e;
  --muted: #6b8196;
  --line: #e1e8ef;
  --line-strong: #c9d5e0;
  --brand: #0f6fff;
  --brand-dark: #0a55c4;
  --brand-soft: #e8f0ff;

  --navy: #0b1f33;
  --navy-2: #12304b;
  --ok: #45e0b4;
  --bad: #ff6f61;
  --warn: #ffc65c;

  --font: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --shadow: 0 1px 2px rgba(15, 42, 67, .06), 0 8px 24px rgba(15, 42, 67, .06);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font);
  font-size: 16px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { width: min(1160px, 100% - 40px); margin-inline: auto; }

/* ---------- navigation ---------- */
.nav { background: var(--surface); border-bottom: 1px solid var(--line); }
.nav .wrap { display: flex; align-items: center; justify-content: space-between; height: 68px; }
.brand { display: flex; align-items: center; gap: 12px; font-weight: 800; font-size: 1.2rem; letter-spacing: -.01em; }
.brand svg { display: block; }
.nav-note { color: var(--muted); font-size: .92rem; font-weight: 500; }

/* ---------- hero ---------- */
.hero {
  background:
    radial-gradient(900px 340px at 85% -10%, rgba(15,111,255,.35), transparent 70%),
    linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%);
  color: #fff;
  padding: 56px 0 120px;
  position: relative;
  overflow: hidden;
}
.hero h1 {
  margin: 0 0 16px;
  max-width: 21ch;
  font-size: clamp(2.2rem, 5.2vw, 3.7rem);
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -.03em;
}
.hero p { margin: 0; max-width: 54ch; font-size: 1.12rem; color: #b9cbdb; }

.stats { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 34px; }
.stat {
  min-width: 150px;
  padding: 14px 18px;
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 12px;
  background: rgba(255,255,255,.06);
}
.stat b { display: block; font-size: 1.6rem; line-height: 1.15; font-weight: 800; letter-spacing: -.02em; }
.stat span { font-size: .88rem; color: #a6bccf; }

.hero-line { position: absolute; left: 0; right: 0; bottom: 0; height: 90px; opacity: .5; pointer-events: none; }
.hero-line path { fill: none; stroke: var(--ok); stroke-width: 2; vector-effect: non-scaling-stroke; }

/* ---------- main grid ---------- */
.main {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
  margin-top: -76px;
  padding-bottom: 48px;
  position: relative;
}
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: var(--shadow);
}
.form-card { padding: 32px; }
.card-title { margin: 0 0 4px; font-size: 1.3rem; font-weight: 800; letter-spacing: -.02em; }
.card-sub { margin: 0 0 26px; color: var(--muted); font-size: .96rem; }

/* ---------- form ---------- */
fieldset { border: 0; margin: 0 0 24px; padding: 0; min-width: 0; }
legend { padding: 0; margin-bottom: 10px; font-weight: 700; font-size: .98rem; }
.hint { margin: -4px 0 12px; color: var(--muted); font-size: .9rem; max-width: 62ch; }

.seg input, .choices input { position: absolute; opacity: 0; width: 1px; height: 1px; pointer-events: none; }

.seg { display: flex; padding: 4px; gap: 4px; background: var(--bg); border: 1px solid var(--line); border-radius: 12px; max-width: 340px; }
.seg label { flex: 1; padding: 10px 16px; text-align: center; font-weight: 600; font-size: .96rem; color: var(--ink-2); border-radius: 9px; cursor: pointer; transition: background .15s, color .15s, box-shadow .15s; }
.seg label:hover { color: var(--ink); }
.seg input:checked + label { background: var(--surface); color: var(--brand-dark); box-shadow: 0 1px 3px rgba(15,42,67,.15), 0 0 0 1px var(--line-strong); }

.choices { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.choices label { display: block; height: 100%; padding: 14px 16px 14px 46px; position: relative; border: 1px solid var(--line-strong); border-radius: 12px; cursor: pointer; transition: border-color .15s, background .15s; }
.choices label::before { content: ""; position: absolute; left: 16px; top: 17px; width: 18px; height: 18px; border: 2px solid var(--line-strong); border-radius: 50%; background: #fff; transition: all .15s; }
.choices label b { display: block; font-weight: 700; font-size: .96rem; }
.choices label span { display: block; margin-top: 2px; font-size: .84rem; line-height: 1.4; color: var(--muted); }
.choices label:hover { border-color: #8fb4f5; }
.choices input:checked + label { border-color: var(--brand); background: var(--brand-soft); }
.choices input:checked + label::before { border-color: var(--brand); box-shadow: inset 0 0 0 3px #fff; background: var(--brand); }

.slider { display: flex; align-items: center; gap: 16px; }
input[type="range"] { flex: 1; height: 32px; accent-color: var(--brand); }
input[type="number"] { width: 96px; padding: 10px; font: inherit; font-weight: 700; text-align: center; color: var(--ink); background: #fff; border: 1px solid var(--line-strong); border-radius: 10px; }
.scale { display: flex; justify-content: space-between; margin: -2px 112px 0 2px; font-size: .78rem; color: var(--muted); }

.btn { width: 100%; margin-top: 4px; padding: 16px 20px; font: inherit; font-weight: 700; font-size: 1.04rem; color: #fff; background: var(--brand); border: 0; border-radius: 12px; cursor: pointer; box-shadow: 0 6px 16px rgba(15,111,255,.28); transition: background .15s, transform .05s; }
.btn:hover { background: var(--brand-dark); }
.btn:active { transform: translateY(1px); }
.btn:disabled { background: #7fa9ee; box-shadow: none; cursor: progress; }

.seg input:focus-visible + label, .choices input:focus-visible + label, input:focus-visible, .btn:focus-visible { outline: 3px solid rgba(15,111,255,.45); outline-offset: 2px; }

/* ---------- result column ---------- */
.side { display: grid; gap: 24px; position: sticky; top: 24px; }

.monitor { background: var(--navy); color: #e8f1f8; border-radius: 18px; padding: 18px 18px 26px; box-shadow: var(--shadow); }
.monitor-head { display: flex; align-items: center; justify-content: space-between; padding: 2px 4px 14px; font-size: .86rem; color: #8fa7bc; font-weight: 600; }
.live { display: inline-flex; align-items: center; gap: 8px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #4b6a7f; }
.monitor[data-state="unlikely"] .dot { background: var(--ok); box-shadow: 0 0 0 4px rgba(69,224,180,.18); }
.monitor[data-state="likely"] .dot { background: var(--bad); box-shadow: 0 0 0 4px rgba(255,111,97,.18); }
.monitor[data-state="loading"] .dot { background: var(--ok); animation: blink .8s infinite alternate; }
.monitor[data-state="error"] .dot { background: var(--warn); }
@keyframes blink { from { opacity: .25; } to { opacity: 1; } }

.screen { border-radius: 12px; overflow: hidden; border: 1px solid #1a3a55; background-color: #07131f;
  background-image: linear-gradient(rgba(69,224,180,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(69,224,180,.07) 1px, transparent 1px);
  background-size: 24px 24px; }
#trace { display: block; width: 100%; height: 180px; }
#trace path { fill: none; stroke: #4b6a7f; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
.monitor[data-state="unlikely"] #trace path { stroke: var(--ok); filter: drop-shadow(0 0 5px rgba(69,224,180,.55)); }
.monitor[data-state="likely"] #trace path { stroke: var(--bad); filter: drop-shadow(0 0 5px rgba(255,111,97,.55)); }
.monitor[data-state="loading"] #trace path { stroke: var(--ok); animation: blink .8s infinite alternate; }

.readout { padding: 22px 6px 0; }
.status { margin: 0 0 8px; font-size: clamp(1.6rem, 3.2vw, 2.1rem); line-height: 1.15; font-weight: 800; letter-spacing: -.02em; }
.monitor[data-state="unlikely"] .status { color: var(--ok); }
.monitor[data-state="likely"] .status { color: var(--bad); }
.monitor[data-state="error"] .status { color: var(--warn); }
.detail { margin: 0; color: #b4c6d6; max-width: 46ch; font-size: .98rem; }

.summary { margin: 22px 6px 0; padding-top: 18px; border-top: 1px solid #1d3a52; display: none; }
.summary.show { display: block; }
.summary h3 { margin: 0 0 10px; font-size: .84rem; font-weight: 600; color: #8fa7bc; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { padding: 6px 12px; font-size: .86rem; font-weight: 600; color: #dbe8f3; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 999px; }

.about { padding: 24px 26px; }
.about h2 { margin: 0 0 14px; font-size: 1.05rem; font-weight: 800; letter-spacing: -.01em; }
.about dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 10px 20px; font-size: .92rem; }
.about dt { color: var(--muted); font-weight: 500; }
.about dd { margin: 0; font-weight: 600; text-align: right; }

/* ---------- footer ---------- */
.foot { border-top: 1px solid var(--line); background: var(--surface); }
.foot .wrap { display: flex; flex-wrap: wrap; gap: 8px 24px; justify-content: space-between; padding: 22px 0; color: var(--muted); font-size: .86rem; }

/* ---------- responsive ---------- */
@media (max-width: 920px) {
  .main { grid-template-columns: 1fr; margin-top: -64px; }
  .side { position: static; }
  .hero { padding: 40px 0 104px; }
}
@media (max-width: 560px) {
  .choices { grid-template-columns: 1fr; }
  .form-card { padding: 24px 18px; }
  .nav-note { display: none; }
  .stats { gap: 8px; flex-wrap: nowrap; }
  .stat { min-width: 0; flex: 1 1 0; padding: 12px; }
  .stat b { font-size: 1.3rem; }
  .stat span { font-size: .78rem; }
  .scale { margin-right: 108px; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; scroll-behavior: auto !important; }
}
</style>
</head>
<body>

<nav class="nav">
  <div class="wrap">
    <div class="brand">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="#0f2a43"/><path d="M4 17h6l3-7 5 13 3-6h7" fill="none" stroke="#45e0b4" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
      CardioCheck
    </div>
    <div class="nav-note">Heart disease risk prediction</div>
  </div>
</nav>

<header class="hero">
  <div class="wrap">
    <h1>Predict heart disease from four clinical inputs</h1>
    <p>Enter a patient's sex, chest pain type, exercise response and ST depression. A trained Support Vector Classifier returns a prediction in under a second.</p>
    <div class="stats">
      <div class="stat"><b>{% if metrics.accuracy %}{{ (metrics.accuracy * 100)|round|int }}%{% else %}83%{% endif %}</b><span>Test accuracy</span></div>
      <div class="stat"><b>{{ metrics.total_patients or 857 }}</b><span>Patients in the data</span></div>
      <div class="stat"><b>4</b><span>Clinical inputs</span></div>
    </div>
  </div>
  <svg class="hero-line" viewBox="0 0 1200 90" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 50 H180 l12 -6 l12 6 H250 l10 12 l14 -52 l14 70 l10 -30 H430 l18 -14 l18 14 H620 l12 -6 l12 6 H690 l10 12 l14 -52 l14 70 l10 -30 H870 l18 -14 l18 14 H1060 l12 -6 l12 6 H1130 l10 12 l14 -52 l14 70 l10 -30 H1200"/>
  </svg>
</header>

<main class="wrap main">

  <!-- ============ Form ============ -->
  <form id="form" class="card form-card" novalidate>
    <h2 class="card-title">Patient details</h2>
    <p class="card-sub">All four answers are required.</p>

    <fieldset>
      <legend>Sex</legend>
      <div class="seg">
        <input type="radio" name="sex" id="sex-1" value="1" checked><label for="sex-1">Male</label>
        <input type="radio" name="sex" id="sex-0" value="0"><label for="sex-0">Female</label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Chest pain type</legend>
      <div class="choices">
        <input type="radio" name="cp" id="cp-1" value="1">
        <label for="cp-1"><b>Typical angina</b><span>Pressure that comes with effort and eases with rest</span></label>

        <input type="radio" name="cp" id="cp-2" value="2">
        <label for="cp-2"><b>Atypical angina</b><span>Chest pain that partly fits the typical pattern</span></label>

        <input type="radio" name="cp" id="cp-3" value="3">
        <label for="cp-3"><b>Non-anginal pain</b><span>Chest pain not caused by the heart</span></label>

        <input type="radio" name="cp" id="cp-4" value="4" checked>
        <label for="cp-4"><b>No symptoms</b><span>No chest pain (asymptomatic)</span></label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Chest pain during exercise</legend>
      <div class="seg">
        <input type="radio" name="exang" id="ex-0" value="0" checked><label for="ex-0">No</label>
        <input type="radio" name="exang" id="ex-1" value="1"><label for="ex-1">Yes</label>
      </div>
    </fieldset>

    <fieldset>
      <legend>ST depression</legend>
      <p class="hint">How far the ECG line drops below its baseline after exercise, as shown on a stress test report. Typical values are 0 to 3.</p>
      <div class="slider">
        <input type="range" id="oldpeak-range" min="-3" max="7" step="0.1" value="1" aria-label="ST depression slider">
        <input type="number" id="oldpeak" name="oldpeak" min="-3" max="7" step="0.1" value="1" aria-label="ST depression value">
      </div>
      <div class="scale" aria-hidden="true"><span>-3</span><span>7</span></div>
    </fieldset>

    <button type="submit" id="submit" class="btn">Run prediction</button>
  </form>

  <!-- ============ Result ============ -->
  <div class="side">
    <section class="monitor" id="monitor" data-state="idle" aria-live="polite">
      <div class="monitor-head">
        <span class="live"><span class="dot"></span><span id="live-text">Ready</span></span>
        <span>Prediction</span>
      </div>
      <div class="screen">
        <svg id="trace" viewBox="0 0 600 160" preserveAspectRatio="none" role="img" aria-label="Heart rhythm trace">
          <path id="trace-path" d="M0 80 H600"/>
        </svg>
      </div>
      <div class="readout">
        <p class="status" id="status">Awaiting input</p>
        <p class="detail" id="detail">Complete the form and run the prediction. The result appears here.</p>
      </div>
      <div class="summary" id="summary">
        <h3>Inputs used</h3>
        <div class="chips" id="chips"></div>
      </div>
    </section>

    <section class="card about">
      <h2>About the model</h2>
      <dl>
        <dt>Algorithm</dt><dd>Support Vector Classifier</dd>
        <dt>Training data</dt><dd>UCI Heart Disease, 4 hospitals</dd>
        <dt>Patients used</dt><dd>{{ metrics.total_patients or 857 }}</dd>
        <dt>Held-out test accuracy</dt><dd>{% if metrics.accuracy %}{{ (metrics.accuracy * 100)|round(1) }}%{% else %}83%{% endif %}</dd>
      </dl>
    </section>
  </div>
</main>

<footer class="foot">
  <div class="wrap">
    <span>CardioCheck. Predictions are informational and are not a medical diagnosis.</span>
    <span>Data: UCI Heart Disease Databases</span>
  </div>
</footer>

<script>
(function () {
  var form = document.getElementById("form");
  var range = document.getElementById("oldpeak-range");
  var number = document.getElementById("oldpeak");
  var monitor = document.getElementById("monitor");
  var statusEl = document.getElementById("status");
  var detailEl = document.getElementById("detail");
  var liveText = document.getElementById("live-text");
  var pathEl = document.getElementById("trace-path");
  var button = document.getElementById("submit");
  var summary = document.getElementById("summary");
  var chips = document.getElementById("chips");
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var CP = { "1": "Typical angina", "2": "Atypical angina", "3": "Non-anginal pain", "4": "No chest pain" };

  range.addEventListener("input", function () { number.value = range.value; });
  number.addEventListener("input", function () { if (number.value !== "") range.value = number.value; });

  /* A normal beat has a small P bump, a tall R spike and a gentle T hump.
     In the abnormal trace the line stays below baseline after the spike and
     the T wave flips down, which is how ST depression looks on a real ECG. */
  function beat(x, abnormal) {
    var pts = abnormal
      ? [[0,80],[14,80],[20,73],[26,80],[34,80],[40,88],[46,26],[52,104],[58,94],[78,94],[88,104],[98,94],[110,82],[120,80]]
      : [[0,80],[14,80],[20,73],[26,80],[34,80],[40,88],[46,26],[52,104],[58,80],[76,80],[86,66],[96,80],[120,80]];
    return pts.map(function (p) { return [p[0] + x, p[1]]; });
  }
  function buildPath(abnormal) {
    var all = [];
    for (var i = 0; i < 5; i++) all = all.concat(beat(i * 120, abnormal));
    return "M" + all.map(function (p) { return p[0] + " " + p[1]; }).join(" L");
  }
  function drawTrace(abnormal) {
    pathEl.setAttribute("d", buildPath(abnormal));
    if (reduceMotion) { pathEl.style.strokeDasharray = "none"; pathEl.style.strokeDashoffset = "0"; return; }
    var len = pathEl.getTotalLength();
    pathEl.style.transition = "none";
    pathEl.style.strokeDasharray = len;
    pathEl.style.strokeDashoffset = len;
    pathEl.getBoundingClientRect();
    pathEl.style.transition = "stroke-dashoffset 1.6s cubic-bezier(.3,.1,.3,1)";
    pathEl.style.strokeDashoffset = "0";
  }
  function flatLine() {
    pathEl.setAttribute("d", "M0 80 H600");
    pathEl.style.transition = "none";
    pathEl.style.strokeDasharray = "none";
    pathEl.style.strokeDashoffset = "0";
  }
  function showError(message) {
    flatLine();
    summary.classList.remove("show");
    monitor.dataset.state = "error";
    liveText.textContent = "Needs attention";
    statusEl.textContent = "Check your answers";
    detailEl.textContent = message;
  }
  function showChips(p) {
    var items = [
      p.sex === "1" ? "Male" : "Female",
      CP[p.cp],
      p.exang === "1" ? "Exercise chest pain: yes" : "Exercise chest pain: no",
      "ST depression " + Number(p.oldpeak).toFixed(1)
    ];
    chips.innerHTML = "";
    items.forEach(function (t) {
      var s = document.createElement("span");
      s.className = "chip";
      s.textContent = t;
      chips.appendChild(s);
    });
    summary.classList.add("show");
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var payload = {
      sex: form.elements["sex"].value,
      cp: form.elements["cp"].value,
      exang: form.elements["exang"].value,
      oldpeak: number.value
    };
    if (payload.oldpeak === "" || isNaN(Number(payload.oldpeak))) {
      showError("Enter a number for ST depression, for example 1.5.");
      return;
    }

    button.disabled = true;
    button.textContent = "Running";
    monitor.dataset.state = "loading";
    liveText.textContent = "Analysing";
    statusEl.textContent = "Reading the trace";
    detailEl.textContent = "The model is reviewing the four inputs.";
    summary.classList.remove("show");
    if (window.innerWidth <= 920) {
      monitor.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    }

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
    .then(function (res) { return res.json().then(function (d) { return { ok: res.ok, data: d }; }); })
    .then(function (r) {
      if (!r.ok) { showError(r.data.error || "Something went wrong. Try again."); return; }
      var likely = r.data.prediction === 1;
      monitor.dataset.state = likely ? "likely" : "unlikely";
      liveText.textContent = "Complete";
      statusEl.textContent = r.data.label;
      detailEl.textContent = likely
        ? "These inputs match patients who were found to have heart disease. A clinician can confirm with further tests."
        : "These inputs match patients who were found to be healthy. Regular check-ups are still recommended.";
      drawTrace(likely);
      showChips(payload);
    })
    .catch(function () { showError("Could not reach the server. Check your connection and try again."); })
    .then(function () { button.disabled = false; button.textContent = "Run prediction"; });
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
