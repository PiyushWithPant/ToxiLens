# ------------------------------------------------------------------------------------
#                               app.py — ToxiLens Streamlit App (Dark UI)
# ------------------------------------------------------------------------------------

import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
from model import ToxicANN

# ========================================= Load Model & Vectorizer ========================================= 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ToxicANN(INPUT_DIM=5000).to(device)
model.load_state_dict(torch.load("model/toxilens.pth", map_location=device))
model.eval()

vectorizer = joblib.load("data/preprocessed data/tfidf_vectorizer.pkl")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ========================================= Streamlit Config ========================================= 
st.set_page_config(page_title="ToxiLens 🔎", page_icon="🤖", layout="centered")

# ========================================= Custom CSS ========================================= 
st.markdown(
    """
    <style>
    /* overall app background and font */
    .stApp {
        background-color: #0b0f14 !important;
        color: #e6eef3;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }

    /* hide streamlit menu, header, footer (removes weird link/header) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Title styling (emoji included in markup) */

    .title { font-size: 3rem; font-weight: 800; text-align: center; color: #00b4d8; text-shadow: 0 0 15px rgba(0, 180, 216, 0.8); margin-bottom: 0.3rem; }
    .subtitle {
        text-align: center;
        color: #9fb8c9;
        margin-top: 6px;
        margin-bottom: 18px;
        font-size: 1.05rem;
    }

    /* center the content container a bit and limit width */
    .main > div[role="main"] > div {
        display: flex;
        justify-content: center;
    }
    .app-container {
        width: 820px;
    }

    /* Text area: larger, readable, nice border */
    .stTextArea > div > div > textarea, textarea[role="textbox"] {
        background-color: #0f1316 !important;
        color: #e6eef3 !important;
        border-radius: 10px !important;
        border: 1px solid #253238 !important;
        min-height: 200px !important;
        padding: 14px !important;
        font-size: 1rem !important;
    }

    /* Button: centered and styled */
    div.stButton {
        display:flex;
        justify-content:center;
        margin-top: 10px;
    }
    div.stButton > button {
        background-color: #33c2ff;
        color: #072028;
        border-radius: 10px;
        padding: 10px 26px;
        font-size: 1.05rem;
        font-weight: 700;
        border: none;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(51,194,255,0.12);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 18px;
        margin-top: 18px;
        box-shadow: 0 6px 22px rgba(0,0,0,0.6);
    }

    /* category lines */
    .cat-row {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:8px 0;
        border-bottom: 1px dashed rgba(255,255,255,0.02);
    }
    .cat-name { font-weight:600; color:#dfeef7; }
    .cat-score { font-weight:700; color:#cfeeff; }

    /* small helper text */
    .muted { color:#7b8f98; font-size:0.95rem; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================================= Header ========================================= 
st.markdown("<h1 class='title'>ToxiLens 🔎</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A smart classifier that detects multiple forms of toxicity in text ⚡</p>", unsafe_allow_html=True)

# ========================================= Input Section ========================================= 
user_input = st.text_area("", height=150, placeholder="Type something toxic... or not 😄")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess input
        X = vectorizer.transform([user_input])
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32).to(device)

        with torch.no_grad():
            binary_pred, multi_pred = model(X_tensor)
            binary_pred = torch.sigmoid(binary_pred).cpu().numpy()[0][0]
            multi_pred = torch.sigmoid(multi_pred).cpu().numpy()[0]

        binary_label = "Toxic" if binary_pred > 0.5 else "Non-toxic"
        binary_color = "#ff4d4d" if binary_label == "Toxic" else "#06d6a0"

        # ========================================= Output Section ========================================= 
        st.markdown(f"""
        <div class='result-box'>
            <h3 style='text-align:center; color:{binary_color};'>
                🧩 Overall Classification: <b>{binary_label}</b>
            </h3>
        """, unsafe_allow_html=True)

        st.progress(float(binary_pred) if binary_pred > 0.5 else 1 - float(binary_pred))

        st.markdown("### Category-level Toxicity Scores:")
        for label, score in zip(LABELS, multi_pred):
            bar_color = "🟥" if score > 0.5 else "🟩"
            st.write(f"{bar_color} **{label.capitalize()}** — {score:.2f}")
            st.progress(float(score))

        toxic_labels = [l for l, s in zip(LABELS, multi_pred) if s > 0.5]
        if toxic_labels:
            st.error(f"⚠️ Detected Toxic Categories: {', '.join(toxic_labels)}")
        else:
            st.success("✅ Clean — No toxicity detected!")

        st.markdown("</div>", unsafe_allow_html=True)
