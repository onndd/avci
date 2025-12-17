
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os
from config_avci import TARGETS, WINDOWS
from features_avci import extract_features
from data_avci import load_data # Simplified loader for demo

# Page Config
st.set_page_config(page_title="AVCI - High X Hunter", layout="wide", page_icon="🦅")

# CSS for Dark Mode & Cards
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .card-safe { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #2e7d32; text-align: center; }
    .card-risk { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #c62828; text-align: center; }
    .card-neutral { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #555; text-align: center; color: #555;}
    .card-gold { background-color: #2a2a10; padding: 20px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; color: #ffd700; animation: blink 1s infinite; }
    
    @keyframes blink { 50% { border-color: #fff; } }
</style>
""", unsafe_allow_html=True)

st.title("🦅 AVCI - Yüksek Oran İstihbarat Sistemi")

# Sidebar
st.sidebar.header("Ayarlar")
refresh_rate = st.sidebar.slider("Yenileme Hızı (sn)", 1, 10, 2)
auto_refresh = st.sidebar.checkbox("Otomatik Yenile", value=True)

# Load Models
@st.cache_resource
def load_models():
    models = {}
    for t in TARGETS:
        model_path = f'models/avci_lgbm_{str(t).replace(".","_")}.txt'
        if os.path.exists(model_path):
            models[t] = lgb.Booster(model_file=model_path)
    return models

models = load_models()

# Mock Data (In production, this would read live DB)
df = load_data('jetx.db', limit=200) 
df_feat = extract_features(df, windows=WINDOWS)
current_probs = {}

if models:
    last_row = df_feat.iloc[[-1]] # Last row for prediction
    for t, model in models.items():
        prob = model.predict(last_row)[0]
        current_probs[t] = prob

# Dashboard Layout
col_radar, col_targets = st.columns([1, 3])

with col_radar:
    st.subheader("Radar")
    last_val = df['value'].iloc[-1]
    st.metric("Son Gelen", f"{last_val}x")
    
    # Simple Trend Indicator
    trend = df['value'].tail(5).mean()
    st.write(f"Trend (5): {trend:.2f}x")

with col_targets:
    st.subheader("Hedef Kartları")
    cols = st.columns(len(TARGETS))
    
    for idx, t in enumerate(TARGETS):
        prob = current_probs.get(t, 0.0)
        
        # Visual Logic
        css_class = "card-neutral"
        status = "Bekliyor"
        
        # Thresholds (Simulated - in real app, use config)
        if prob > 0.85:
            css_class = "card-gold" if t >= 5.0 else "card-safe"
            status = "YAKALA!"
        elif prob > 0.60:
            css_class = "card-risk" # Warning / Watch
            status = "İzle"
            
        with cols[idx]:
            st.markdown(f"""
            <div class="{css_class}">
                <h3>{t}x</h3>
                <p>{status}</p>
                <small>%{prob*100:.1f}</small>
            </div>
            """, unsafe_allow_html=True)

st.write("---")
st.caption("Avcı Modeli v1.0 | Sadece İstatistiksel Tahmindir.")

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
