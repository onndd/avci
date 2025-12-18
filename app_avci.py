
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os
import sys

# Add current directory to path if needed
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from config_avci import TARGETS, WINDOWS, CARD_THRESHOLDS
from features_avci import extract_features
# from data_avci import load_data # We will import inside try/except block or use it safely

# Page Config
st.set_page_config(page_title="AVCI - High X Hunter", layout="wide", page_icon="🦅")

# CSS for Dark Mode & Cards
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .card-safe { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #2e7d32; text-align: center; }
    .card-risk { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #c62828; text-align: center; }
    .card-neutral { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #555; text-align: center; color: #555;}
    .card-missing { background-color: #121212; padding: 20px; border-radius: 10px; border: 1px dashed #444; text-align: center; color: #444; opacity: 0.7; }
    .card-gold { background-color: #2a2a10; padding: 20px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; color: #ffd700; animation: blink 1s infinite; }
    
    @keyframes blink { 50% { border-color: #fff; } }
</style>
""", unsafe_allow_html=True)

st.title("🦅 AVCI - Yüksek Oran İstihbarat Sistemi")

# Sidebar
st.sidebar.header("Ayarlar")
refresh_rate = st.sidebar.slider("Yenileme Hızı (sn)", 1, 10, 2)
auto_refresh = st.sidebar.checkbox("Otomatik Yenile", value=True)

# --- 1. Robust Data Loading ---
def robust_load_data(limit=200):
    """Safely load data with error handling."""
    try:
        from data_avci import load_data
        
        # Check if DB file exists
        if not os.path.exists('jetx.db'):
            st.error("⚠️ Veritabanı Bulunamadı: 'jetx.db' yok.")
            return None
            
        df = load_data('jetx.db', limit=limit)
        
        if df is None or len(df) == 0:
            st.warning("⚠️ Veritabanı boş veya okunamadı.")
            return None
            
        return df
    except Exception as e:
        st.error(f"⚠️ Veri Yükleme Hatası: {str(e)}")
        return None

# --- 2. Robust Model Loading ---
@st.cache_resource
def load_models_safely():
    """Load models and track missing ones."""
    models = {}
    missing_models = []
    
    if not os.path.exists('models'):
        # Create models dir if not exists to avoid strict errors, though models won't be there
        try: os.makedirs('models') 
        except: pass
        
    for t in TARGETS:
        model_name = f'avci_lgbm_{str(t).replace(".","_")}.txt'
        model_path = os.path.join('models', model_name)
        
        if os.path.exists(model_path):
            try:
                models[t] = lgb.Booster(model_file=model_path)
            except Exception as e:
                print(f"Error loading model {t}x: {e}")
                missing_models.append(t)
        else:
            missing_models.append(t)
            
    return models, missing_models

# --- Main App Logic Wrapped for Safety ---
try:
    models, missing_models_list = load_models_safely()
    
    df = robust_load_data(limit=200)

    if df is not None:
        # --- 3. Safe Feature Extraction ---
        try:
            df_feat = extract_features(df, windows=WINDOWS)
            
            # Check if we have enough data for at least one prediction row
            if len(df_feat) > 0:
                last_row = df_feat.iloc[[-1]]
                
                # Make Predictions
                current_probs = {}
                for t in TARGETS:
                    if t in models:
                        try:
                            # Verify features match
                            required_feats = models[t].feature_name()
                            # Check if all required features exist in df_feat columns
                            missing_cols = [f for f in required_feats if f not in df_feat.columns]
                            
                            if missing_cols:
                                st.warning(f"⚠️ Model ({t}x) Uyuşmazlığı: {missing_cols[:3]}... eksik.")
                                current_probs[t] = -1.0
                            else:
                                prob = models[t].predict(last_row)[0]
                                current_probs[t] = prob
                        except Exception as e:
                            # Model exists but prediction failed (feature mismatch?)
                            current_probs[t] = -1.0 # Error code
                            print(f"Prediction error for {t}x: {e}")
                    else:
                        current_probs[t] = None # Feature not available
            else:
                st.info("ℹ️ Yeterli veri yok (Özellik çıkarımı için daha fazla geçmiş veri lazım).")
                current_probs = {}
                df_feat = None # Signal UI to show placeholders

        except Exception as e:
            st.error(f"⚠️ Özellik Çıkarımı Hatası: {e}")
            df_feat = None
            current_probs = {}

        # --- Dashboard Layout ---
        col_radar, col_targets = st.columns([1, 3])

        with col_radar:
            st.subheader("Radar")
            if len(df) > 0:
                last_val = df['value'].iloc[-1]
                st.metric("Son Gelen", f"{last_val}x")
                
                if len(df) >= 5:
                    trend = df['value'].tail(5).mean()
                    st.write(f"Trend (5): {trend:.2f}x")
            else:
                st.write("Veri bekleniyor...")

        with col_targets:
            st.subheader("Hedef Kartları")
            
            # Divide into rows of 3 or 4 for better mobile view if needed
            cols = st.columns(len(TARGETS))
            
            for idx, t in enumerate(TARGETS):
                start_prob = current_probs.get(t, None) if df_feat is not None else None
                
                status = "Bekliyor"
                css_class = "card-neutral"
                prob_str = "%0.0"
                
                if start_prob is None:
                    # Model Missing or Data Missing
                    status = "Model Yok" if t in missing_models_list else "Veri Yok"
                    css_class = "card-missing"
                    prob_str = "--"
                elif start_prob == -1.0:
                    status = "HATA"
                    css_class = "card-risk"
                    prob_str = "ERR"
                else:
                    # We have a valid probability
                    prob_val = float(start_prob)
                    prob_str = f"%{prob_val*100:.1f}"
                    
                    # Get thresholds for this target or default
                    thresholds = CARD_THRESHOLDS.get(t, CARD_THRESHOLDS['DEFAULT'])
                    gold_thr = thresholds['GOLD']
                    risk_thr = thresholds['RISK']

                    if prob_val > gold_thr:
                        css_class = "card-gold" if t >= 5.0 else "card-safe"
                        status = "YAKALA!"
                    elif prob_val > risk_thr:
                        css_class = "card-risk"
                        status = "İzle"
                    else:
                        css_class = "card-neutral"
                        status = "Bekliyor"
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="{css_class}">
                        <h4>{t}x</h4>
                        <p>{status}</p>
                        <small>{prob_str}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
    else:
        # DF is None (DB Error handled in robust_load_data)
        st.info("Veritabanı bağlantısı bekleniyor... (Lütfen 'Colab' tarafının çalıştığından emin olun)")

    st.write("---")
    st.caption(f"Avcı Modeli v0.1.5 | Sniper Mode Active (100x @ {CARD_THRESHOLDS.get(100.0, {'GOLD':0}).get('GOLD', 0)*100:.0f}%)")

    # --- 4. Safe Auto Refresh ---
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

except Exception as main_e:
    # Catch-all for top level errors to prevent White Screen of Death
    st.error(f"🛑 Kritik Uygulama Hatası: {main_e}")
    if auto_refresh:
        st.warning("⚠️ Otomatik yenileme 10 saniye duraklatıldı...")
        time.sleep(10)
        st.rerun()
