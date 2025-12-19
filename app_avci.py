
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
# --- CSS: High Visibility Cards ---
st.markdown("""
<style>
    .card-container {
        display: flex;
        flex-direction: column;
        align_items: center;
        justify-content: center;
        padding: 15px;
        margin: 5px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.1s;
    }
    .card-container:hover { transform: scale(1.02); }
    
    .status-badge {
        font-size: 14px;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        margin-top: 5px;
        display: inline-block;
    }
    
    /* STATES */
    .card-action { 
        background: linear-gradient(135deg, #1b5e20, #2e7d32); 
        border: 2px solid #00e676;
        color: white;
    }
    .card-action h3 { color: #fff !important; }
    .card-action .prob { color: #00e676 !important; font-size: 28px !important; font-weight: 800; }
    
    .card-wait { 
        background-color: #212121; 
        border: 1px solid #424242;
        color: #e0e0e0;
    }
    .card-wait .prob { color: #9e9e9e; font-size: 20px; }
    
    .card-risk { 
        background-color: #261111; 
        border: 1px solid #b71c1c;
        color: #ffcdd2;
    }
    .card-risk .prob { color: #ef5350; font-size: 20px; }
    
    .card-err { background-color: #000; border: 1px dashed red; color: red; }
    
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Sayfalar", ["🦅 Canlı Takip", "💾 Veri Yükleme"])

st.sidebar.markdown("---")
# Settings common to both or specific to Dashboard
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

# --- PAGE 1: Canlı Takip (Dashboard) ---
if page == "🦅 Canlı Takip":
    try:
        # --- Single Data Entry Logic (Callback) ---
        if 'dash_input' not in st.session_state:
            st.session_state.dash_input = ""

        def submit_data():
            val_str = st.session_state.dash_input
            if val_str:
                try:
                    val = float(val_str.lower().replace('x', '').strip())
                    from data_avci import save_new_data
                    if save_new_data([val]):
                        st.toast(f"✅ Eklendi: {val}x")
                    st.session_state.dash_input = "" # Clear input
                except Exception as e:
                    st.toast(f"❌ Hata: {e}")
                    # Keep input so user can fix it? Or clear? Let's keep it if error, but Streamlit callback complexity...
                    # For simplicity, we just show error toast.

        # --- Single Data Entry UI ---
        with st.container():
             c1, c2 = st.columns([1, 3])
             with c1:
                 st.markdown("### ⚡ Hızlı Giriş")
                 st.text_input("Sonuç (örn: 1.50)", 
                              label_visibility="collapsed", 
                              placeholder="1.50", 
                              key="dash_input", 
                              on_change=submit_data)
        
        st.markdown("---")

        models, missing_models_list = load_models_safely()
        df = robust_load_data(limit=1000)

        if df is not None:
            # --- 3. Safe Feature Extraction ---
            try:
                df_feat = extract_features(df, windows=WINDOWS)
                
                if len(df_feat) > 0:
                    last_row = df_feat.iloc[[-1]]
                    
                    current_probs = {}
                    for t in TARGETS:
                        if t in models:
                            try:
                                # Ensure features match
                                row_for_pred = last_row.drop(columns=['id', 'value'], errors='ignore')
                                
                                required_feats = models[t].feature_name()
                                missing_cols = [f for f in required_feats if f not in row_for_pred.columns]
                                
                                if missing_cols:
                                    st.warning(f"⚠️ Model ({t}x) Uyuşmazlığı.")
                                    current_probs[t] = -1.0
                                else:
                                    prob = models[t].predict(row_for_pred)[0]
                                    current_probs[t] = prob
                            except Exception as e:
                                current_probs[t] = -1.0
                                print(f"Prediction error for {t}x: {e}")
                        else:
                            current_probs[t] = None
                else:
                    st.info("ℹ️ Yeterli veri yok.")
                    current_probs = {}
                    df_feat = None

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
                
                # Grid Layout: 2 Rows of 4 Columns (since we have 8 targets)
                # Row 1: 2.0, 3.0, 5.0, 10.0
                # Row 2: 20.0, 30.0, 40.0, 50.0
                
                # Split targets into chunks of 4
                rows = [TARGETS[i:i + 4] for i in range(0, len(TARGETS), 4)]
                
                for row_targets in rows:
                    cols = st.columns(len(row_targets))
                    for idx, t in enumerate(row_targets):
                        start_prob = current_probs.get(t, None) if df_feat is not None else None
                    

                    with cols[idx]:
                        # Determine Style
                        card_style = "card-wait"
                        status_text = "BEKLİYOR"
                        prob_display = "--"
                        
                        if start_prob is None:
                            status_text = "VERİ YOK"
                            card_style = "card-err"
                        elif start_prob == -1.0:
                            status_text = "HATA"
                            card_style = "card-err"
                            prob_display = "ERR"
                        else:
                            prob_val = float(start_prob)
                            prob_pct = prob_val * 100
                            prob_display = f"%{prob_pct:.1f}"
                            
                            thresholds = CARD_THRESHOLDS.get(t, CARD_THRESHOLDS['DEFAULT'])
                            
                            if prob_val > thresholds['GOLD']:
                                card_style = "card-action"
                                status_text = "🚀 YAKALA!"
                            elif prob_val > thresholds['RISK']:
                                card_style = "card-risk"
                                status_text = "RİSKLİ"
                            else:
                                card_style = "card-wait"
                                status_text = "ZAYIF"

                        st.markdown(f"""
                        <div class="card-container {card_style}">
                            <h3 style="margin:0; font-size: 20px;">{t}x</h3>
                            <div class="prob" style="padding: 5px 0;">{prob_display}</div>
                            <div class="status-badge" style="background: rgba(0,0,0,0.3);">{status_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
        else:
            st.info("Veritabanı bağlantısı bekleniyor... (Lütfen 'Colab' tarafının çalıştığından emin olun)")

    # --- Last Update Timestamp ---
    # --- Last Update Timestamp ---
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 12px; margin-top: 20px;">
            Son Güncelleme: <b>{now}</b> | Otomatik Yenileme: {'Aktif' if auto_refresh else 'Pasif'} ({refresh_rate}sn)
        </div>
        """, unsafe_allow_html=True)

        # --- 4. Safe Auto Refresh ---
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

    except Exception as main_e:
        st.error(f"🛑 Kritik Uygulama Hatası: {main_e}")
        if auto_refresh:
            time.sleep(10)
            st.rerun()

# --- PAGE 2: Veri Yükleme (Batch Upload) ---
elif page == "💾 Veri Yükleme":
    st.subheader("Toplu Veri Yükleme")
    st.info("Elinizdeki oyun geçmişi listesini buraya yapıştırın.")
    
    st.markdown("""
    **Format:**
    - Her satırda bir sonuç (örn: 2.01x)
    - **En üst satır = En Yeni El** (Sistem otomatik olarak tersine çevirip kaydeder)
    """)
    
    batch_text = st.text_area("Liste (Yapıştırın)", height=300)
    
    if st.button("💾 Kaydet ve Analize Başla"):
        if batch_text:
            try:
                values = []
                lines = batch_text.strip().split('\n')
                # Reverse Logic: Top = Newest -> Bottom = Oldest. 
                # DB Insert Order: Oldest -> Newest.
                for line in reversed(lines):
                    clean_line = line.lower().replace('x', '').strip()
                    if clean_line:
                        values.append(float(clean_line))
                
                if values:
                    from data_avci import save_new_data
                    if save_new_data(values):
                        st.success(f"✅ {len(values)} satır başarıyla eklendi!")
                        st.balloons()
            except Exception as e:
                st.error(f"Hata: {e}")
