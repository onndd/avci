import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Header
nb.cells.append(nbf.v4.new_markdown_cell("""# 🦅 AVCI - Yüksek Oran Avcısı Eğitim Paneli v0.2.2 (Precision Update)
**Yeni Mimari:**
- **Hedef Odak:** 3x, 5x, 10x, 20x, 50x (100x+ kaldırıldı).
- **Strateji:** "Her Şeye Oyna" yerine "Keskin Nişancı" (Precision) odaklı eğitim.
- **Görselleştirme:** Basitleştirilmiş Kâr/Zarar grafikleri.
"""))

# Cell 2: Setup
nb.cells.append(nbf.v4.new_code_cell("""# 1. GitHub Deposunu Çek ve Kurulum Yap (Sessiz Kurulum)
import os
import sys
import shutil

print("🚀 Kurulum Başlatılıyor...")

# 1. Kök dizine (Colab varsayılanı) dön
try:
    os.chdir('/content')
except:
    pass

GITHUB_REPO = "https://github.com/onndd/avci.git"
PROJECT_DIR = "avci"

# 2. Varsa eski projeyi değil, sadece bu projeyi kontrol et
if not os.path.exists(PROJECT_DIR):
    print(f"📥 Proje indiriliyor: {GITHUB_REPO}")
    !git clone $GITHUB_REPO
else:
    print("✅ Proje klasörü mevcut. Güncelleniyor...")
    os.chdir(PROJECT_DIR)
    !git fetch --all
    !git reset --hard origin/main
    os.chdir('..')

# 3. Çalışma dizinine gir ve YOLU EKLE
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    print(f"✅ Çalışma Dizini Ayarlandı: {os.getcwd()}")
else:
    print(f"❌ HATA: {PROJECT_DIR} klasörü bulunamadı!")

# 4. Dosyaları listele (Hata ayıklama için)
print("📂 Klasör İçeriği:")
!ls -F

!nvidia-smi
!pip install optuna streamlit matplotlib pandas tensorflow hmmlearn plotly lightgbm
print("✅ Kurulum ve Hazırlık Tamamlandı!")"""))

# Cell 3: Load Modules
nb.cells.append(nbf.v4.new_code_cell("""# 2. Veri Yükleme ve Ayarlar
import optuna
import logging
import sys
import os

optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Mevcut dizini path'e ekle (Garanti olsun)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    import train_avci
    import models_avci
    import model_ae_avci
    import model_hmm_avci
    load_and_prep = train_avci.load_and_prep
    visualize_performance = train_avci.visualize_performance
    optimize_target = train_avci.optimize_target
    train_target_final = train_avci.train_target_final
    train_meta_model = train_avci.train_meta_model
    print("✅ Modüller başarıyla yüklendi!")
except ImportError as e:
    print(f"❌ KRİTİK HATA: Modüller yüklenemedi. Detay: {e}")
    print("Lütfen 'Setup' hücresini (yukarıdaki) çalıştırdığınızdan emin olun.")
    print(f"Şu anki dizin: {os.getcwd()}")

# Veri Yükleme
if 'load_and_prep' in locals():
    print("📂 Veri yükleniyor...")
    try:
        df = load_and_prep(limit=100000)
        print(f"✅ Veri Hazır: {len(df)} satır.")
    except Exception as e:
        print(f"⚠️ Veri yükleme hatası: {e}")"""))

# Cell 4: Sanity Check
nb.cells.append(nbf.v4.new_code_cell("""# 📊 VERİ SAĞLIĞI KONTROLÜ (SANITY CHECK)
print("\\n🔍 VERİSETİ İNCELEMESİ (SANITY CHECK)")
print("-" * 50)
total_games = len(df)
print(f"Toplam Oyun Sayısı: {total_games}")

multipliers_to_check = [3.0, 5.0, 10.0, 20.0, 50.0]

for m in multipliers_to_check:
    count = df[df['value'] >= m].shape[0]
    percentage = (count / total_games) * 100
    print(f"Outcome >= {m:6.1f}x : {count:5d} oyun (%{percentage:.2f})")

max_mult = df['value'].max()
print("-" * 50)
print(f"🔥 Verisetindeki En Yüksek Çarpan (Max): {max_mult:.2f}x")
print("-" * 50)"""))

# Cell 5: Advanced Models Header
nb.cells.append(nbf.v4.new_markdown_cell("## 🕵️‍♂️ Bölüm 1: İstihbarat & Komuta"))

# Cell 6: Autoencoder
nb.cells.append(nbf.v4.new_code_cell("""if 'model_ae_avci' in locals():
    ae = model_ae_avci.train_autoencoder(df)"""))

# Cell 7: HMM
nb.cells.append(nbf.v4.new_code_cell("""if 'model_hmm_avci' in locals():
    hmm_model = model_hmm_avci.train_hmm(df)"""))

# Cell 8: Training Header
nb.cells.append(nbf.v4.new_markdown_cell("## 🎯 Bölüm 2: Nişancı Eğitimi (Adım Adım)"))

# Cell 9: 3.0x Optimize
nb.cells.append(nbf.v4.new_code_cell("""# 3.00x - OPTİMİZASYON 🛠️
TARGET = 3.0
EPOCHS = 30
study_30, params_30 = optimize_target(df, TARGET, epochs=EPOCHS)"""))

# Cell 10: 3.0x Train
nb.cells.append(nbf.v4.new_code_cell("""# 3.0x - FİNAL EĞİTİM 🏃
model_30, X_val, y_val = train_target_final(df, TARGET, params_30)
visualize_performance(model_30, X_val, y_val, TARGET)"""))

# Cell 11: 5.0x Optimize
nb.cells.append(nbf.v4.new_code_cell("""# 5.00x - OPTİMİZASYON 🛠️
TARGET = 5.0
EPOCHS = 30
study_50, params_50 = optimize_target(df, TARGET, epochs=EPOCHS)"""))

# Cell 12: 5.0x Train
nb.cells.append(nbf.v4.new_code_cell("""# 5.0x - FİNAL EĞİTİM 🏃
model_50, X_val, y_val = train_target_final(df, TARGET, params_50)
visualize_performance(model_50, X_val, y_val, TARGET)"""))

# Cell 13: 10.0x Optimize
nb.cells.append(nbf.v4.new_code_cell("""# 10.0x - OPTİMİZASYON 🛠️
TARGET = 10.0
EPOCHS = 30
study_100, params_100 = optimize_target(df, TARGET, epochs=EPOCHS)"""))

# Cell 14: 10.0x Train
nb.cells.append(nbf.v4.new_code_cell("""# 10.0x - FİNAL EĞİTİM 🏃
model_100, X_val, y_val = train_target_final(df, TARGET, params_100)
visualize_performance(model_100, X_val, y_val, TARGET)"""))

# Cell 15: 20.0x Optimize
nb.cells.append(nbf.v4.new_code_cell("""# 20.0x - OPTİMİZASYON 🛠️
TARGET = 20.0
EPOCHS = 30
study_20, params_20 = optimize_target(df, TARGET, epochs=EPOCHS)"""))

# Cell 16: 20.0x Train
nb.cells.append(nbf.v4.new_code_cell("""# 20.0x - FİNAL EĞİTİM 🏃
model_20, X_val, y_val = train_target_final(df, TARGET, params_20)
visualize_performance(model_20, X_val, y_val, TARGET)"""))

# Cell 17: 50.0x Optimize
nb.cells.append(nbf.v4.new_code_cell("""# 50.0x - OPTİMİZASYON (Yeni Hedef) 🛠️
TARGET = 50.0
EPOCHS = 30
study_500, params_500 = optimize_target(df, TARGET, epochs=EPOCHS)"""))

# Cell 18: 50.0x Train
nb.cells.append(nbf.v4.new_code_cell("""# 50.0x - FİNAL EĞİTİM 🏃
model_500, X_val, y_val = train_target_final(df, TARGET, params_500)
visualize_performance(model_500, X_val, y_val, TARGET)"""))

# Cell 19: Meta Model
nb.cells.append(nbf.v4.new_code_cell("""# ENSEMBLE (META-MODEL) EĞİTİMİ 🤝
# Tüm alt modeller eğitildikten sonra, onların görmediği %15'lik veride Meta-Model eğitilir.

# Önce eğitilen modelleri toplayalım (Hata almamak için try-except veya check)
trained_models = {}
if 'model_30' in locals(): trained_models[3.0] = model_30
if 'model_50' in locals(): trained_models[5.0] = model_50
if 'model_100' in locals(): trained_models[10.0] = model_100
if 'model_20' in locals(): trained_models[20.0] = model_20
if 'model_500' in locals(): trained_models[50.0] = model_500

if len(trained_models) > 0:
    # 50x Hedefi için Meta-Model Eğit (Veya genel performans)
    print("🦅 Meta-Model (Ensemble) Devreye Giriyor...")
    meta_results = train_meta_model(df, trained_models, target=50.0)
    if meta_results is not None:
        print("✅ Meta-Model Analizi Tamamlandı. Veri özeti:")
        print(meta_results.tail())
else:
    print("⚠️ Hiçbir alt model bulunamadı (Eğitimleri çalıştırdınız mı?)")"""))

# Cell 20: Save Header
nb.cells.append(nbf.v4.new_markdown_cell("## 💾 Bölüm 3: Modelleri Kaydet ve İndir"))

# Cell 21: Save Code
nb.cells.append(nbf.v4.new_code_cell("""# Tüm Modelleri Ziple ve Drive'a Gönder
import os
if not os.path.exists('models'):
    os.makedirs('models')
    
!zip -r avci_models.zip models/

# Drive Mount (Sadece Şimdi)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    !cp avci_models.zip /content/drive/MyDrive/avci_backup_latest.zip
    print("✅ Yedek Drive'a (Ana Dizin) başarıyla kopyalandı.")
except Exception as e:
    print(f"⚠️ Drive'a kopyalanamadı: {e}")

# İndir (Opsiyonel)
try:
    from google.colab import files
    files.download('avci_models.zip')
except Exception as e:
    pass"""))

with open('Avci_Trainer.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Avci_Trainer.ipynb updated to v0.2.2 ✅")
