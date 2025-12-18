
import pandas as pd
import numpy as np
import optuna
import joblib
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from config_avci import DB_PATH, TARGETS, SCORING_2_0, SCORING_3_0, SCORING_5_0, SCORING_10_0, SCORING_20_0, SCORING_50_0
from data_avci import load_data, add_targets
from features_avci import extract_features
from models_avci import train_lgbm, objective_lgbm

def get_scoring_params(target):
    if target == 2.0: return SCORING_2_0
    if target == 3.0: return SCORING_3_0
    if target == 5.0: return SCORING_5_0
    if target == 10.0: return SCORING_10_0
    if target == 20.0: return SCORING_20_0
    if target == 50.0: return SCORING_50_0
    return SCORING_3_0 # Default

import matplotlib.pyplot as plt

def load_and_prep(limit=100000):
    """Loads and features engineering"""
    print("Loading Data...")
    df = load_data(DB_PATH, limit=limit)
    print("Extracting Features...")
    # Import WINDOWS locally if not available globally
    from config_avci import WINDOWS
        
    df = extract_features(df, windows=WINDOWS)
    print("Labelling Targets...")
    df = add_targets(df, TARGETS)
    return df

def optimize_target(df, target, epochs=20):
    """
    Runs Optuna optimization for a specific target and returns the study and best parameters.
    Does NOT train the final model.
    """
    print(f"\n--- Optimizing Target: {target}x (Trials: {epochs}) ---")
    
    # Split
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    
    # 3-Way Split: Train(70%) - Val(15%) - Meta(15%)
    # Optimization only sees Train and Val.
    # Meta part is HIDDEN from this step.
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85) # Val is 70% to 85%
    
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    
    y_col = f'target_{str(target).replace(".","_")}'
    y = df[y_col]
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    
    scoring = get_scoring_params(target)
    print(f"Scoring Rules for {target}x: {scoring}")
    
    # Calculate scale_pos_weight: DISABLED (User request: Avoid "play everything" strategy)
    extra_params = {}
    # if target >= 10.0:
    #     pos_count = y_train.sum()
    #     neg_count = len(y_train) - pos_count
    #     if pos_count > 0:
    #         scale_pos_weight = neg_count / pos_count
    #         extra_params['scale_pos_weight'] = scale_pos_weight
    #         print(f"⚖️ Balancing Classes: scale_pos_weight = {scale_pos_weight:.2f}")

    # Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring, use_gpu=True, extra_params=extra_params), n_trials=epochs)
    
    print(f"Best Params: {study.best_params}")
    print(f"Best Profit Score: {study.best_value}")
    
    return study, study.best_params

def get_best_device():
    try:
        import lightgbm as lgb
        # Basic check: try to train a dummy model on GPU
        # If unavailable, it might crash or throw warning.
        # A simpler way is to default to 'cpu' unless user explicitly enabled GPU in setup.
        # But we will return 'gpu' and catch error during training if needed?
        # Safe Default: Check simple heuristic or let LightGBM handle fallback?
        # LightGBM crashes on 'gpu' if not supported.
        # Let's assume CPU default unless we are sure.
        # For Colab T4 we want GPU.
        # We can check 'nvidia-smi' presence?
        import subprocess
        try:
            subprocess.check_output('nvidia-smi')
            return 'gpu'
        except:
            return 'cpu'
    except:
        return 'cpu'

def train_target_final(df, target, best_params):
    """
    Trains the final model using the provided best parameters.
    """
    print(f"\n--- Final Training Target: {target}x ---")
    
    # Split (Same as optimize)
    # Split for Final Sub-Model Training
    # Use 70% for Train, 15% for Val (Early Stopping). 
    # The last 15% (Meta) is still HIDDEN.
    
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]  # Used for early stopping
    
    y_col = f'target_{str(target).replace(".","_")}'
    y = df[y_col]
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    
    # Update params for final training (Device Auto-Detect)
    device_type = get_best_device()
    print(f"⚙️ Training Device: {device_type.upper()}")
    
    final_params = best_params.copy()
    final_params.update({'metric': 'binary_logloss', 'objective': 'binary', 'verbosity': -1, 'device': device_type})

    # Apply same balancing logic for final training
    # Apply same balancing logic: DISABLED
    # if target >= 10.0:
    #     pos_count = y_train.sum()
    #     neg_count = len(y_train) - pos_count
    #     if pos_count > 0:
    #         weight = neg_count / pos_count
    #         final_params['scale_pos_weight'] = weight
    #         print(f"⚖️ Final Training Weight: {weight:.2f}")
    
    try:
        model = train_lgbm(X_train, y_train, X_val, y_val, final_params)
    except Exception as e:
        if device_type == 'gpu':
            print(f"⚠️ GPU Hatası: {e}. CPU ile tekrar deneniyor...")
            final_params['device'] = 'cpu'
            model = train_lgbm(X_train, y_train, X_val, y_val, final_params)
        else:
            raise e
    
    os.makedirs('models', exist_ok=True)
    model.save_model(f'models/avci_lgbm_{str(target).replace(".","_")}.txt')
    print(f"Model saved.")
    
    return model, X_val, y_val

def visualize_performance(model, X_val, y_val, target):
    """
    Plots Confidence vs Game Time and Cumulative Profit.
    """
    preds_proba = model.predict(X_val)
    
    # Create Analysis DF
    res = pd.DataFrame({
        'Game_ID': range(len(preds_proba)), # Simulation ID
        'Probability': preds_proba,
        'Actual': y_val.values
    })
    
    # --- 1. Probability Distribution (Histogram) - REMOVED (Confusing) ---
    # --- 2. Confidence Plot - REMOVED (Confusing) ---
    
    
    # --- 3. Feature Importance ---
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', figsize=(10,6), title=f'Feature Importance (Gain) - {target}x')
    plt.show()

    # --- 4. Optimization & Confusion Matrix ---
    scoring = get_scoring_params(target)
    best_thr = 0.5
    best_score = -float('inf')
    thresholds = np.arange(0.1, 0.99, 0.01) # Expanded range for high targets
    
    for thr in thresholds:
        tp = ((res['Probability'] > thr) & (res['Actual'] == 1)).sum()
        fp = ((res['Probability'] > thr) & (res['Actual'] == 0)).sum()
        score = (tp * scoring['TP']) - (fp * scoring['FP'])
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"✅ Optimal Threshold found: {best_thr:.2f}")
    
    # Confusion Matrix at Optimal Threshold
    preds_binary = (res['Probability'] > best_thr).astype(int)
    cm = confusion_matrix(res['Actual'], preds_binary)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Crash', 'Pred WIN'],
                yticklabels=['Actual Crash', 'Actual WIN'])
    plt.title(f"Confusion Matrix @ {best_thr:.2f}")
    plt.show()
    
    # --- 5. Cumulative Profit & Drawdown ---
    res['Action'] = (res['Probability'] > best_thr).astype(int)
    profit_mult = target - 1.0
    res['PnL'] = np.where(res['Action'] == 1, 
                          np.where(res['Actual'] == 1, profit_mult, -1.0), 
                          0.0)
    
    res['Equity'] = res['PnL'].cumsum()
    
    # Calculate Drawdown
    res['Peak'] = res['Equity'].cummax()
    res['Drawdown'] = res['Equity'] - res['Peak']
    max_drawdown = res['Drawdown'].min()
    
    plt.figure(figsize=(12, 5))
    plt.plot(res['Game_ID'], res['Equity'], color='green', linewidth=2, label='Equity')
    plt.fill_between(res['Game_ID'], res['Drawdown'], 0, color='red', alpha=0.3, label=f'Drawdown (Max: {max_drawdown:.1f})')
    plt.title(f"Profit & Risk: Max Drawdown {max_drawdown:.1f} Units ({target}x)")
    plt.xlabel("Games Played")
    plt.ylabel("Net Units Won")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 6. Pie Chart (Sabır Pastası) ---
    played_wins = ((res['Action'] == 1) & (res['Actual'] == 1)).sum()
    played_losses = ((res['Action'] == 1) & (res['Actual'] == 0)).sum()
    passed = (res['Action'] == 0).sum()
    
    labels = ['Pas (Bekle)', 'Kayıp', 'Kazanç']
    sizes = [passed, played_losses, played_wins]
    colors = ['lightgray', 'salmon', 'lightgreen']
    explode = (0, 0.1, 0.1) 
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(f"Sabır Pastası: Hangi Sıklıkla Oynuyor? ({target}x)")
    plt.show()

    # --- 7. Streak Analysis (Seri Grafiği) ---
    # Only analyze streaks for played games
    played_games = res[res['Action'] == 1].copy()
    played_games['Win'] = (played_games['Actual'] == 1).astype(int)
    
    if len(played_games) > 0:
        # Calculate streaks
        # Group consecutive identical values
        # Logic: compare current with prev, if diff then accumulate
        played_games['grp'] = (played_games['Win'] != played_games['Win'].shift()).cumsum()
        streaks = played_games.groupby(['grp', 'Win']).size().reset_index(name='count')
        
        max_win_streak = streaks[streaks['Win'] == 1]['count'].max() if 1 in streaks['Win'].values else 0
        max_loss_streak = streaks[streaks['Win'] == 0]['count'].max() if 0 in streaks['Win'].values else 0
        
        plt.figure(figsize=(8, 4))
        plt.bar(['Max Kazanç Serisi', 'Max Kayıp Serisi'], [max_win_streak, max_loss_streak], color=['green', 'red'])
        plt.title(f"Seri Analizi: Peş Peşe Ne Geldi? ({target}x)")
        plt.ylabel("Ardışık Oyun Sayısı")
        for i, v in enumerate([max_win_streak, max_loss_streak]):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.show()

    # --- 8. Confidence vs Accuracy (Güven Analizi) ---
    # Show how accurate the model is at different confidence levels (Action only)
    plt.figure(figsize=(10, 5))
    
    # Correct Predictions (Green)
    correct_preds = res[(res['Action'] == 1) & (res['Actual'] == res['Action'])]
    # Wrong Predictions (Red)
    wrong_preds = res[(res['Action'] == 1) & (res['Actual'] != res['Action'])]
    
    plt.scatter(correct_preds['Game_ID'], correct_preds['Probability'], color='green', alpha=0.6, label='Doğru Tahmin', s=20)
    plt.scatter(wrong_preds['Game_ID'], wrong_preds['Probability'], color='red', alpha=0.6, label='Yanlış Tahmin', s=20)
    
    plt.axhline(best_thr, color='black', linestyle='--', label=f'Eşik ({best_thr:.2f})')
    plt.title(f"Güven Analizi: Yüksek Güven = Doğru Sonuç mu? ({target}x)")
    plt.xlabel("Oyun No")
    plt.ylabel("Model Güveni (Olasılık)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def train_meta_model(df, models, target=100.0):
    """
    Trains a simple Ensemble Meta-Model on the 'Hidden' 15% data.
    """
    print(f"\n--- Training Meta-Model (Ensemble) for {target}x ---")
    
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    
    n = len(df)
    meta_start = int(n * 0.85)
    
    # Meta Data (The part sub-models haven't seen during training/opt)
    X_meta = X.iloc[meta_start:].copy()
    y_col = f'target_{str(target).replace(".","_")}'
    y_meta = df[y_col].iloc[meta_start:]
    
    # 1. Generate Sub-Model Predictions
    meta_features = pd.DataFrame(index=X_meta.index)
    
    print("Generating predictions from sub-models...")
    for t_sub, model in models.items():
        try:
            preds = model.predict(X_meta)
            meta_features[f'pred_{t_sub}'] = preds
        except Exception as e:
            print(f"Skipping model {t_sub}x in ensemble: {e}")
            
    if meta_features.empty:
        print("No sub-model predictions available.")
        return None
        
    # 2. Simple Weighted Average or Logistic Regression?
    # For now, let's do a simple correlation/weight check
    # Or just return this dataframe for the user to analyze in notebook
    
    meta_features['Actual'] = y_meta
    
    print("Ensemble Data Prepared. (Showing first 5 rows)")
    print(meta_features.head())
    
    return meta_features

def run_training():
    # Legacy wrapper
    pass

