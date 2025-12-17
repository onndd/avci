
import pandas as pd
import numpy as np
import optuna
import joblib
import os
from config_avci import DB_PATH, TARGETS, SCORING_1_5, SCORING_2_0, SCORING_3_0, SCORING_5_0, SCORING_10_0, SCORING_20_0, SCORING_50_0, SCORING_100_0
from data_avci import load_data, add_targets
from features_avci import extract_features
from models_avci import train_lgbm, objective_lgbm

def get_scoring_params(target):
    if target == 1.5: return SCORING_1_5
    if target == 2.0: return SCORING_2_0
    if target == 3.0: return SCORING_3_0
    if target == 5.0: return SCORING_5_0
    if target == 10.0: return SCORING_10_0
    if target == 20.0: return SCORING_20_0
    if target == 50.0: return SCORING_50_0
    if target == 100.0: return SCORING_100_0
    if target == 1000.0: return SCORING_1000_0
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
    X = df[features]
    
    split_idx = int(len(df) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    
    y_col = f'target_{str(target).replace(".","_")}'
    y = df[y_col]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scoring = get_scoring_params(target)
    print(f"Scoring Rules for {target}x: {scoring}")
    
    # Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring, use_gpu=True), n_trials=epochs)
    
    print(f"Best Params: {study.best_params}")
    print(f"Best Profit Score: {study.best_value}")
    
    return study, study.best_params

def train_target_final(df, target, best_params):
    """
    Trains the final model using the provided best parameters.
    """
    print(f"\n--- Final Training Target: {target}x ---")
    
    # Split (Same as optimize)
    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    split_idx = int(len(df) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_col = f'target_{str(target).replace(".","_")}'
    y = df[y_col]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Update params for final training
    final_params = best_params.copy()
    final_params.update({'metric': 'binary_logloss', 'objective': 'binary', 'verbosity': -1, 'device': 'gpu'})
    
    model = train_lgbm(X_train, y_train, X_val, y_val, final_params)
    
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
    
    # 1. Confidence Plot (Scatter)
    plt.figure(figsize=(12, 5))
    plt.scatter(res['Game_ID'], res['Probability'], c=res['Actual'], cmap='coolwarm', alpha=0.6, s=15)
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title(f"Avci Confidence Level ({target}x) - Red: Crash, Blue: Hit")
    plt.xlabel("Game Sequence")
    plt.ylabel("Confidence (Probability)")
    plt.colorbar(label='Actual Outcome (0/1)')
    plt.show()
    
    # 2. Cumulative Profit (Simulation)
    # Assume we bet 1 unit whenever prob > Threshold
    # We need to find the 'Best Threshold' used implicitly or define one.
    # Let's find best threshold on this Val set for the plot
    # This matches the Optuna logic logic roughly
    
    scoring = get_scoring_params(target)
    best_thr = 0.5
    best_score = -float('inf')
    thresholds = np.arange(0.5, 0.99, 0.01)
    
    for thr in thresholds:
        tp = ((res['Probability'] > thr) & (res['Actual'] == 1)).sum()
        fp = ((res['Probability'] > thr) & (res['Actual'] == 0)).sum()
        score = (tp * scoring['TP']) - (fp * scoring['FP'])
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"Visualizing for Optimal Threshold: {best_thr:.2f}")
    
    res['Action'] = (res['Probability'] > best_thr).astype(int)
    # PnL: If Action=1 and Actual=1 -> +Profit (Target - 1). If Action=1 and Actual=0 -> -1.
    # Note: Target 3.0x means Profit is 2.0 per unit.
    profit_mult = target - 1.0
    res['PnL'] = np.where(res['Action'] == 1, 
                          np.where(res['Actual'] == 1, profit_mult, -1.0), 
                          0.0)
    
    res['Equity'] = res['PnL'].cumsum()
    
    plt.figure(figsize=(12, 5))
    plt.plot(res['Game_ID'], res['Equity'], color='green', linewidth=2)
    plt.title(f"Simulated Profit/Loss Curve ({target}x) @ Threshold {best_thr:.2f}")
    plt.xlabel("Games Played")
    plt.ylabel("Net Units Won")
    plt.grid(True, alpha=0.3)
    plt.show()

def run_training():
    # Legacy wrapper
    df = load_and_prep()
    for t in TARGETS:
        model, X_val, y_val, _ = train_target(df, t, epochs=20)
        visualize_performance(model, X_val, y_val, t)
