
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
from features_avci import extract_features, get_model_features
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

def optimize_target(df, target, n_trials=20):
    """
    Runs Optuna optimization for a specific target and returns the study and best parameters.
    Does NOT train the final model.
    """
    print(f"\n--- Optimizing Target: {target}x (Trials: {n_trials}) ---")
    
    # Split
    # Filter features using centralized logic
    features = get_model_features(target, df.columns)
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
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring, use_gpu=True, extra_params=extra_params), n_trials=n_trials)
    
    print(f"Best Params: {study.best_params}")
    print(f"Best Profit Score: {study.best_value}")
    
    return study

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
    
    # Filter features using centralized logic
    features = get_model_features(target, df.columns)
    print(f"ℹ️ Target {target}x Strategy: Using {len(features)} selected features.")

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
    Also prints detailed text reports for Agent Analysis.
    """
    preds_proba = model.predict(X_val)
    
    # Create Analysis DF
    res = pd.DataFrame({
        'Game_ID': range(len(preds_proba)), # Simulation ID
        'Probability': preds_proba,
        'Actual': y_val.values
    })
    
    # --- 1. Feature Importance ---
    print(f"\n📊 --- REPORT FOR TARGET {target}x ---")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', figsize=(10,6), title=f'Feature Importance (Gain) - {target}x')
    plt.show()

    # --- 2. Optimization & Threshold Finding ---
    scoring = get_scoring_params(target)
    best_thr = 0.5
    best_score = -float('inf')
    thresholds = np.arange(0.1, 0.99, 0.01) # Expanded range
    
    for thr in thresholds:
        tp = ((res['Probability'] > thr) & (res['Actual'] == 1)).sum()
        fp = ((res['Probability'] > thr) & (res['Actual'] == 0)).sum()
        score = (tp * scoring['TP']) - (fp * scoring['FP'])
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"✅ Optimal Threshold found: {best_thr:.2f}")
    
    # --- 3. Confusion Matrix & Detailed Counts ---
    res['Action'] = (res['Probability'] > best_thr).astype(int)
    
    tp = ((res['Action'] == 1) & (res['Actual'] == 1)).sum()
    fp = ((res['Action'] == 1) & (res['Actual'] == 0)).sum()
    tn = ((res['Action'] == 0) & (res['Actual'] == 0)).sum()
    fn = ((res['Action'] == 0) & (res['Actual'] == 1)).sum()
    
    total_games = len(res)
    played = tp + fp
    win_rate = (tp / played * 100) if played > 0 else 0.0
    
    print(f"\n🔢 [CONFUSION MATRIX & STATS]")
    print(f"   Total Games Evaluated : {total_games}")
    print(f"   played (Action=1)     : {played} ({played/total_games*100:.1f}%)")
    print(f"   Passed (Action=0)     : {tn + fn}")
    print(f"   ---------------------------")
    print(f"   Wins (TP)             : {tp}")
    print(f"   Losses (FP)           : {fp}")
    print(f"   Win Rate (Accuracy)   : {win_rate:.2f}%")
    print(f"   ---------------------------")
    print(f"   Missed Wins (FN)      : {fn}")
    print(f"   Avoided Crash (TN)    : {tn}")

    # Plot Confusion Matrix
    cm = confusion_matrix(res['Actual'], res['Action'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Crash', 'Pred WIN'],
                yticklabels=['Actual Crash', 'Actual WIN'])
    plt.title(f"Confusion Matrix @ {best_thr:.2f}")
    plt.show()
    
    # --- 4. Cumulative Profit & Drawdown ---
    profit_mult = target - 1.0
    res['PnL'] = np.where(res['Action'] == 1, 
                          np.where(res['Actual'] == 1, profit_mult, -1.0), 
                          0.0)
    
    res['Equity'] = res['PnL'].cumsum()
    res['Peak'] = res['Equity'].cummax()
    res['Drawdown'] = res['Equity'] - res['Peak']
    max_drawdown = res['Drawdown'].min()
    final_profit = res['Equity'].iloc[-1]
    
    # Advanced Metrics
    gross_profit = res[res['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(res[res['PnL'] < 0]['PnL'].sum())
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    total_invested = played  # Assuming 1 unit per bet
    roi = (final_profit / total_invested * 100) if total_invested > 0 else 0.0
    
    win_rate_dec = win_rate / 100
    ev_per_trade = (win_rate_dec * profit_mult) - ((1 - win_rate_dec) * 1.0)
    
    recovery_factor = (final_profit / abs(max_drawdown)) if max_drawdown < 0 else 999.0

    # --- COMPOUNDING SIMULATOR (Let It Ride) ---
    # Logic: Bet 1. Win -> Bet (Target). Win -> Bet (Target^2)... until Target Streak
    # If lose at any point -> Lose 1 unit initial risk.
    
    target_streak = 2
    if target == 2.0: target_streak = 5 # User request: 5 times for 2x
    elif target <= 3.0: target_streak = 4
    elif target <= 5.0: target_streak = 3
    
    combo_bankroll = 0.0
    current_combo_bet = 0.0 # Virtual money on table
    combo_level = 0
    combo_equity = []
    
    # We iterate through 'Action' (Played Games)
    # We need sequential processing for compounding
    
    actions = res[res['Action'] == 1]
    
    for idx, row in actions.iterrows():
        is_win = (row['Actual'] == 1)
        
        if combo_level == 0:
            # Start new chain
            combo_bankroll -= 1.0 # Cost to start
            current_combo_bet = 1.0
            
        if is_win:
            # Won! Let it ride.
            current_combo_bet = current_combo_bet * target
            combo_level += 1
            
            if combo_level >= target_streak:
                # Target Hit! CASHOUT.
                combo_bankroll += current_combo_bet
                combo_level = 0
                current_combo_bet = 0.0
        else:
            # Clean loss. Chain broken.
            # We already paid the 1 unit at start.
            # Money on table is gone.
            combo_level = 0
            current_combo_bet = 0.0
            
    print(f"\n💰 [FINANCIAL PERFORMANCE]")
    print(f"   Final Net Profit      : {final_profit:.1f} Units (Standard Flat Bet)")
    print(f"   Compound Profit (Sim) : {combo_bankroll:.1f} Units (Target Streak: {target_streak})")
    print(f"   Max Drawdown          : {max_drawdown:.1f} Units")
    print(f"   Profit Factor         : {profit_factor:.2f} (Target > 1.5)")
    print(f"   Return on Inv (ROI)   : {roi:.1f}%")
    print(f"   Expected Value (EV)   : {ev_per_trade:.2f} Units/Bet")
    print(f"   Recovery Factor       : {recovery_factor:.2f} (Higher is better)")
    
    plt.figure(figsize=(12, 5))
    plt.plot(res['Game_ID'], res['Equity'], color='green', linewidth=2, label='Equity')
    plt.fill_between(res['Game_ID'], res['Drawdown'], 0, color='red', alpha=0.3, label=f'Drawdown (Max: {max_drawdown:.1f})')
    plt.title(f"Profit & Risk: Max Drawdown {max_drawdown:.1f} Units ({target}x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 5. Pie Chart Info ---
    # Already printed in stats section above, simply plotting
    played_wins = tp
    played_losses = fp
    passed = tn + fn
    
    labels = ['Pas (Bekle)', 'Kayıp', 'Kazanç']
    sizes = [passed, played_losses, played_wins]
    colors = ['lightgray', 'salmon', 'lightgreen']
    explode = (0, 0.1, 0.1) 
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(f"Sabır Pastası: Hangi Sıklıkla Oynuyor? ({target}x)")
    plt.show()

    # --- 6. Streak Analysis ---
    played_games = res[res['Action'] == 1].copy()
    played_games['Win'] = (played_games['Actual'] == 1).astype(int)
    
    max_win_streak = 0
    max_loss_streak = 0
    
    if len(played_games) > 0:
        played_games['grp'] = (played_games['Win'] != played_games['Win'].shift()).cumsum()
        streaks = played_games.groupby(['grp', 'Win']).size().reset_index(name='count')
        
        if 1 in streaks['Win'].values:
            max_win_streak = streaks[streaks['Win'] == 1]['count'].max()
        if 0 in streaks['Win'].values:
            max_loss_streak = streaks[streaks['Win'] == 0]['count'].max()
            
    print(f"\n🔥 [STREAK ANALYSIS]")
    print(f"   Max Winning Streak    : {max_win_streak}")
    print(f"   Max Losing Streak     : {max_loss_streak}")
    
    if len(played_games) > 0:
        plt.figure(figsize=(8, 4))
        plt.bar(['Max Kazanç Serisi', 'Max Kayıp Serisi'], [max_win_streak, max_loss_streak], color=['green', 'red'])
        plt.title(f"Seri Analizi: Peş Peşe Ne Geldi? ({target}x)")
        plt.ylabel("Ardışık Oyun Sayısı")
        for i, v in enumerate([max_win_streak, max_loss_streak]):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.show()

    # --- 7. Confidence Distribution Stats ---
    print(f"\n🧠 [MODEL CONFIDENCE STATS]")
    print(f"   Mean Probability      : {res['Probability'].mean():.4f}")
    print(f"   Max Probability       : {res['Probability'].max():.4f}")
    print(f"   Min Probability       : {res['Probability'].min():.4f}")
    print(f"📊 --------------------------- end report ---------------------------\n")

    # Confidence Plot
    plt.figure(figsize=(10, 5))
    correct_preds = res[(res['Action'] == 1) & (res['Actual'] == res['Action'])]
    wrong_preds = res[(res['Action'] == 1) & (res['Actual'] != res['Action'])]
    plt.scatter(correct_preds['Game_ID'], correct_preds['Probability'], color='green', alpha=0.6, label='Doğru Tahmin', s=20)
    plt.scatter(wrong_preds['Game_ID'], wrong_preds['Probability'], color='red', alpha=0.6, label='Yanlış Tahmin', s=20)
    plt.axhline(best_thr, color='black', linestyle='--', label=f'Eşik ({best_thr:.2f})')
    plt.title(f"Güven Analizi: Yüksek Güven = Doğru Sonuç mu? ({target}x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 8. Confidence Interval Analysis (Binning) ---
    print(f"\n🔬 [CONFIDENCE BINNING ANALYSIS]")
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    res['Conf_Bin'] = pd.cut(res['Probability'], bins=bins, labels=labels)
    
    bin_stats = res.groupby('Conf_Bin', observed=False).agg(
        Count=('Actual', 'count'),
        Wins=('Actual', 'sum'),
        Mean_Prob=('Probability', 'mean')
    )
    bin_stats['Win_Rate'] = (bin_stats['Wins'] / bin_stats['Count'] * 100).fillna(0)
    
    print(bin_stats[['Count', 'Wins', 'Win_Rate']])
    
    # Plot Binning
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar Chart (Volume)
    ax1.set_xlabel('Güven Aralığı (Confidence Bin)')
    ax1.set_ylabel('İşlem Sayısı', color='tab:blue')
    ax1.bar(bin_stats.index, bin_stats['Count'], color='tab:blue', alpha=0.6, label='İşlem Hacmi')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Line Chart (Win Rate)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Kazanma Oranı (%)', color='tab:green')
    ax2.plot(bin_stats.index, bin_stats['Win_Rate'], color='tab:green', marker='o', linewidth=2, label='Win Rate')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(0, 100)
    
    # Threshold Line
    ax2.axhline(50, color='grey', linestyle='--', alpha=0.5)
    
    plt.title(f"Güven Aralığı Analizi: Model Nerede Daha Başarılı? ({target}x)")
    fig.tight_layout()
    plt.show()

    # --- 9. Save JSON Report (Agent Readable) ---
    import json
    # --- 9. Save JSON Report (Agent Readable) ---
    import json
    report_file = "reports/v0.9.4_training_metrics.json"
    os.makedirs("reports", exist_ok=True)
    
    # ... (code omitted) ...

    # --- 9b. Save Feature Analysis Report (Separate File) ---
    feature_report_file = "reports/v0.9.4_feature_analysis.json"
    
    # Prepare Stats
    # 1. Feature Importance Extraction
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(10)
    feat_imp_dict = feat_imp.set_index('feature')['importance'].to_dict()

    stats = {
        "target": target,
        "best_threshold": f"{best_thr:.2f}",
        "win_rate": f"{win_rate:.2f}%",
        "total_trades": int(played),
        "wins": int(tp),
        "losses": int(fp),
        "net_profit": f"{final_profit:.2f}",
        "max_drawdown": f"{max_drawdown:.2f}",
        "confidence_bins": bin_stats[['Count', 'Win_Rate']].to_dict('index'),
        "streak_analysis": {
            "max_win_streak": int(max_win_streak),
            "max_loss_streak": int(max_loss_streak)
        },
        "compound_profit": f"{combo_bankroll:.2f}",
        "target_streak": int(target_streak),
        "feature_importance": feat_imp_dict
    }
    
    # Load existing or create new
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                full_report = json.load(f)
        except:
            full_report = {}
    else:
        full_report = {}
        
    full_report[str(target)] = stats
    
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=4)
        
    print(f"\n📝 Report saved to {report_file}")

    # --- 9b. Save Feature Analysis Report (Separate File) ---
    feature_report_file = "reports/v0.5.0_feature_analysis.json"
    feature_data = {}
    for t, metrics in full_report.items():
        feature_data[str(t)] = metrics.get('feature_importance', {})
    
    with open(feature_report_file, 'w', encoding='utf-8') as f:
        json.dump(feature_data, f, indent=4)
    print(f"✅ Feature Analysis Saved: {feature_report_file}")


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
    """
    Orchestrates the full local training pipeline.
    Useful for running via terminal: 'python train_avci.py'
    """
    print("🦅 AVCI LOCAL TRAINING ORCHESTRATOR STARTED 🦅")
    
    # 1. Load Data
    print("\n📂 Loading Data...")
    df = load_and_prep(limit=200000) # Use a good amount of history
    if df is None: return

    trained_models = {}
    
    # 2. Train Each Target
    for target in TARGETS:
        print(f"\n{'='*40}")
        print(f"🎯 PROCESSING TARGET: {target}x")
        print(f"{'='*40}")
        
        # A. Optimization
        study = optimize_target(df, target, n_trials=100)
        best_params = study.best_trial.params
        
        # B. Final Training
        model, X_val, y_val = train_target_final(df, target, best_params)
        
        # C. Reporting
        visualize_performance(model, X_val, y_val, target)
        
        trained_models[target] = model
        
    # 3. Train Meta Model
    print(f"\n{'='*40}")
    print(f"🧠 TRAINING META-MODEL (ENSEMBLE)")
    print(f"{'='*40}")
    
    if len(trained_models) > 0:
        meta_results = train_meta_model(df, trained_models, target=50.0)
        if meta_results is not None:
            print("\nMeta-Model Sample Predictions:")
            print(meta_results.tail())
            
    print("\n✅ ALL LOCAL TRAINING COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_training()

