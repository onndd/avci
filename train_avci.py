
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
    from config_avci import WINDOWS
    df = extract_features(df, windows=WINDOWS)
    print("Labelling Targets...")
    df = add_targets(df, TARGETS)
    return df

def optimize_target(df, target, n_trials=20):
    """
    Optimizes LightGBM hyperparameters using Optuna (WFV STABILITY MODE).
    """
    print(f"\n--- Optimizing Target: {target}x (Trials: {n_trials}) ---")
    
    scoring_params = get_scoring_params(target)
    print(f"Scoring Rules for {target}x: {scoring_params}")

    features = get_model_features(target, df.columns)
    X = df[features]
    y_col = f'target_{str(target).replace(".","_")}'
    y = df[y_col]
        
    pos_samples = y.sum()
    if pos_samples < 50:
         print(f"⚠️ Not enough positive samples ({pos_samples}) for {target}x. Skipping optimization.")
         return {}
          
    import functools
    from models_avci import objective_lgbm_wfv
    
    extra_params = {}
    if target == 2.0:
        extra_params = {'min_data_in_leaf': 50} 
        
    objective = functools.partial(
        objective_lgbm_wfv, 
        X=X, 
        y=y, 
        scoring_params=scoring_params,
        use_gpu=True,
        extra_params=extra_params
    )
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Best Trial: {study.best_trial.value:.2f}")
    print(f"   Params: {study.best_trial.params}")
    
    return study.best_trial.params

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
    
    device_type = get_best_device()
    print(f"⚙️ Training Device: {device_type.upper()}")
    
    final_params = best_params.copy()
    final_params.update({'metric': 'binary_logloss', 'objective': 'binary', 'verbosity': -1, 'device': device_type})

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

def visualize_performance(model, X_val, y_val, target, wfv_stats=None):
    """
    Plots Confidence vs Game Time and Cumulative Profit.
    Also prints detailed text reports for Agent Analysis.
    """
    preds_proba = model.predict(X_val)
    
    res = pd.DataFrame({
        'Game_ID': range(len(preds_proba)), 
        'Probability': preds_proba,
        'Actual': y_val.values
    })
    
    print(f"\n📊 --- REPORT FOR TARGET {target}x ---")
    
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = pd.DataFrame({'feature': feature_names, 'gain': importance}).sort_values('gain', ascending=False)
    top_features = feat_imp.head(10).set_index('feature')['gain'].to_dict()

    scoring = get_scoring_params(target)
    best_thr = 0.5
    best_score = -float('inf')
    thresholds = np.arange(0.1, 0.99, 0.01) 
    
    for thr in thresholds:
        tp_ = ((res['Probability'] > thr) & (res['Actual'] == 1)).sum()
        fp_ = ((res['Probability'] > thr) & (res['Actual'] == 0)).sum()
        score = (tp_ * scoring['TP']) - (fp_ * scoring['FP'])
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"✅ Optimal Threshold found: {best_thr:.2f}")
    
    res['Action'] = (res['Probability'] > best_thr).astype(int)
    
    tp = ((res['Action'] == 1) & (res['Actual'] == 1)).sum()
    fp = ((res['Action'] == 1) & (res['Actual'] == 0)).sum()
    tn = ((res['Action'] == 0) & (res['Actual'] == 0)).sum()
    fn = ((res['Action'] == 0) & (res['Actual'] == 1)).sum()
    
    total_games = len(res)
    played = tp + fp
    win_rate = (tp / played * 100) if played > 0 else 0.0
    
    print(f"\\n🔢 [CONFUSION MATRIX & STATS]")
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

    profit_mult = target - 1.0
    res['PnL'] = np.where(res['Action'] == 1, 
                          np.where(res['Actual'] == 1, profit_mult, -1.0), 
                          0.0)
    
    res['Equity'] = res['PnL'].cumsum()
    res['Peak'] = res['Equity'].cummax()
    res['Drawdown'] = res['Equity'] - res['Peak']
    max_drawdown = res['Drawdown'].min()
    final_profit = res['Equity'].iloc[-1]
    
    gross_profit = res[res['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(res[res['PnL'] < 0]['PnL'].sum())
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    total_invested = played 
    roi = (final_profit / total_invested * 100) if total_invested > 0 else 0.0
    
    win_rate_dec = win_rate / 100
    ev_per_trade = (win_rate_dec * profit_mult) - ((1 - win_rate_dec) * 1.0)
    
    recovery_factor = (final_profit / abs(max_drawdown)) if max_drawdown < 0 else 999.0

    target_streak = 2
    if target == 2.0: target_streak = 5
    elif target <= 3.0: target_streak = 4
    elif target <= 5.0: target_streak = 3
    
    combo_bankroll = 0.0
    current_combo_bet = 0.0
    combo_level = 0
    
    actions = res[res['Action'] == 1]
    for idx, row in actions.iterrows():
        is_win = (row['Actual'] == 1)
        if combo_level == 0:
            combo_bankroll -= 1.0
            current_combo_bet = 1.0
        if is_win:
            current_combo_bet = current_combo_bet * target
            combo_level += 1
            if combo_level >= target_streak:
                combo_bankroll += current_combo_bet
                combo_level = 0
                current_combo_bet = 0.0
        else:
            combo_level = 0
            current_combo_bet = 0.0
            
    final_compound_profit = combo_bankroll
    
    # --- STREAK ANALYSIS (MOVED UP FOR SCOPE) ---
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
    
    max_win_streak = int(max_win_streak)
    max_loss_streak = int(max_loss_streak)
    
    print(f"\n🔥 [STREAK ANALYSIS]")
    print(f"   Max Winning Streak    : {max_win_streak}")
    print(f"   Max Losing Streak     : {max_loss_streak}")

    # Confidence Binning
    bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    res['Conf_Bin'] = pd.cut(res['Probability'], bins=bins, labels=labels)
    
    bin_stats = res.groupby('Conf_Bin', observed=False).agg(
        Count=('Actual', 'count'),
        Wins=('Actual', 'sum'),
        Mean_Prob=('Probability', 'mean')
    )
    bin_stats['Win_Rate'] = (bin_stats['Wins'] / bin_stats['Count'] * 100).fillna(0)
    
    print(f"\\n\\n🔬 [CONFIDENCE BINNING ANALYSIS]")
    print(bin_stats)
    
    # 8b. Chain Method Profit
    chain_target = 2
    if target == 2.0: chain_target = 5
    elif target == 3.0: chain_target = 4
    elif target == 5.0: chain_target = 3
    elif target == 10.0: chain_target = 3
    
    chain_bankroll = 0.0
    current_bet = 1.0 
    streak_count = 0
    
    played_df = res[res['Action'] == 1].sort_values('Game_ID')
    
    for idx, row in played_df.iterrows():
        is_win = (row['Actual'] == 1)
        if is_win:
            gross_win = current_bet * target
            streak_count += 1
            if streak_count >= chain_target:
                chain_bankroll += (gross_win - 1.0)
                current_bet = 1.0
                streak_count = 0
            else:
                current_bet = gross_win
        else:
            chain_bankroll -= 1.0
            current_bet = 1.0
            streak_count = 0
            
    print(f"\\n🔗 [CHAIN METHOD REPORT]")
    print(f"   Target Win Streak     : {chain_target}")
    print(f"   Chain Profit (Units)  : {chain_bankroll:.2f}")

    # Construct Report Dictionary
    stats = {
        'target': float(target),
        'best_threshold': f"{best_thr:.2f}",
        'win_rate': f"{win_rate:.2f}%",
        'total_trades': int(played),
        'wins': int(tp),
        'losses': int(fp),
        'net_profit': f"{final_profit:.2f}",
        'max_drawdown': f"{max_drawdown:.2f}",
        'confidence_bins': bin_stats[['Count', 'Win_Rate']].to_dict('index'),
        'streak_analysis': {
            'max_win_streak': int(max_win_streak),
            'max_loss_streak': int(max_loss_streak)
        },
        'compound_profit': f"{final_compound_profit:.2f}",
        'chain_profit': f"{chain_bankroll:.2f}",
        'chain_target': int(chain_target),
        'target_streak': int(target_streak),
        'feature_importance': top_features
    }

    # ADD WFV STATS IF AVAILABLE
    if wfv_stats:
        print(f"\\n🔗 Merging Walk-Forward Validation Stats into Report...")
        stats['walk_forward_validation'] = wfv_stats
        
        # MEANINGFUL REALISM: Overwrite misleading "Static Validation" metrics with "Real WFV" metrics
        stats['REAL_wfv_net_profit'] = f"{wfv_stats['avg_profit']:.2f}"
        stats['REAL_wfv_win_rate'] = f"{wfv_stats['avg_win_rate']:.2f}%"


    print(f"\n💰 [FINANCIAL PERFORMANCE]")
    print(f"   Final Net Profit      : {final_profit:.1f} Units (Standard Flat Bet)")
    print(f"   Compound Profit (Sim) : {combo_bankroll:.1f} Units (Target Streak: {target_streak})")
    print(f"   Max Drawdown          : {max_drawdown:.1f} Units")
    print(f"   Profit Factor         : {profit_factor:.2f} (Target > 1.5)")
    print(f"   Return on Inv (ROI)   : {roi:.1f}%")
    print(f"   Expected Value (EV)   : {ev_per_trade:.2f} Units/Bet")
    print(f"   Recovery Factor       : {recovery_factor:.2f} (Higher is better)")
    
    print(f"\n{'-'*30}\n📊 Saving Reports...\n{'-'*30}")
    
    import json
    report_file = "reports/v0.05_training_metrics.json"
    
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
        
    print(f"\\n📝 Report saved to {report_file}")

    feature_report_file = "reports/v0.05_feature_analysis.json"
    feature_data = {}
    for t, metrics in full_report.items():
        feature_data[str(t)] = metrics.get('feature_importance', {})
    
    with open(feature_report_file, 'w', encoding='utf-8') as f:
        json.dump(feature_data, f, indent=4)
    print(f"✅ Feature Analysis Saved: {feature_report_file}")
    
    return stats

def train_meta_model(df, models, target=50.0):
    """
    Trains a simple Ensemble Meta-Model on the 'Hidden' 15% data.
    """
    print(f"\n--- Training Meta-Model (Ensemble) for {target}x ---")
    
    y_col = f'target_{str(target).replace(".","_")}'
    if y_col not in df.columns:
        print(f"❌ Target column {y_col} not found in dataframe. Skipping Meta-Model.")
        return None

    features = [c for c in df.columns if 'target' not in c and 'result' not in c and 'value' not in c and 'id' not in c]
    X = df[features]
    
    n = len(df)
    meta_start = int(n * 0.85)
    
    X_meta = X.iloc[meta_start:].copy()
    y_meta = df[y_col].iloc[meta_start:]
    
    meta_features = pd.DataFrame(index=X_meta.index)
    
    print("Generating predictions from sub-models...")
    for t_sub, model in models.items():
        try:
            sub_feats = get_model_features(t_sub, df.columns)
            X_sub = X_meta[sub_feats]
            preds = model.predict(X_sub)
            meta_features[f'pred_{t_sub}'] = preds
        except Exception as e:
            print(f"Skipping model {t_sub}x in ensemble: {e}")
            
    if meta_features.empty:
        print("No sub-model predictions available.")
        return None
        
    meta_features['Actual'] = y_meta
    print("Ensemble Data Prepared. (Showing first 5 rows)")
    print(meta_features.head())
    
    return meta_features

def walk_forward_validation(df, target, params):
    """
    Performs Walk-Forward Validation (Rolling Window) to test strategy robustness over time.
    """
    print(f"\n🚶‍♂️ --- WALK-FORWARD VALIDATION (Target: {target}x) ---")
    
    n = len(df)
    train_window_size = int(n * 0.40) 
    test_window_size = int(n * 0.10)
    gap = 500 
    
    start_idx = 0
    fold = 1
    
    results = []
    
    features = get_model_features(target, df.columns)
    y_col = f'target_{str(target).replace(".","_")}'
    scoring = get_scoring_params(target)
    
    print(f"   Dataset: {n} rows | Train: {train_window_size} | Gap: {gap} | Test: {test_window_size}")
    
    print(f"{'Fold':<6} | {'Profit':<10} | {'Win Rate':<10} | {'Trades':<8} | {'Thr':<6}")
    print("-" * 55)
    
    while True:
        train_end = start_idx + train_window_size
        test_start = train_end + gap
        test_end = test_start + test_window_size
        
        if test_end > n:
            break
        
        train_slice = df.iloc[start_idx:train_end]
        test_slice = df.iloc[test_start:test_end]
        
        sub_train_end = int(len(train_slice) * 0.80)
        
        sub_X_train = train_slice[features].iloc[:sub_train_end]
        sub_y_train = train_slice[y_col].iloc[:sub_train_end]
        
        sub_X_val = train_slice[features].iloc[sub_train_end:]
        sub_y_val = train_slice[y_col].iloc[sub_train_end:]
        
        X_test = test_slice[features]
        y_test = test_slice[y_col]
        
        try:
            d_train = lgb.Dataset(sub_X_train, label=sub_y_train)
            d_val = lgb.Dataset(sub_X_val, label=sub_y_val, reference=d_train)
            
            lgb_params = params.copy()
            
            model = lgb.train(
                lgb_params,
                d_train,
                valid_sets=[d_val],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0) # Silent
                ]
            )
            
            val_preds = model.predict(sub_X_val)
            best_thr = 0.5
            best_profit_val = -float('inf')
            
            start_thr = 0.50 if target <= 3.0 else 0.10
            
            val_df = pd.DataFrame({'prob': val_preds, 'actual': sub_y_val.values})
            
            for thr in np.arange(start_thr, 0.99, 0.02):
                tp = ((val_df['prob'] > thr) & (val_df['actual'] == 1)).sum()
                fp = ((val_df['prob'] > thr) & (val_df['actual'] == 0)).sum()
                profit_score = (tp * scoring['TP']) - (fp * scoring['FP'])
                if profit_score > best_profit_val:
                    best_profit_val = profit_score
                    best_thr = thr
            
            test_preds = model.predict(X_test)
            
            temp_res = pd.DataFrame({'prob': test_preds, 'actual': y_test.values})
            temp_res['action'] = (temp_res['prob'] > best_thr).astype(int)
            
            tp = ((temp_res['action'] == 1) & (temp_res['actual'] == 1)).sum()
            fp = ((temp_res['action'] == 1) & (temp_res['actual'] == 0)).sum()
            
            played = tp + fp
            win_rate = (tp / played * 100) if played > 0 else 0.0
            
            profit_mult = target - 1.0
            net_profit = (tp * profit_mult) - (fp * 1.0)
            
            print(f"{fold:<6} | {net_profit:<10.1f} | {win_rate:<9.1f}% | {played:<8d} | {best_thr:<6.2f}")
            
            results.append({
                'fold': int(fold),
                'profit': float(net_profit),
                'win_rate': float(win_rate),
                'trades': int(played)
            })
            
        except Exception as e:
            print(f"   Fold {fold}: Error - {e}")
            import traceback
            traceback.print_exc()
        
        start_idx += test_window_size
        fold += 1
        
    if not results:
        print("❌ WFV Failed: No results.")
        return None
        
    avg_profit = float(np.mean([r['profit'] for r in results]))
    avg_wr = float(np.mean([r['win_rate'] for r in results]))
    total_trades = int(sum([r['trades'] for r in results]))
    
    print("-" * 55)
    print(f"📊 AVG   | {avg_profit:<10.1f} | {avg_wr:<9.1f}% | {total_trades:<8}")
    print("-" * 55)
    
    return {'avg_profit': avg_profit, 'avg_win_rate': avg_wr, 'details': results}

def run_training():
    print("🦅 AVCI LOCAL TRAINING ORCHESTRATOR STARTED (v0.7.0 - Total Recall) 🦅")
    print("\n📂 Loading Data...")
    df = load_and_prep(limit=200000)
    if df is None: return

    trained_models = {}
    summary_stats = []
    
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING TARGET: {target}x")
        print(f"{'='*60}")
        
        best_params = optimize_target(df, target, n_trials=100)
        
        if not best_params:
            print(f"⚠️ Skipping {target}x due to optimization failure.")
            continue
            
        model, X_val, y_val = train_target_final(df, target, best_params)
        trained_models[target] = model
        
        wfv_stats = walk_forward_validation(df, target, best_params)
        
        stats = visualize_performance(model, X_val, y_val, target, wfv_stats=wfv_stats)
        
        status = "✅ APPROVED" if ((wfv_stats and wfv_stats['avg_profit'] > 0) or float(stats['net_profit']) > 0) else "❌ REJECTED"
        if wfv_stats and wfv_stats['avg_profit'] <= 0 and float(stats['chain_profit']) > 0:
             status = "⚙️ CHAIN ONLY"
             
        stats['status'] = status
        summary_stats.append(stats)

    print("\n" + "="*80)
    print("🎓 FINAL TRAINING SUMMARY REPORT")
    print(f"{'Target':<8} | {'WFV Profit':<12} | {'Chain Profit':<12} | {'WFV Win%':<10} | {'Status':<15}")
    print("-" * 80)
    for s in summary_stats:
        wfv_p = s.get('REAL_wfv_net_profit', 'N/A')
        chain_p = s.get('chain_profit', 'N/A')
        wfv_wr = s.get('REAL_wfv_win_rate', 'N/A')
        status = s.get('status', 'Unknown')
        print(f"{s['target']:<8} | {wfv_p:<12} | {chain_p:<12} | {wfv_wr:<10} | {status:<15}")
    print("-" * 80)
    print("\n✅ ALL LOCAL TRAINING COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_training()
