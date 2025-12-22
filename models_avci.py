
import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import confusion_matrix

def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a single LightGBM model.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 1000
        }
        
    # Fixed Seed for Reproducibility
    params['seed'] = 42
    params['feature_fraction_seed'] = 42
    params['bagging_seed'] = 42
    params['deterministic'] = True
    
    # GPU Support
    if params.get('device') == 'gpu':
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks
    )
    return model

def calculate_max_streak(y_true, y_pred, target_val):
    """
    Calculates the maximum consecutive WIN streak and returns a bonus score.
    Returns (streak_count, bonus_points)
    """
    # Filter only plays (where y_pred is 1)
    # acts contains 1 (Win) or 0 (Loss) for played games
    # We only care about streaks of WINS (1s) within the played set.
    
    # We need to map y_true to y_pred.
    # y_pred is the action. y_true is the result.
    # A "Win" is when y_pred=1 and y_true=1.
    
    # Create array of Results where Action=1
    # 1 = Win, 0 = Loss
    
    # Efficient numpy approach
    # Get indices where action is taken
    action_indices = np.where(y_pred == 1)[0]
    
    if len(action_indices) == 0:
        return 0, 0
        
    # Get actual outcomes for these actions
    outcomes = y_true.iloc[action_indices].values if hasattr(y_true, 'iloc') else y_true[action_indices]
    
    # Find streaks of 1s in 'outcomes'
    # e.g. [1, 1, 0, 1, 1, 1, 0] -> Max streak 3
    
    # Pad with 0 to handle edge cases
    padded = np.concatenate(([0], outcomes, [0]))
    diffs = np.diff(padded)
    
    # Starts are where value goes 0->1 (diff=1)
    # Ends are where value goes 1->0 (diff=-1)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    if len(starts) == 0:
        return 0, 0
        
    streak_lengths = ends - starts
    max_streak = np.max(streak_lengths)
    
    # --- BONUS LOGIC ---
    # 3.0x -> Target 4 Streak (81x)
    # 5.0x -> Target 3 Streak (125x)
    # 10.0x+ -> Target 2 Streak
    
    required_streak = 2
    if target_val <= 3.0: required_streak = 4
    elif target_val <= 5.0: required_streak = 3
    
    bonus = 0
    if max_streak >= required_streak:
        # Exponential Reward: (Streak - Requirement + 1) * 1000
        # e.g. Req 4. Got 4 -> 1000
        # Got 5 -> 2000
        multiplier = (max_streak - required_streak + 1)
        bonus = multiplier * 1000
        
    return max_streak, bonus

def objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring_params, use_gpu=True, extra_params=None):
    """
    Optuna Objective Function.
    Optimizes for HYBRID SCORE (Profit + Streak Bonus).
    """
    # Determine Target Value from scoring params (Approximation)
    tp_val = scoring_params['TP']
    target_est = 2.0 # Default
    if tp_val >= 400: target_est = 5.0
    if tp_val >= 1000: target_est = 10.0
    if tp_val >= 2000: target_est = 20.0
    if tp_val >= 5000: target_est = 50.0
    
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'seed': 42,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'deterministic': True
    }
    
    # Class Weighting Strategy for High Curve (v0.9.8)
    # Class Weighting Strategy for High Curve (v0.9.8 & v0.9.9)
    if target_est >= 30.0:
        # AGGRESSIVE ADRENALINE for Ghost Models (30x, 40x, 50x)
        # They are too timid. Force them to care about the 1 positive sample.
        weight = trial.suggest_categorical('scale_pos_weight', [100.0, 200.0, 500.0, 1000.0])
        param['scale_pos_weight'] = weight
    elif target_est >= 10.0:
        # Moderate boost for 10x, 20x
        weight = trial.suggest_categorical('scale_pos_weight', [1.0, 10.0, 25.0, 50.0, 100.0])
        param['scale_pos_weight'] = weight

    if extra_params:
        param.update(extra_params)
    
    if use_gpu:
        param['device'] = 'gpu'
        param['gpu_platform_id'] = 0
        param['gpu_device_id'] = 0

    try:
        model = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)])
    except lgb.basic.LightGBMError:
        if use_gpu:
            print("Warning: GPU failed, falling back to CPU.")
            param['device'] = 'cpu'
            model = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)])
        else:
            raise
            
    preds_proba = model.predict(X_val)
    
    # Find Best Threshold
    best_final_score = -float('inf')
    thresholds = np.arange(0.10, 0.99, 0.01)
    
    for thr in thresholds:
        preds = (preds_proba > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        
        # 1. Base Profit Score
        precision_score = tp / (tp + fp + 1e-9)
        base_score = (tp * scoring_params['TP']) + \
                     (tn * scoring_params['TN']) - \
                     (fp * scoring_params['FP']) - \
                     (fn * scoring_params['FN']) + \
                     (precision_score * scoring_params.get('PRECISION', 0))
                     
        # 2. Hybrid Streak Bonus
        # Only check for bonus if base strategy is profitable/valid (Score > 0)
        # We don't want to reward lucky streaks in a losing model.
        streak_bonus = 0
        if base_score > 0:
             _, streak_bonus = calculate_max_streak(y_val, preds, target_est)
             
        final_score = base_score + streak_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            
    return best_final_score
