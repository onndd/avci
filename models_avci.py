
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

def objective_lgbm(trial, X_train, y_train, X_val, y_val, scoring_params, use_gpu=True):
    """
    Optuna Objective Function.
    Optimizes for PROFIT SCORE using the weights in config_avci.
    """
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
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    if use_gpu:
        param['device'] = 'gpu'
        param['gpu_platform_id'] = 0
        param['gpu_device_id'] = 0

    try:
        model = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)])
    except lgb.basic.LightGBMError:
        # Fallback to CPU if GPU fails (common in some envs)
        if use_gpu:
            print("Warning: GPU failed, falling back to CPU.")
            param['device'] = 'cpu'
            model = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)])
        else:
            raise
    preds_proba = model.predict(X_val)
    
    # Find Best Threshold
    best_score = -float('inf')
    thresholds = np.arange(0.50, 0.99, 0.01)
    
    for thr in thresholds:
        preds = (preds_proba > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        # Profit Score Calculation
        precision_score = tp / (tp + fp + 1e-9)
        score = (tp * scoring_params['TP']) + \
                (tn * scoring_params['TN']) - \
                (fp * scoring_params['FP']) - \
                (fn * scoring_params['FN']) + \
                (precision_score * scoring_params.get('PRECISION', 0))
        if score > best_score:
            best_score = score
            
    return best_score
