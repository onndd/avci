
# Avci Project Configuration

# Window sizes for feature extraction (User requested: 15, 25, 50)
WINDOWS = [15, 25, 50, 100, 200, 300]

# Scoring Weights for various targets
# Note: For High Multipliers (5x, 10x), loss is minimal (1 unit) compared to huge gain.
# But we must avoid "false hope" spam.

# SCORING_1_5 Removed.

SCORING_2_0 = {'TP': 125, 'TN': 1, 'FP': 150, 'FN': 35, 'PRECISION': 80}
SCORING_3_0 = {'TP': 400, 'TN': 10, 'FP': 185, 'FN': 50, 'PRECISION': 100}

# High Multipliers - Aggressive Reward but Strict Precision Bonus
SCORING_5_0 = {'TP': 800, 'TN': 25, 'FP': 200, 'FN': 65, 'PRECISION': 150}
SCORING_10_0 = {'TP': 2000, 'TN': 1, 'FP': 350, 'FN': 50, 'PRECISION': 200}
SCORING_20_0 = {'TP': 5000, 'TN': 1, 'FP': 300, 'FN': 50, 'PRECISION': 300}
SCORING_30_0 = {'TP': 8000, 'TN': 1, 'FP': 450, 'FN': 50, 'PRECISION': 350}  # New
SCORING_40_0 = {'TP': 12000, 'TN': 1, 'FP': 600, 'FN': 50, 'PRECISION': 450} # New
SCORING_50_0 = {'TP': 16000, 'TN': 1, 'FP': 800, 'FN': 50, 'PRECISION': 500}
SCORING_100_0 = {'TP': 40000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 1000}
SCORING_1000_0 = {'TP': 200000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 5000}

# Targets to train for (Removed 1.5, Added 30, 40)
TARGETS = [2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]

# UI Thresholds for Card Colors (Target-Specific Overrides Supported in App)
# Default is safely conservative.
CARD_THRESHOLDS = {
    'DEFAULT': {'GOLD': 0.85, 'RISK': 0.60},
    2.0: {'GOLD': 0.75, 'RISK': 0.50},    # Strong signal for low x
    3.0: {'GOLD': 0.65, 'RISK': 0.40},
    5.0: {'GOLD': 0.50, 'RISK': 0.25},
    10.0: {'GOLD': 0.35, 'RISK': 0.15},
    20.0: {'GOLD': 0.25, 'RISK': 0.10},
    30.0: {'GOLD': 0.20, 'RISK': 0.08},   # New
    40.0: {'GOLD': 0.18, 'RISK': 0.06},   # New
    50.0: {'GOLD': 0.15, 'RISK': 0.05}    # 15% is huge for 50x
}

DB_PATH = 'jetx.db'
MODEL_DIR = 'models'
