
# Avci Project Configuration

# Window sizes for feature extraction (Expanded for rare signals)
WINDOWS = [25, 50, 100, 150, 200, 300]

# Scoring Weights for various targets
# Note: For High Multipliers (5x, 10x), loss is minimal (1 unit) compared to huge gain.
# But we must avoid "false hope" spam.

# SCORING_1_5 Removed.

SCORING_2_0 = {'TP': 150, 'TN': 1, 'FP': 150, 'FN': 20, 'PRECISION': 80}
SCORING_3_0 = {'TP': 400, 'TN': 10, 'FP': 185, 'FN': 50, 'PRECISION': 100}

# High Multipliers - Aggressive Reward but Strict Precision Bonus
SCORING_5_0 = {'TP': 800, 'TN': 25, 'FP': 250, 'FN': 50, 'PRECISION': 150}
SCORING_10_0 = {'TP': 2000, 'TN': 1, 'FP': 550, 'FN': 50, 'PRECISION': 200}
SCORING_20_0 = {'TP': 5000, 'TN': 1, 'FP': 600, 'FN': 50, 'PRECISION': 300}
SCORING_50_0 = {'TP': 16000, 'TN': 1, 'FP': 750, 'FN': 50, 'PRECISION': 500}
SCORING_100_0 = {'TP': 40000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 1000}
SCORING_1000_0 = {'TP': 200000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 5000}

# Targets to train for (Removed 1.5)
TARGETS = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

# UI Thresholds for Card Colors (Target-Specific Overrides Supported in App)
# Default is safely conservative.
CARD_THRESHOLDS = {
    'DEFAULT': {'GOLD': 0.85, 'RISK': 0.60},
    10.0: {'GOLD': 0.40, 'RISK': 0.30},   # Manual Adjustment
    100.0: {'GOLD': 0.15, 'RISK': 0.10},  # Aggressive Sniper Mode
    1000.0: {'GOLD': 0.10, 'RISK': 0.05}
}

DB_PATH = 'jetx.db'
MODEL_DIR = 'models'
