
# Avci Project Configuration

# Window sizes for feature extraction (User requested: 15, 25, 50)
WINDOWS = [10, 15, 25, 50, 100, 200, 300]

# Scoring Weights for various targets
# Note: For High Multipliers (5x, 10x), loss is minimal (1 unit) compared to huge gain.
# But we must avoid "false hope" spam.

# SCORING_1_5 Removed.

SCORING_2_0 = {'TP': 135, 'TN': 25, 'FP': 105, 'FN': 100, 'PRECISION': 90}  # Strict FP (was 165)
SCORING_3_0 = {'TP': 450, 'TN': 20, 'FP': 350, 'FN': 80, 'PRECISION': 100} # Strict FP (was 220)

# High Multipliers - Aggressive Reward but Strict Precision Bonus
SCORING_5_0 = {'TP': 890, 'TN': 45, 'FP': 250, 'FN': 95, 'PRECISION': 150}
SCORING_10_0 = {'TP': 1950, 'TN': 1, 'FP': 250, 'FN': 60, 'PRECISION': 200}
SCORING_20_0 = {'TP': 8500, 'TN': 1, 'FP': 225, 'FN': 195, 'PRECISION': 300}
SCORING_30_0 = {'TP': 9900, 'TN': 1, 'FP': 285, 'FN': 170, 'PRECISION': 350}  # Reverted: Fix Features instead of Scoring
SCORING_40_0 = {'TP': 17500, 'TN': 1, 'FP': 450, 'FN': 200, 'PRECISION': 450} # Reverted
SCORING_50_0 = {'TP': 22500, 'TN': 1, 'FP': 500, 'FN': 350, 'PRECISION': 500} # FP reduced from 750
SCORING_100_0 = {'TP': 45000, 'TN': 1, 'FP': 125, 'FN': 50, 'PRECISION': 1000}
SCORING_1000_0 = {'TP': 200000, 'TN': 1, 'FP': 150, 'FN': 40, 'PRECISION': 2000}

# Targets to train for (Removed 1.5, Added 30, 40)
TARGETS = [2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]

# UI Thresholds for Card Colors (Target-Specific Overrides Supported in App)
# Default is safely conservative.
# UI Thresholds for Card Colors (Updated from Latest Training - v0.2.12)
CARD_THRESHOLDS = {
    'DEFAULT': {'GOLD': 0.85, 'RISK': 0.60},
    2.0: {'GOLD': 0.58, 'RISK': 0.45},    # High Safety
    3.0: {'GOLD': 0.48, 'RISK': 0.38},    # High Safety
    5.0: {'GOLD': 0.27, 'RISK': 0.20},    # v0.03 Smart (New Features)
    10.0: {'GOLD': 0.41, 'RISK': 0.35},   # v0.03 Smart (+503 Profit)
    20.0: {'GOLD': 0.10, 'RISK': 0.08},   # v0.03 Smart (+213 Profit)
    30.0: {'GOLD': 0.32, 'RISK': 0.25},   # High Safety
    40.0: {'GOLD': 0.22, 'RISK': 0.18},   # v0.03 Smart (+308 Profit)
    50.0: {'GOLD': 0.25, 'RISK': 0.20}    # v0.03 Smart (+787 Profit)
}

DB_PATH = 'jetx.db'
MODEL_DIR = 'models'
