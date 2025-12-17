
# Avci Project Configuration

# Window sizes for feature extraction (Expanded for rare signals)
WINDOWS = [25, 50, 100, 150, 200, 300]

# Scoring Weights for various targets
# Note: For High Multipliers (5x, 10x), loss is minimal (1 unit) compared to huge gain.
# But we must avoid "false hope" spam.

SCORING_1_5 = {'TP': 90, 'TN': 1, 'FP': 170, 'FN': 10, 'PRECISION': 75}
SCORING_2_0 = {'TP': 150, 'TN': 1, 'FP': 150, 'FN': 20, 'PRECISION': 80}
SCORING_3_0 = {'TP': 400, 'TN': 1, 'FP': 143, 'FN': 50, 'PRECISION': 100}

# High Multipliers - Aggressive Reward but Strict Precision Bonus
SCORING_5_0 = {'TP': 800, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 150}
SCORING_10_0 = {'TP': 2000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 200}
SCORING_20_0 = {'TP': 5000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 300}
SCORING_50_0 = {'TP': 15000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 500}
SCORING_100_0 = {'TP': 40000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 1000}
SCORING_1000_0 = {'TP': 200000, 'TN': 1, 'FP': 150, 'FN': 50, 'PRECISION': 5000}

# Targets to train for
TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]

DB_PATH = 'jetx.db'
MODEL_DIR = 'models'
