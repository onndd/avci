import json
import os

metrics_path = 'reports/v0.9.4_training_metrics.json'
feature_path = 'reports/v0.9.4_feature_analysis.json'

print(f"Reading from {metrics_path}...")

if not os.path.exists(metrics_path):
    print("Error: Metrics file not found!")
    exit(1)

with open(metrics_path, 'r') as f:
    data = json.load(f)

feature_analysis = {}

for target, stats in data.items():
    if 'feature_importance' in stats:
        feature_analysis[target] = stats['feature_importance']
        print(f"Extracted features for {target}x")
    else:
        print(f"Warning: No feature importance found for {target}x")

print(f"Saving to {feature_path}...")
with open(feature_path, 'w') as f:
    json.dump(feature_analysis, f, indent=4)

print("✅ Recovery Complete.")
