import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

from train_avci import load_and_prep, optimize_target, train_target_final, visualize_performance, train_meta_model
from config_avci import TARGETS

def main():
    print("🚀 Starting Local Training Pipeline...")
    
    # 1. Load Data
    try:
        df = load_and_prep(limit=200000) # Use a decent amount of data
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    trained_models = {}

    # 2. Train Each Target
    for target in TARGETS: # [3.0, 5.0, 10.0, 20.0, 50.0]
        print(f"\n\n{'='*50}")
        print(f"🎯 PROCESSING TARGET: {target}x")
        print(f"{'='*50}\n")
        
        try:
            # Optimize
            study = optimize_target(df, target, n_trials=100) # 100 trials for full training
            best_params = study.best_params
            
            # WFV (Validation)
            from train_avci import walk_forward_validation
            wfv_stats = walk_forward_validation(df, target, best_params)
            
            # Train Final
            model, X_val, y_val = train_target_final(df, target, best_params)
            trained_models[target] = model
            
            # Visualize / Report
            visualize_performance(model, X_val, y_val, target, wfv_stats=wfv_stats)
            
        except Exception as e:
            print(f"❌ Failed to train target {target}x: {e}")
            import traceback
            traceback.print_exc()

    # 3. Train Meta Model (Optional for now, but good to check if it runs)
    try:
        train_meta_model(df, trained_models)
    except Exception as e:
        print(f"Error training meta model: {e}")

    print("\n✅ Local Training Pipeline Completed.")

if __name__ == "__main__":
    main()
