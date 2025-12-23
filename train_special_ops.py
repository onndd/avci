from train_avci import load_and_prep, optimize_target, train_target_final
import pandas as pd

def run_special_ops():
    print("🚀 Starting Special Ops: Optimizing 2x and 3x with Zig-Zag & Fakeout Features...")
    
    # 1. Load Data with New Features (Zig-Zag, Fakeout are calculated in extract_features called by load_and_prep)
    df = load_and_prep(limit=100000)
    
    targets = [2.0, 3.0]
    
    for t in targets:
        print(f"\n⚡ PROCESSING TARGET: {t}x ⚡")
        
        # 2. Find Best Hyperparameters for New Features
        # We run a quick optimization (15 trials) because new features might need different tree depth etc.
        study = optimize_target(df, target=t, n_trials=15)
        best_params = study.best_params
        
        print(f"✅ Optimization Complete for {t}x. Params: {best_params}")
        
        # 3. Train Final Model and Save
        train_target_final(df, target=t, best_params=best_params)
        
    print("\n🦅 Special Ops Mission Accomplished. New models are live in models/.")

if __name__ == "__main__":
    run_special_ops()
