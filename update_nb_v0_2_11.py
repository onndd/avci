import nbformat as nbf
import os

nb_path = "Avci_Trainer.ipynb"

# --- Define New Cells for 30.0x and 40.0x ---

# 30.0 Target Cells
optimization_30_cell = nbf.v4.new_code_cell("""# --- 30.0 (High Multiplier) ---
target = 30.0
print(f"\\n🔍 --- Optimizing for Target {target}x ---")
study_30 = optimize_target(df, target, n_trials=30) # Reduced trials for high x
best_params_30 = study_30.best_trial.params
print(f"✅ Best Params for {target}x: {best_params_30}")""")

training_30_cell = nbf.v4.new_code_cell("""# Train 30.0 Final Model
model_30, X_val_30, y_val_30 = train_target_final(df, 30.0, best_params_30)
visualize_performance(model_30, X_val_30, y_val_30, 30.0)""")

# 40.0 Target Cells
optimization_40_cell = nbf.v4.new_code_cell("""# --- 40.0 (Ultra High Multiplier) ---
target = 40.0
print(f"\\n🔍 --- Optimizing for Target {target}x ---")
study_40 = optimize_target(df, target, n_trials=30)
best_params_40 = study_40.best_trial.params
print(f"✅ Best Params for {target}x: {best_params_40}")""")

training_40_cell = nbf.v4.new_code_cell("""# Train 40.0 Final Model
model_40, X_val_40, y_val_40 = train_target_final(df, 40.0, best_params_40)
visualize_performance(model_40, X_val_40, y_val_40, 40.0)""")


def update_notebook():
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}")
        return

    nb = nbf.read(nb_path, as_version=4)
    
    # Identify insert position: Before the Ensemble/Meta-Model Step
    insert_idx = -1
    for idx, cell in enumerate(nb.cells):
        if "Ensemble" in cell.source or "train_meta_model" in cell.source:
            insert_idx = idx
            break
            
    if insert_idx == -1:
        print("Could not find Ensemble cell to insert before. Appending to end.")
        insert_idx = len(nb.cells)

    # Insert cells in order: 30 Opt, 30 Train, 40 Opt, 40 Train
    # We insert in reverse order at the same index effectively
    
    # 30 and 40
    new_cells = [
        optimization_30_cell,
        training_30_cell,
        optimization_40_cell,
        training_40_cell
    ]
    
    # Check if they already exist to avoid duplicates
    existing_sources = [c.source for c in nb.cells]
    final_new_cells = []
    
    for c in new_cells:
        if c.source not in existing_sources:
            final_new_cells.append(c)
        else:
            print(f"Skipping duplicate cell: {c.source[:30]}...")

    if final_new_cells:
        # Insert
        nb.cells[insert_idx:insert_idx] = final_new_cells
        
        # Update Version Header if exists
        if nb.cells[0].cell_type == 'markdown':
             nb.cells[0].source = "# 🦅 AVCI - Model Eğitim Notebook'u (v0.2.11)\n\n**Güncellemeler:**\n- **30x ve 40x** Modelleri Eklendi.\n- **Güven Aralığı Analizi** Eklendi.\n- Feature Window [15, 25, 50...] Güncellendi."
        
        nbf.write(nb, nb_path)
        print(f"Notebook updated to v0.2.11 with 30x/40x models. saved to {nb_path}")
    else:
        print("No new cells to add.")

if __name__ == "__main__":
    update_notebook()
