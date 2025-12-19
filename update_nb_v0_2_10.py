import json
import os

nb_path = 'Avci_Trainer.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# 1. Define New Cells for 2.0x
cell_opt_2 = {
   "cell_type": "code",
   "execution_count": None,
   "id": "opt_2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.00x - OPTİMİZASYON 🛠️\n",
    "TARGET = 2.0\n",
    "EPOCHS = 50\n",
    "study_2, params_2 = optimize_target(df, TARGET, epochs=EPOCHS)"
   ]
}

cell_train_2 = {
   "cell_type": "code",
   "execution_count": None,
   "id": "train_2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.0x - FİNAL EĞİTİM 🏃\n",
    "model_2, X_val, y_val = train_target_final(df, TARGET, params_2)\n",
    "visualize_performance(model_2, X_val, y_val, TARGET)"
   ]
}

# 2. Find insertion point (After Markdown 'Bölüm 2', before 3.0x Optimize)
insert_idx = -1
for i, cell in enumerate(cells):
    src = "".join(cell.get('source', []))
    if "# 3.00x - OPTİMİZASYON" in src:
        insert_idx = i
        break

if insert_idx != -1:
    print(f"Inserting 2.0x cells at index {insert_idx}")
    cells.insert(insert_idx, cell_train_2) # Insert reverse order because inserting at same index pushes down
    cells.insert(insert_idx, cell_opt_2)
else:
    print("Could not find insertion point for 2.0x cells")

# 3. Update Ensemble Cell to include model_2
for cell in cells:
    src = "".join(cell.get('source', []))
    if "# ENSEMBLE (META-MODEL)" in src:
        new_source = []
        for line in cell['source']:
            if "if 'model_30' in locals(): trained_models[3.0] = model_30" in line:
                # Add 2.0 line before 3.0
                new_source.append("    if 'model_2' in locals(): trained_models[2.0] = model_2\n")
            new_source.append(line)
        cell['source'] = new_source
        print("Updated Ensemble cell.")
        break

# 4. Save
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
