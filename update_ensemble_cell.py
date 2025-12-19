import nbformat as nbf
import os

nb_path = "Avci_Trainer.ipynb"

def update_ensemble():
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}")
        return

    nb = nbf.read(nb_path, as_version=4)
    found = False
    
    # Target content to find (the old dictionary)
    # We look for the cell containing 'models = {' and 'train_meta_model'
    
    for cell in nb.cells:
        if cell.cell_type == 'code' and "train_meta_model" in cell.source and "models = {" in cell.source:
            
            # Construct new source code with 30 and 40 included
            new_source = """# 7. Meta-Model (Ensemble) Eğitimi
# Tüm alt modellerin tahminlerini birleştirir.

models = {
    2.0: model_2,
    3.0: model_3,
    5.0: model_5,
    10.0: model_10,
    20.0: model_20,
    30.0: model_30,
    40.0: model_40,
    50.0: model_50
}

# Meta Model Hedefi: Genelde 100x veya güvenli bir 'Süper Sinyal' olabilir.
# Şimdilik 50.0x üzerindeki başarıyı maksimize etmeye çalışalım.
meta_results = train_meta_model(df, models, target=50.0)

if meta_results is not None:
    print(meta_results.tail(10))"""
            
            cell.source = new_source
            found = True
            print("Ensemble cell updated with 30x and 40x models.")
            break
            
    if found:
        nbf.write(nb, nb_path)
        print(f"Notebook saved: {nb_path}")
    else:
        print("Ensemble cell not found. Please check manually.")

if __name__ == "__main__":
    update_ensemble()
