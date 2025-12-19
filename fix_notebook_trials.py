import nbformat as nbf
import os
import re

nb_path = "Avci_Trainer.ipynb"

def fix_notebook_trials():
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}")
        return

    nb = nbf.read(nb_path, as_version=4)
    changes_made = 0
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            src = cell.source
            
            # Pattern 1: epochs=EPOCHS -> n_trials=100
            if "epochs=EPOCHS" in src:
                src = src.replace("epochs=EPOCHS", "n_trials=100")
                changes_made += 1
                
            # Pattern 2: n_trials=30 (My previous 30x/40x blocks) -> n_trials=100
            if "n_trials=30" in src:
                src = src.replace("n_trials=30", "n_trials=100")
                changes_made += 1
                
            # Pattern 3: epochs=50 -> n_trials=100
            if "epochs=50" in src:
                src = src.replace("epochs=50", "n_trials=100")
                changes_made += 1
            
            # Pattern 4: epochs=20 -> n_trials=100
            if "epochs=20" in src:
                src = src.replace("epochs=20", "n_trials=100")
                changes_made += 1

            # General Safety: optimize_target(..., epochs=...) -> n_trials=100
            # This handles cases like epochs=10 or variable usage
            # Regex to find 'epochs=X' or 'epochs=var' inside optimize_target
            # Be careful not to break things, but 'epochs=' is the offender.
            
            if "optimize_target" in src and "epochs=" in src:
                 src = re.sub(r'epochs=\w+', 'n_trials=100', src)
                 changes_made += 1

            cell.source = src

    if changes_made > 0:
        nbf.write(nb, nb_path)
        print(f"✅ Fixed {changes_made} cells. All set to n_trials=100.")
    else:
        print("No changes needed (or patterns not found).")

if __name__ == "__main__":
    fix_notebook_trials()
