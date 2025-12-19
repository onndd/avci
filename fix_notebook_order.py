import nbformat as nbf
import os

nb_path = "Avci_Trainer.ipynb"

def reorder_notebook():
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}")
        return

    nb = nbf.read(nb_path, as_version=4)
    
    # Store cells by type/content to identify them
    setup_cells = []
    model_cells = {} # Key: Target (e.g., 2.0, 30.0) -> List of cells [Opt, Train]
    other_cells = []
    
    # Temporary lists to hold cells we want to move
    cells_30 = []
    cells_40 = []
    
    # Filter out the 30 and 40 cells from the main list first
    kept_cells = []
    
    for cell in nb.cells:
        src = cell.source
        if "target = 30.0" in src or "train_target_final(df, 30.0" in src:
            cells_30.append(cell)
        elif "target = 40.0" in src or "train_target_final(df, 40.0" in src:
            cells_40.append(cell)
        else:
            kept_cells.append(cell)
            
    if not cells_30 or not cells_40:
        print("Could not find 30x/40x cells to move. Aborting.")
        return

    # Now find the insertion point: After 20.0x cells
    insert_idx = -1
    for i, cell in enumerate(kept_cells):
        if "target = 20.0" in cell.source or "visualize_performance" in cell.source and ", 20.0)" in cell.source:
            # We want to insert AFTER the LAST 20.0 block
            # This logic is a bit loose, better to find the START of 50.0 and insert BEFORE it
            pass
        if "target = 50.0" in cell.source:
            insert_idx = i
            break
            
    if insert_idx == -1:
        # If 50.0 not found, look for Ensemble
        for i, cell in enumerate(kept_cells):
             if "Ensemble" in cell.source:
                insert_idx = i
                break
    
    if insert_idx != -1:
        print(f"Inserting 30x/40x cells at index {insert_idx} (Before 50.0x or Ensemble)")
        # Insert 30 then 40
        kept_cells[insert_idx:insert_idx] = cells_30 + cells_40
        
        nb.cells = kept_cells
        nbf.write(nb, nb_path)
        print(f"Notebook reordered. Saved to {nb_path}")
    else:
        print("Could not find a valid insertion point (Before 50.0 or Ensemble).")

if __name__ == "__main__":
    reorder_notebook()
