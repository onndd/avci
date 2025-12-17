
import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path, limit=50000):
    """
    Loads data from local jetx.db
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM game_results ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Sort by ID ascending (Oldest first for Time Series)
    df = df.sort_values('id').reset_index(drop=True)
    
    # Parse Crash Point
    # Assumes format "1.23x" or similar. Clean it.
    df['value'] = df['result'].astype(str).str.replace('x', '', regex=False).astype(float)
    
    return df

def add_targets(df, targets):
    """
    Adds binary targets for each customized threshold.
    """
    for t in targets:
        col_name = f'target_{str(t).replace(".","_")}'
        df[col_name] = (df['value'] >= t).astype(int)
    
    # Regression Target (Next Value) - Shifted
    # We want to predict current row's value based on PREVIOUS rows.
    # But usually features are shifted. Here we label the row with its own outcome.
    return df
