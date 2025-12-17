
import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path, limit=50000):
    """
    Loads data from local jetx.db
    """
    import os
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Veritabanı dosyası bulunamadı: {os.path.abspath(db_path)}")
    
    if os.path.getsize(db_path) == 0:
        raise ValueError(f"Veritabanı dosyası BOŞ (0 byte): {db_path}")

    conn = sqlite3.connect(db_path)
    
    # Check if table exists
    cursor = conn.cursor()
    # Check for 'jetx_results' (New) or 'game_results' (Old/Fallback)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jetx_results';")
    table_exists = cursor.fetchone()
    
    table_name = 'jetx_results'
    if not table_exists:
        # Fallback check
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_results';")
        if cursor.fetchone():
            table_name = 'game_results'
        else:
            conn.close()
            raise ValueError(f"Veritabanı Uyumsuz: 'jetx_results' veya 'game_results' tablosu bulunamadı! DB: {db_path}")

    # conn = sqlite3.connect(db_path) # Already connected
    query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Sort by ID ascending (Oldest first for Time Series)
    df = df.sort_values('id').reset_index(drop=True)
    
    # Parse Crash Point
    # If 'value' column exists (New DB), use it.
    # If 'result' column exists (Old DB), parse "1.23x".
    
    col_to_use = 'value' if 'value' in df.columns else 'result'
    if col_to_use not in df.columns:
         raise ValueError("Veri şeması hatalı: 'value' veya 'result' sütunu bulunamadı.")

    # Robust Cleaning Function
    def clean_currency(x):
        if pd.isna(x): return np.nan
        s = str(x)
        # Remove 'x' and replace parsed line separators with space
        s = s.replace('x', '').replace('\u2028', ' ').replace('\n', ' ').strip()
        # If multiple numbers exist (e.g. "2.29 1.29"), take the first one
        parts = s.split()
        if not parts: return np.nan
        try:
            return float(parts[0])
        except:
            return np.nan

    df['value'] = df[col_to_use].apply(clean_currency)
    
    # Drop rows where value could not be parsed
    df = df.dropna(subset=['value']).reset_index(drop=True)
    
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
