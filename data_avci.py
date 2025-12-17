
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
    if 'value' in df.columns:
        df['value'] = df['value'].astype(float)
    elif 'result' in df.columns:
        df['value'] = df['result'].astype(str).str.replace('x', '', regex=False).astype(float)
    else:
        raise ValueError("Veri şeması hatalı: 'value' veya 'result' sütunu bulunamadı.")
    
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
