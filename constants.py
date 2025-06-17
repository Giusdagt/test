"""
constants.py
Impostazioni globali dinamiche e robuste per IA e sistema.
"""

import polars as pl
import os

print("constants.py caricato ✅")

# Se True, la rete adatta dinamicamente la dimensione dello stato
USE_DYNAMIC_STATE_SIZE = False

# Lunghezza della sequenza temporale usata dall’ambiente di training (e IA)
SEQUENCE_LENGTH = 50

# Calcolo dinamico robusto della dimensione dello stato
def compute_dynamic_state_size():
    try:
        data_path = "D:/trading_data/processed_data.zstd.parquet"
        if not os.path.exists(data_path):
            print("⚠️ File dati non trovato, uso valore di default.")
            return 8190  # Fallback

        df = pl.read_parquet(data_path).filter(pl.col("timeframe") == "1m")
        excluded_cols = {"timestamp", "symbol", "timeframe"}
        valid_cols = []

        for col in df.columns:
            if col in excluded_cols:
                continue
            try:
                # Esclude colonne costanti o con solo null/nan
                if df[col].null_count() == df.height:
                    continue
                if df[col].max() == df[col].min():
                    continue
                valid_cols.append(col)
            except Exception:
                continue  # Colonna non gestibile, la saltiamo

        if not valid_cols:
            print("⚠️ Nessuna colonna valida trovata, uso fallback.")
            return 8190

        size = SEQUENCE_LENGTH * len(valid_cols)
        print(f"📐 DESIRED_STATE_SIZE calcolato dinamicamente: {size}")
        return size

    except Exception as e:
        print(f"❌ Errore durante il calcolo DESIRED_STATE_SIZE: {e}")
        return 8190

# Applica il calcolo dinamico o fallback statico
DESIRED_STATE_SIZE = compute_dynamic_state_size() if USE_DYNAMIC_STATE_SIZE else 8190
