# realtime_update.py
# Modulo per aggiornamento costante embedding + segnali strutturali dai dati realtime

import os
import polars as pl
import logging
from pathlib import Path
from market_fingerprint import compress_to_vector
from ai_features import get_features_by_strategy_type
from smart_features import apply_all_market_structure_signals

DATA_PATH = "D:/trading_data"
EMBEDDING_FILE = Path("D:/trading_data/embedding_data.zstd.parquet")


def update_realtime_embedding(symbol: str, df: pl.DataFrame, timeframe: str = "1m", length: int = 32):
    """
    Aggiorna embedding + segnali strutturali AI da dati realtime (senza salvare l'intero storico).
    """
    # Applica segnali strutturali
    enriched_df = apply_all_market_structure_signals(df)

    # Estrai features per la strategia AI (scalping, swing, macro)
    strategy_type = detect_strategy_type(timeframe)
    features = get_features_by_strategy_type(strategy_type)
    if not isinstance(features, list):
        features = [features]
    valid_cols = [col for col in features if col in enriched_df.columns]
    filtered_df = enriched_df.select(valid_cols) if valid_cols else enriched_df

    # Comprimi in vettore embedding
    vector = compress_to_vector(filtered_df, length=length, timeframe=timeframe)

    # Crea nuova riga per salvataggio
    embedding_cols = {f"embedding_{i}": [float(x)] for i, x in enumerate(vector)}
    row = pl.DataFrame({
        "symbol": [symbol],
        "timeframe": [timeframe],
        **embedding_cols
    })

    # Se il file esiste, caricalo e sostituisci riga esistente
    if EMBEDDING_FILE.exists():
        existing = pl.read_parquet(EMBEDDING_FILE)
        symbol_lit = pl.lit(symbol).cast(pl.Utf8())
        timeframe_lit = pl.lit(timeframe).cast(pl.Utf8())
        existing = existing.filter(
            ~(
                (pl.col("symbol") == symbol_lit) &
                (pl.col("timeframe") == timeframe_lit)
            )
        )

        updated = pl.concat([existing, row])
    else:
        updated = row

    # Salva file aggiornato
    updated.write_parquet(EMBEDDING_FILE, compression="zstd")
    print(f"✅ Embedding realtime aggiornato per {symbol} [{timeframe}]")


def detect_strategy_type(timeframe: str) -> str:
    tf = str(timeframe).lower()
    if tf in ["1m", "5m", "15m", "30m"]:
        return "scalping"
    elif tf in ["1h", "4h"]:
        return "swing"
    else:
        return "macro"

def update_memory_after_trade(symbol, embedding, confidence, trade_profit, action_rl, result):
    """
    Aggiorna la memoria AI dopo un trade, salvando embedding e risultato.
    """
    try:
        from ai_memory_sync import AIMemoryCompact
        memory_path = Path("D:/trading_data/ai_memory_sync.npz")
        if memory_path.exists():
            ai_memory = AIMemoryCompact.load(memory_path)
        else:
            ai_memory = AIMemoryCompact()
        if embedding is not None:
            ai_memory.update(trade_profit, embedding, confidence)
            ai_memory.save(memory_path)
            logging.info(f"🧠 Memoria AI aggiornata per {symbol} con risultato {result}")
    except Exception as e:
        logging.error(f"❌ Errore aggiornamento memoria dopo il trade: {e}")



#blocco non unasto, usato quello in market_fingerprint.py
def update_embedding_in_processed_file_veloce(symbol: str, timeframe: str, embedding: list[float]):
    """
    Aggiorna SOLO le colonne embedding nel file processed_data.zstd.parquet.
    Tutti gli altri dati normalizzati restano invariati.
    """
    file_path = os.path.join(DATA_PATH, "processed_data.zstd.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError("processed_data.zstd.parquet non trovato")
    try:
        df = pl.read_parquet(file_path)
        mask = (df["symbol"] == symbol) & (df["timeframe"] == timeframe)
        if mask.sum() == 0:
            logging.warning(f"⚠️ Nessun dato trovato per {symbol} {timeframe} nel file processed_data.")
            return

        # Prepara le colonne embedding
        embedding_cols = [f"embedding_{i}" for i in range(len(embedding))]
        for i, col in enumerate(embedding_cols):
            # Se la colonna non esiste, aggiungila con valori nulli
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
            # Aggiorna solo le righe selezionate
            df = df.with_columns(
                pl.when(mask).then(embedding[i]).otherwise(pl.col(col)).alias(col)
            )

        df.write_parquet(file_path, compression="zstd")
        logging.info(f"✅ Embedding aggiornato SOLO per {symbol} [{timeframe}]")
    except Exception as e:
        logging.error(f"❌ Errore durante aggiornamento embedding: {e}")
