"""
market_fingerprint.py
Questo modulo gestisce la compressione dei dati di
mercato in vettori numerici
rappresentativi e l'aggiornamento/ripristino degli embedding
per simboli e timeframe specifici.
Funzionalità principali:
- compress_to_vector: Comprimi un DataFrame in un vettore numerico.
- update_embedding_in_processed_file:
Aggiorna il file processato con un embedding.
- get_embedding_for_symbol:
Recupera l'embedding per un dato simbolo e timeframe.
"""
from pathlib import Path
import hashlib
import polars as pl
import numpy as np
from datetime import datetime, timezone
import time
import gc
import logging
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from ai_features import get_features_by_strategy_type

print("market_fingerprint.py caricato ✅")

DATA_DIR = Path("D:/trading_data")
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.zstd.parquet"
EMBEDDING_FILE = DATA_DIR / "embedding_data.zstd.parquet"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_write_parquet(df: pl.DataFrame, path: Path, retries: int = 3, wait: float = 1.0):
    tmp_path = path.with_suffix(".temp")
    for attempt in range(retries):
        try:
            df.write_parquet(tmp_path, compression="zstd")
            tmp_path.replace(path)  # Scrivi prima il file temporaneo, poi lo rinomini
            return True
        except OSError as e:
            logging.warning(f"⚠️ Tentativo {attempt+1} fallito per {path}: {e}")
            time.sleep(wait)
            gc.collect()
    logging.error(f"❌ Scrittura fallita su {path} dopo {retries} tentativi.")
    return False


def get_strategy_type_from_timeframe(timeframe: str) -> str:
    """Restituisce la strategia AI in base al timeframe."""
    tf = str(timeframe).lower()
    if tf in ["1m", "5m", "15m", "30m"]:
        return "scalping"
    elif tf in ["1h", "4h"]:
        return "swing"
    else:
        return "macro"

def compress_to_vector(df: pl.DataFrame, length: int = 32, timeframe: str = "1m") -> np.ndarray:
    """
    Comprimi un DataFrame di candele in un vettore numerico rappresentativo.
    Usa PCA se possibile, altrimenti fingerprint statistico.
    """
    strategy_type = get_strategy_type_from_timeframe(timeframe)
    features = get_features_by_strategy_type(strategy_type)
    if not isinstance(features, list):
        features = [features]
    valid_cols = [col for col in features if col in df.columns]
    filtered_df = df.select(valid_cols) if valid_cols else df
    numeric_cols = filtered_df.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy()

    # Se hai abbastanza dati, usa PCA
    if numeric_cols.shape[0] >= length and numeric_cols.shape[1] > 0:
        try:
            pca = PCA(n_components=length)
            compressed = pca.fit_transform(numeric_cols)
            vector = compressed[-1]  # Ultima riga (più recente)
        except Exception:
            vector = np.zeros(length)
    else:
        # Fingerprint statistico: mean, std, min, max, skew, kurtosis per ogni colonna
        stats = []
        if numeric_cols.size > 0:
            stats.extend(np.mean(numeric_cols, axis=0))
            stats.extend(np.std(numeric_cols, axis=0))
            stats.extend(np.min(numeric_cols, axis=0))
            stats.extend(np.max(numeric_cols, axis=0))
            stats.extend(skew(numeric_cols, axis=0, nan_policy="omit"))
            stats.extend(kurtosis(numeric_cols, axis=0, nan_policy="omit"))
        stats = np.nan_to_num(np.array(stats), nan=0.0, posinf=0.0, neginf=0.0)
        # Padding/troncamento per arrivare a 'length'
        if stats.size < length:
            stats = np.pad(stats, (0, length - stats.size), 'constant')
        else:
            stats = stats[:length]
        vector = stats

    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def add_embedding_columns(df: pl.DataFrame, symbol: str, timeframe: str = "1m", length: int = 32) -> pl.DataFrame:
    """
    Aggiunge le colonne embedding_* al DataFrame usando il vettore salvato.
    """

    embedding = get_embedding_for_symbol(symbol, timeframe, length=length)
    if embedding is None or len(embedding) != length:
        embedding = np.zeros(length, dtype=np.float32)

    for i in range(length):
        col_name = f"embedding_{i}"
        df = df.with_columns(pl.Series(col_name, [float(embedding[i])] * df.height))

    return df


def update_embedding_in_processed_file(
    symbol: str, new_df: pl.DataFrame, timeframe: str = "1m", length: int = 32, keep_last_n: int = 2
):
    wait_time = 0
    while (not PROCESSED_DATA_PATH.exists() or PROCESSED_DATA_PATH.stat().st_size == 0) and wait_time < 2:
        time.sleep(0.1)
        wait_time += 0.1

    if not PROCESSED_DATA_PATH.exists():
        print("❌ File processed_data.zstd.parquet non trovato.")
        return

    try:
        df = pl.read_parquet(PROCESSED_DATA_PATH)
        for col in [f"embedding_{i}" for i in range(length)]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        if isinstance(new_df, str):
            raise TypeError("❌ ERRORE: new_df è una stringa, non un DataFrame. Qualcosa ha passato un parametro errato.")
        if new_df.is_empty() or new_df.height < 50:
            print(f"❌ Impossibile aggiornare embedding: df vuoto o troppo corto per {symbol}")
            return
        compressed_vector = compress_to_vector(new_df, length=length, timeframe=timeframe)

        # Estrai segnali
        latest_row_data = {}
        for k in new_df.columns:
            try:
                val = new_df[k][-1]
                if isinstance(val, (pl.DataFrame, pl.Series)):
                    val = val.item() if hasattr(val, "item") else None
                latest_row_data[k] = val
            except Exception:
                latest_row_data[k] = None

        latest_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now(timezone.utc).timestamp()
        }
        for i, val in enumerate(compressed_vector):
            latest_dict[f"embedding_{i}"] = float(val)

        columns_to_add = []
        for col in new_df.columns:
            if not isinstance(col, str):
                try:
                    col = str(col)
                except Exception:
                    continue
            col_lower = col.lower()
            try:
               if (
                    not col_lower.startswith("embedding")
                    and col_lower not in ["symbol", "timeframe", "timestamp"]
                    and new_df[col].dtype != pl.Null
                    and new_df[col].max() != new_df[col].min()
               ):
                columns_to_add.append(col)
            except Exception:
                continue

        for col in columns_to_add:
            latest_dict[col] = latest_row_data.get(col, None)

        for col in df.columns:
            if col not in latest_dict:
                latest_dict[col] = None

        # Allinea colonne tra df (storico) e latest_row (nuova riga)
        missing_cols = [col for col in latest_dict.keys() if col not in df.columns]
        for col in missing_cols:
            df = df.with_columns(pl.lit(None).alias(col))

        missing_cols_existing = [col for col in df.columns if col not in latest_dict]
        for col in missing_cols_existing:
            latest_dict[col] = None

        latest_row = pl.DataFrame([latest_dict])
        # Ordina le colonne per coerenza
        df = df.select(sorted(df.columns))
        latest_row = latest_row.select(sorted(latest_row.columns))


        updated_df = pl.concat([df, latest_row])

        embedding_cols = [f"embedding_{i}" for i in range(length)]
        if embedding_cols[0] in updated_df.columns:
            df_embeddings = updated_df.filter(
                (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe) &
                (pl.col(embedding_cols[0]).is_not_null())
            ).sort("timestamp", descending=True)
        else:
            df_embeddings = pl.DataFrame(schema=updated_df.schema)

        df_non_embed = updated_df.filter(
            ~((pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe) &
              (pl.col(embedding_cols[0]).is_not_null()))
        )

        final_df = pl.concat([df_non_embed, df_embeddings.head(keep_last_n)])
        from embedding_sync import write_temp_embedding
        write_temp_embedding(final_df.tail(1))

        emb_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
        }
        for i, val in enumerate(compressed_vector):
            emb_dict[f"embedding_{i}"] = float(val)
        emb_df = pl.DataFrame([emb_dict])

        if EMBEDDING_FILE.exists():
            old_emb = pl.read_parquet(EMBEDDING_FILE)
            old_emb = old_emb.filter(~((pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)))
            emb_df = pl.concat([old_emb, emb_df])

        safe_write_parquet(emb_df, EMBEDDING_FILE)


        print(f"✅ Embedding aggiornato per {symbol} [{timeframe}]")
        print(f"📦 File: {PROCESSED_DATA_PATH.name} | Dim: {PROCESSED_DATA_PATH.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        print(f"❌ Errore durante aggiornamento embedding: {e}")


def get_embedding_for_symbol(
    symbol: str, timeframe: str = "1m", length: int = 32
) -> np.ndarray:
    """
    Recupera l'embedding vettoriale salvato per un dato simbolo e timeframe.
    Restituisce un array numpy normalizzato (32 valori).
    """
    try:
        if not EMBEDDING_FILE.exists():
            return np.zeros(length, dtype=np.float32)

        df = pl.read_parquet(EMBEDDING_FILE)
        for col in [f"embedding_{i}" for i in range(length)]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        row = df.filter(
            (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)
        )

        if row.is_empty():
            return np.zeros(length, dtype=np.float32)

        embedding_cols = [f"embedding_{i}" for i in range(length)]
        vector = row.select(embedding_cols).to_numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"❌ Errore durante il recupero dell'embedding: {e}")
        return np.zeros(length, dtype=np.float32)

    
def load_latest_embedding_and_features(symbol: str, timeframe: str = "1m", length: int = 32) -> np.ndarray:
    """
    Carica embedding e segnali chiave da processed_data.zstd.parquet per l'osservazione AI.
    """
    if not PROCESSED_DATA_PATH.exists():
        return np.zeros(length, dtype=np.float32)

    df = pl.read_parquet(PROCESSED_DATA_PATH)
    rows = df.filter((pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe))
    if rows.is_empty():
        return np.zeros(length, dtype=np.float32)

    rows = rows.sort("timestamp", descending=True)
    for row in rows.iter_rows(named=True):
        if all(row.get(f"embedding_{i}") is not None for i in range(length)):
            embed = np.array([row.get(f"embedding_{i}", 0.0) for i in range(length)], dtype=np.float32)
            extra = []
            for col in ["RSI", "MACD", "MACD_Signal", "signal_score", "weighted_signal_score", "ILQ_Zone"]:
                val = row.get(col)
                extra.append(float(val) if val is not None else 0.0)
            return np.concatenate([embed, np.array(extra, dtype=np.float32)])
    return np.zeros(length, dtype=np.float32)


def validate_processed_data(required_columns: list[str] = None):
    """Controlla la consistenza del file processed_data."""
    if required_columns is None:
        ["symbol", "timeframe"] + [f"embedding_{i}" for i in range(32)] + ["RSI", "ILQ_Zone", "signal_score"]

    if not PROCESSED_DATA_PATH.exists():
        print("❌ File processed_data.zstd.parquet mancante.")
        return

    df = pl.read_parquet(PROCESSED_DATA_PATH)

    # Verifica colonne mancanti
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️ Colonna mancante: {col}")
        else:
            null_count = df[col].null_count()
            if null_count > 0:
                print(f"⚠️ Colonna {col} contiene {null_count} valori nulli")

    print(f"✅ Validazione completata - Righe totali: {df.shape[0]}")
