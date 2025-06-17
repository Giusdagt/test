from pathlib import Path
import polars as pl
import logging
import time
import gc

TEMP_EMBEDDING_FILE = Path("D:/trading_data/temp_embedding_data.zstd.parquet")
PROCESSED_DATA_PATH = Path("D:/trading_data/processed_data.zstd.parquet")

def write_temp_embedding(latest_row: pl.DataFrame):
    """Scrive una riga embedding temporanea in un file ponte, allineando le colonne."""
    try:
        if TEMP_EMBEDDING_FILE.exists():
            existing = pl.read_parquet(TEMP_EMBEDDING_FILE)

            # Allinea colonne mancanti
            missing_cols = [col for col in latest_row.columns if col not in existing.columns]
            for col in missing_cols:
                existing = existing.with_columns(pl.lit(None).alias(col))

            missing_cols_existing = [col for col in existing.columns if col not in latest_row.columns]
            for col in missing_cols_existing:
                latest_row = latest_row.with_columns(pl.lit(None).alias(col))

            # Riordina le colonne
            existing = existing.select(sorted(existing.columns))
            latest_row = latest_row.select(sorted(latest_row.columns))

            # ❗ Casta latest_row allo schema di existing (tollerante)
            latest_row = latest_row.cast(existing.schema, strict=False)

            combined = pl.concat([existing, latest_row])
        else:
            combined = latest_row

        combined.write_parquet(TEMP_EMBEDDING_FILE, compression="zstd")
        logging.info(f"📩 Embedding temporaneo scritto in {TEMP_EMBEDDING_FILE.name}")
    except Exception as e:
        logging.warning(f"⚠️ Errore scrittura temp_embedding: {e}")

def sync_temp_embeddings_to_main():
    """Fonde le righe embedding temporanee nel file processed_data.zstd.parquet"""
    if not TEMP_EMBEDDING_FILE.exists() or TEMP_EMBEDDING_FILE.stat().st_size == 0:
        return

    if not PROCESSED_DATA_PATH.exists():
        logging.warning("⚠️ File principale non trovato per la sincronizzazione embedding.")
        return

    try:
        temp_df = pl.read_parquet(TEMP_EMBEDDING_FILE)
        main_df = pl.read_parquet(PROCESSED_DATA_PATH)

        for col in temp_df.columns:
            if col not in main_df.columns:
                main_df = main_df.with_columns(pl.lit(None).alias(col))
        for col in main_df.columns:
            if col not in temp_df.columns:
                temp_df = temp_df.with_columns(pl.lit(None).alias(col))
        # Allinea schema tra main_df e temp_df
        for col in temp_df.columns:
            if col in main_df.columns:
                try:
                    temp_dtype = temp_df[col].dtype
                    main_dtype = main_df[col].dtype
                    if temp_dtype != main_dtype:
                        # Forza cast del temp al tipo del file principale
                        temp_df = temp_df.with_columns([
                            pl.col(col).cast(main_dtype)
                        ])
                except Exception as e:
                    logging.warning(f"⚠️ Cast forzato fallito per colonna {col}: {e}")

        temp_df = temp_df.select(main_df.columns)
        merged = pl.concat([main_df, temp_df]).unique(subset=["symbol", "timeframe", "timestamp"])
        merged = merged.sort("timestamp")

        merged.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
        TEMP_EMBEDDING_FILE.unlink(missing_ok=True)
        logging.info("✅ Embedding temporanei sincronizzati con processed_data.")
    except Exception as e:
        logging.error(f"❌ Errore durante sync embedding: {e}")
        gc.collect()
