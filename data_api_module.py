"""
data_api_module.py
Modulo avanzato per il download e la gestione dei dati di mercato.
"""

import asyncio
import logging
import os
import stat
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import polars as pl
import yfinance as yf
import MetaTrader5 as mt5
from polars.exceptions import ComputeError
from data_loader import (
    load_auto_symbol_mapping,
    standardize_symbol,
    load_preset_assets,
)
from column_definitions import required_columns, apply_all_column_features

print("data_api_module.py caricato ✅")

# Configurazione globale
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(SCRIPT_DIR, "market_data.zstd.parquet")
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.zstd.parquet"
ENABLE_CLOUD_SYNC = False
USE_MARKET_APIS = False
ENABLE_YFINANCE = False   # Attiva/disattiva yfinance
ENABLE_MT5 = True   # Attiva/disattiva MT5
CHUNK_SIZE = 30000        # Numero massimo di righe per ogni chunk
start_date = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")
DAYS_HISTORY = 60
executor = ThreadPoolExecutor(max_workers=8)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Utility
def ensure_permissions(file_path):
    """Garantisce che il file abbia i permessi di lettura e scrittura."""
    try:
        created = False
        if not os.path.exists(file_path):
            with open(file_path, 'w'):
                pass
            created = True
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        if created:
            logging.info("✅ File creato e permessi impostati: %s", file_path)
        else:
            logging.debug("Permessi verificati per il file: %s", file_path)
    except Exception as e:
        logging.error(
            "❌ Errore nell'impostare i permessi per %s: %s", file_path, e
        )


def ensure_all_columns(df):
    """Garantisce che il DataFrame contenga tutte le colonne richieste."""
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    for col in missing_columns:
        df = df.with_columns(pl.lit(0).alias(col))
    return df.fill_nan(0).fill_null(0)

def delay_request(seconds=2):
    """Introduce un ritardo tra le richieste."""
    logging.info(
        "⏳ Attendo %d secondi prima della prossima richiesta...", seconds
    )
    time.sleep(seconds)


def retry_request(func, *args, retries=3, delay=2, **kwargs):
    """
    Riprova una funzione in caso di errore,
    con un ritardo esponenziale tra i tentativi.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"⚠️ Errore tentativo {attempt + 1}: {e}")
            time.sleep(delay * (2 ** attempt))
    return None


def download_data_with_yfinance(symbols, start_date=start_date):
    """Scarica dati storici da Yahoo Finance usando batch di ticker."""
    if start_date is None:
        start_date = (
            datetime.today() - timedelta(days=120)
        ).strftime("%Y-%m-%d")
    data = []
    try:
        yf_symbols = [standardize_symbol(
            s, load_auto_symbol_mapping(),
            provider="yfinance") for s in symbols]
        combined_data = None

        for attempt in range(3):
            try:
                combined_data = yf.download(
                    yf_symbols,
                    start=start_date,
                    end=datetime.today().strftime("%Y-%m-%d"),
                    progress=False,
                    group_by=None
                )
                if combined_data.empty or combined_data.isna().all().all():
                    logging.warning(
                        "⚠️ Nessun dato valido ricevuto dal batch di yfinance."
                    )
                    return []

                combined_data = combined_data.reset_index()
                if isinstance(combined_data.columns, pd.MultiIndex):
                    combined_data.columns = [
                        "_".join(
                            [str(i) for i in col if i]
                        ) for col in combined_data.columns.values
                    ]
                else:
                    combined_data.columns = [
                        str(col) for col in combined_data.columns
                    ]
                combined_data["symbol"] = (
                    yf_symbols[0] if len(yf_symbols) == 1 else "MULTIPLE"
                )
                data.extend(combined_data.to_dict(orient="records"))
                logging.info("✅ Dati batch scaricati con successo.")
                break
            except Exception as e:
                if "Rate limited" in str(e):
                    logging.warning(
                        "⚠️ Rate limit raggiunto. Attendo 60sec prima di riprovare"
                    )
                    time.sleep(60)
                else:
                    logging.error(f"❌ Errore durante il download batch: {e}")
                    return []
    except Exception as e:
        logging.error(f"❌ Errore durante il download batch: {e}")
    return data


def download_data_with_mt5(symbols, days=60, timeframes=None):
    """
    Scarica dati storici da MT5 per gli ultimi N giorni e per più timeframe.
    """
    if not mt5.initialize():
        logging.error("❌ Impossibile inizializzare MT5: %s", mt5.last_error())
        return []

    if timeframes is None:
        timeframes = [mt5.TIMEFRAME_D1]  # Default solo daily
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "h1": mt5.TIMEFRAME_H1,
        "h4": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1
    }

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    data = []

    for tf_name in timeframes:
        tf = tf_map[tf_name]
        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                logging.warning(f"❌ Impossibile attivare il simbolo in MT5: {symbol}")
                continue
            rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
            if rates is None or len(rates) == 0:
                logging.warning(
                    f"⚠️ Nessun dato per {symbol} timeframe {tf_name} da MT5."
                )
                continue
            df = pd.DataFrame(rates)
            if df.empty:
                continue
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            df["timeframe"] = tf_name
            df.rename(columns={
                "time": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume"
            }, inplace=True)
            data.extend(df.to_dict(orient="records"))
            time.sleep(0.1)
    mt5.shutdown()
    return data


def save_and_sync(data):
    """Salva i dati grezzi e sincronizza se richiesto."""
    if not data:
        logging.warning("⚠️ Nessun dato da salvare. Skip del salvataggio.")
        return

    ensure_permissions(STORAGE_PATH)
    try:
        df_new = pl.DataFrame(data)
        df_new = ensure_all_columns(df_new)
        df_new = apply_all_column_features(df_new)
        if df_new.is_empty():
            logging.warning(
                "⚠️ DataFrame vuoto dopo la pulizia. Skip del salvataggio."
            )
            return

        if os.path.exists(STORAGE_PATH) and os.path.getsize(STORAGE_PATH) > 0:
            existing_df = pl.read_parquet(STORAGE_PATH)
            # Allinea le colonne di existing_df e df_new
            for col in required_columns:
                if col not in existing_df.columns:
                    existing_df = existing_df.with_columns([pl.lit(None).alias(col)])
                if col not in df_new.columns:
                    df_new = df_new.with_columns([pl.lit(None).alias(col)])
            # Ordina le colonne nello stesso ordine
            existing_df = existing_df.select(required_columns)
            df_new = df_new.select(required_columns)
            df_final = pl.concat(
                [existing_df, df_new]
            ).unique(subset=["symbol", "timeframe", "timestamp"])
        else:
            df_final = df_new
            df_final = df_final.select(required_columns)

        df_final = df_final.sort(["symbol", "timeframe", "timestamp"])
        df_final = df_final.fill_null(0)
        df_final.write_parquet(STORAGE_PATH, compression="zstd")
        logging.info("✅ Dati grezzi salvati: %s", STORAGE_PATH)

        if ENABLE_CLOUD_SYNC:
            sync_to_cloud()
    except (OSError, IOError, ComputeError) as e:
        logging.error("❌ Errore salvataggio dati: %s", e)


def sync_to_cloud():
    """Sincronizzazione avanzata intelligente con il cloud."""
    try:
        if os.path.exists(STORAGE_PATH):
            os.replace(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Sincronizzazione intelligente cloud completata.")
    except OSError as e:
        logging.error("❌ Errore sincronizzazione cloud: %s", e)


async def main():
    """Download e salvataggio dei dati di mercato con fallback automatico."""
    try:
        assets = load_preset_assets()
        mapping = load_auto_symbol_mapping()

        enabled_assets = []
        symbol_category_map = {}

        for provider, config in assets.items():
            if isinstance(config, dict) and config.get("enabled", False):
                for symbol in config.get("assets", []):
                    enabled_assets.append(symbol)
                    symbol_category_map[symbol] = provider
        if not enabled_assets:
            logging.error("❌ Nessun asset abilitato trovato in preset_assets.json.")
            return
        
        yf_symbols = [
            standardize_symbol(symbol, mapping, provider="yfinance")
            for symbol in enabled_assets
        ]
        data_all = []

        if ENABLE_YFINANCE:
            data_yf = []
            for i in range(0, len(yf_symbols), 50):
                batch = yf_symbols[i:i + 50]
                data_batch = retry_request(
                    download_data_with_yfinance, batch,
                    start_date=start_date
                )
                if data_batch:
                    for row in data_batch:
                        row["category"] = symbol_category_map.get(row["symbol"], "unknown")
                    data_yf.extend(data_batch)
                delay_request(10)
            data_all.extend(data_yf)

        if ENABLE_MT5:
            mt5_symbols = enabled_assets

            data_mt5 = download_data_with_mt5(
                mt5_symbols, days=DAYS_HISTORY,
                timeframes=["1m", "5m", "15m", "30m", "h1", "h4", "1d"]
            )
            if not data_mt5:
                logging.warning("⚠️ Nessun dato ricevuto da MT5 per nessun timeframe.")
            for row in data_mt5:    
                row["category"] = symbol_category_map.get(row["symbol"], "unknown")
            data_all.extend(data_mt5)

        if len(data_all) > CHUNK_SIZE:
            for chunk in range(0, len(data_all), CHUNK_SIZE):
                partial_data = data_all[chunk:chunk + CHUNK_SIZE]
                save_and_sync(partial_data)
        else:
            save_and_sync(data_all)
        try:
            df_test = pl.read_parquet(STORAGE_PATH)
            if "timeframe" not in df_test.columns:
                    logging.warning("⚠️ Colonna 'timeframe' mancante nel file finale. Controlla se i dati sono validi.")
        except Exception as e:
            logging.warning(f"⚠️ Impossibile verificare la colonna 'timeframe': {e}")

        logging.info(
            "🎉 Processo completato con successo! File generato: %s",
            STORAGE_PATH
        )
    except Exception as e:
        logging.error("❌ Errore durante l'esecuzione: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
