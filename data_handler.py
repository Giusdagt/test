"""
data_handler.py
Modulo definitivo per la gestione autonoma, intelligente e ottimizzata
per la normalizzazione e gestione avanzata dei dati storici e realtime.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e scalping
con MetaTrader5.
"""

import os
import sys
import logging
import hashlib
import shutil
import asyncio
import threading
from pathlib import Path
from datetime import datetime
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import polars as pl
import polars.selectors as cs
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from column_definitions import required_columns, apply_all_column_features, add_realtime_orderbook_columns
from indicators import (
    get_indicators_list,
    calculate_intraday_indicators
)
from data_loader import (
    load_auto_symbol_mapping,
    standardize_symbol,
    USE_PRESET_ASSETS,
    ENABLE_SYMBOL_STANDARDIZATION,
    load_preset_assets
)
from data_api_module import (
    ensure_permissions,
    main as fetch_new_data
)
from delivery_zone_utils import add_delivery_zone_columns
from smart_features import apply_all_advanced_features, detect_strategy_type, apply_all_market_structure_signals
from ai_features import get_features_by_strategy_type
from market_fingerprint import add_embedding_columns, update_embedding_in_processed_file
from realtime_update import update_realtime_embedding #update_embedding_in_processed_file


ENABLE_SYMBOL_STANDARDIZATION = False  # O True se vuoi attivarla
ENABLE_CLOUD_SYNC = False
update_lock = threading.Lock()

print("data_handler.py caricato ✅")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Disabilita l'inizializzazione MetaTrader5 durante i test o in modalità mock
if "pytest" not in sys.modules and "MOCK_MT5" not in os.environ:
    if not mt5.initialize():
        logging.error("❌ Errore inizializzazione MT5: %s", mt5.last_error())
        sys.exit()
else:
    logging.info("⚠️ Inizializzazione MT5 disabilitata (modalità test/mock).")

RAW_DATA_PATH = "market_data.zstd.parquet"
SAVE_DIRECTORY = Path("D:/trading_data")
PROCESSED_DATA_PATH = SAVE_DIRECTORY / "processed_data.zstd.parquet"
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/processed_data.zstd.parquet"
RAM_WINDOW = 1000   # quante righe usare nei calcoli in RAM
SAVE_WINDOW = 9000  # quante righe salvare su disco
MAX_INITIAL = 9000  # Solo al primo avvio
MAX_UPDATE = 1000    # Ogni aggiornamento successivo

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(CLOUD_SYNC_PATH), exist_ok=True)

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)

def deep_fill_null(df, value=0):
    for col in df.columns:
        dtype = df.schema[col]
        if dtype == pl.Struct or dtype == pl.List:
            df = df.with_columns(
                pl.col(col).apply(lambda x: value if x is None else x).alias(col)
            )
    df = df.fill_null(value)
    return df

def ensure_file_and_permissions(filepath):
    """Crea il file se non esiste e garantisce i permessi di scrittura."""
    if not os.path.exists(filepath):
        logging.warning(f"⚠️ File non trovato, lo creo: {filepath}")
        with open(filepath, 'w'):
            pass
    ensure_permissions(filepath)

def get_realtime_symbols():
    """
    Restituisce la lista dei simboli da usare per il realtime,
    scegliendo tra preset_assets e mappatura automatica in base a USE_PRESET_ASSETS.
    """
    if USE_PRESET_ASSETS:
        assets = load_preset_assets()
        return [asset for info in assets.values() for asset in info["assets"]]
    else:
        auto_mapping = load_auto_symbol_mapping()
        return list(auto_mapping.values())


def file_hash(filepath):
    """Calcola l'hash del file per rilevare modifiche."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def sync_to_cloud():
    """Sincronizzazione con Google Drive solo se necessario."""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            return
        existing_hash = file_hash(CLOUD_SYNC_PATH)
        new_hash = file_hash(PROCESSED_DATA_PATH)
        if existing_hash == new_hash:
            logging.info("☁️ Nessuna modifica, skip sincronizzazione.")
            return
        shutil.copy2(PROCESSED_DATA_PATH, CLOUD_SYNC_PATH)
        logging.info("☁️ Sincronizzazione cloud completata.")
    except (OSError, IOError) as e:
        logging.error("❌ Errore sincronizzazione cloud: %s", e)


def save_and_sync(df, symbol, timeframe=None, embedding=None):
    """Salvataggio intelligente con verifica delle modifiche e merge dei dati."""
    try:
        if df.is_empty():
            logging.warning("⚠️ Tentativo salvataggio, un DataFrame vuoto.")
            return
        df = calculate_intraday_indicators(df)
        df = apply_all_advanced_features(df)
        df = apply_all_market_structure_signals(df)
        df = add_delivery_zone_columns(df, symbol)
        df = ensure_all_columns(df)
        df = normalize_data(df, symbol, embedding)
        # df.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
        ensure_file_and_permissions(PROCESSED_DATA_PATH)

        if symbol and timeframe:
            try:
                update_embedding_in_processed_file(symbol=symbol, new_df=df, timeframe=timeframe)
                logging.info(f"✅ Embedding scritto su processed_data.zstd.parquet per {symbol} [{timeframe}]")
            except Exception as e:
                logging.error(f"❌ Errore durante update_embedding_in_processed_file: {e}")

        if os.path.exists(PROCESSED_DATA_PATH) and os.path.getsize(PROCESSED_DATA_PATH) > 0:
            existing_df = pl.read_parquet(PROCESSED_DATA_PATH)
            existing_df = ensure_all_columns(existing_df)
            df = ensure_all_columns(df)
            df = df.group_by(["symbol", "timeframe"]).tail(RAM_WINDOW)
            # Ordina le colonne nello stesso ordine
            all_cols = required_columns + [col for col in df.columns if col not in required_columns]
            existing_df = existing_df.select([col for col in all_cols if col in existing_df.columns])
            df = df.select([col for col in all_cols if col in df.columns])
            # Concatena e rimuovi duplicati su symbol, timeframe, timestamp
            existing_df, df = align_column_types(existing_df, df)
            common_cols = sorted(set(existing_df.columns) & set(df.columns))
            existing_df = existing_df.select(common_cols)
            df = df.select(common_cols)
            df_merged = pl.concat([existing_df, df], how="vertical")
            df_merged = df_merged.sort(["symbol", "timeframe", "timestamp"], descending=True)
            df_final = df_merged.unique(subset=["symbol", "timeframe", "timestamp"], keep="first")
        else:
            df_final = df # df_final = df.select(required_columns)

        # Verifica che le colonne chiave esistano PRIMA di ordinare
        for col in ["symbol", "timeframe", "timestamp"]:
            if col not in df_final.columns:
                df_final = df_final.with_columns(pl.lit(None).alias(col))
        
        df_final = df_final.sort(["symbol", "timeframe", "timestamp"], descending=True)
        df_final = df_final.group_by(["symbol", "timeframe"]).head(SAVE_WINDOW)
        df_final = df_final.sort(["symbol", "timeframe", "timestamp"])
        if embedding is not None:
            df = df.with_columns([
                pl.Series(f"embedding_{i}", [float(val)] * df.height) for i, val in enumerate(embedding)
            ])

        df_final.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
        logging.info(f"✅ Dati salvati in: {PROCESSED_DATA_PATH} (max {SAVE_WINDOW} righe per symbol/timeframe)")

        if ENABLE_CLOUD_SYNC:
            executor.submit(sync_to_cloud)
    except (OSError, IOError, ValueError, pl.exceptions.ColumnNotFoundError) as e:
        logging.error("❌ Errore durante il salvataggio dati: %s", e)

    """try:
        timestamps = df.select("timestamp").to_series().to_list()
        with open("update_log.txt", "a") as log_file:
            log_file.write(f"{df[0,'symbol']} {df[0,'timeframe']} -> {timestamps[-1]}\n")
    except Exception as e:
        logging.warning(f"⚠️ Log aggiornamento fallito: {e}")"""

def ensure_all_columns(df):
    """Garantisce che il DataFrame contenga tutte le colonne richieste."""
    for col in required_columns:
        if col not in df.columns:
            if col == "symbol":
                df = df.with_columns(pl.lit("UNKNOWN").alias("symbol"))
            elif col == "timeframe":
                df = df.with_columns(pl.lit("HIST").alias("timeframe"))
            else:
                df = df.with_columns(pl.lit(None).alias(col))
    return df

def process_day(symbol: str, timeframe: str = "1m", length: int = 32):
    """
    Elabora la giornata corrente per un simbolo e timeframe.
    - Carica le ultime 1000 candele da processed_data.
    - Aggiunge le nuove da MT5 (realtime).
    - Calcola segnali, embedding.
    - Salva tutto in modo coerente e leggero.
    """

    if not os.path.exists(PROCESSED_DATA_PATH):
        print("❌ File processed_data non trovato.")
        return

    # 1. Carica le ultime 1000 righe per quel simbolo+timeframe
    df_hist = pl.read_parquet(PROCESSED_DATA_PATH)
    df_hist = df_hist.filter((pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe))
    df_hist = df_hist.sort("timestamp", descending=True).head(1000).sort("timestamp")

    if df_hist.is_empty():
        print("❌ Nessun dato storico trovato.")
        return

    # 2. Ottieni i nuovi dati realtime
    df_rt = fetch_mt5_data(symbol, timeframe)
    if df_rt is None or df_rt.is_empty():
        print("❌ Nessun dato realtime disponibile.")
        return

    # 3. Unisci e rimuovi duplicati
    df_combined = pl.concat([df_hist, df_rt]).unique(subset=["timestamp"]).sort("timestamp")

    # 4. Calcola segnali
    df_combined = calculate_intraday_indicators(df_combined)
    df_combined = apply_all_column_features(df_combined)
    df_combined = apply_all_advanced_features(df_combined)
    df_combined = apply_all_market_structure_signals(df_combined)

    # 5. Salva gli ultimi 1000 nel file principale
    df_final = df_combined.sort("timestamp", descending=True).head(3000).sort("timestamp")

    # 6. Embedding e AI update
    update_embedding_in_processed_file(symbol=symbol, new_df=df_combined, timeframe=timeframe, length=length)
    update_realtime_embedding(symbol, df_combined, timeframe=timeframe, length=length)

    save_and_sync(df_final, symbol=symbol, timeframe=timeframe)

    print(f"✅ Giornata aggiornata per {symbol} [{timeframe}]")


def align_column_types(df1, df2):
    """Allinea i tipi di colonna tra due DataFrame Polars senza perdere colonne."""
    all_cols = set(df1.columns).union(set(df2.columns))
    for col in all_cols:
        if col in df1.columns and col in df2.columns:
            dtype1 = str(df1.schema[col])
            dtype2 = str(df2.schema[col])
            if "float" in dtype1 or "float" in dtype2:
                df1 = df1.with_columns(pl.col(col).cast(pl.Float64))
                df2 = df2.with_columns(pl.col(col).cast(pl.Float64))
            elif dtype1 != dtype2:
                df1 = df1.with_columns(pl.col(col).cast(pl.Utf8))
                df2 = df2.with_columns(pl.col(col).cast(pl.Utf8))
        elif col in df1.columns:
            df2 = df2.with_columns(pl.lit(None).cast(df1.schema[col]).alias(col))
        elif col in df2.columns:
            df1 = df1.with_columns(pl.lit(None).cast(df2.schema[col]).alias(col))
    return df1, df2

def normalize_data(df, symbol, embedding=None):
    """Normalizzazione avanzata con selezione dinamica delle feature per IA."""
    if df.is_empty():
        return df
    df = calculate_intraday_indicators(df)
    df = apply_all_column_features(df)
    df = apply_all_advanced_features(df)
    df = apply_all_market_structure_signals(df)
    df = add_delivery_zone_columns(df, symbol)
    df = ensure_all_columns(df)
    for col in ["delivery_zone_buy", "delivery_zone_sell"]:
        if col not in df.columns:
            df = df.with_columns(pl.Series(col, [0.0] * df.height))
        else:
            df = df.with_columns(
                df[col].fill_null(0.0).fill_nan(0.0).alias(col)
            )
    
    for col in df.select(cs.numeric()).columns:
        df = df.with_columns(
            df[col]
            .cast(pl.Float64)
            .fill_nan(0)
            .fill_null(0)
            .map_elements(lambda x: 0.0 if not np.isfinite(x) or abs(x) > 1e12 else x, return_dtype=pl.Float64)
            .alias(col)
        )
    strategy_type = detect_strategy_type(df)
    df = df.with_columns([pl.lit(strategy_type).alias("strategy_type")])
    strategy_cols = get_features_by_strategy_type(strategy_type)
    all_numeric = df.select(cs.numeric()).columns
    numeric_cols = [col for col in strategy_cols if col in all_numeric]
    if not numeric_cols or len(numeric_cols) < 10:
        logging.warning("⚠️ Strategy type ha restituito troppe poche colonne. Uso tutte le colonne numeriche valide.")
        numeric_cols = [col for col in all_numeric if df[col].max() != df[col].min()]
        logging.debug(f"📊 Colonne numeriche usate: {numeric_cols}")

    
    CRITICAL_COLUMNS = ["delivery_zone_buy", "delivery_zone_sell", "ILQ_Zone", 
                        "engulfing_bullish", "engulfing_bearish", "fakeout_up", 
                        "fakeout_down", "volatility_squeeze"]
    valid_cols = [col for col in numeric_cols if (df[col].max() != df[col].min() or col in CRITICAL_COLUMNS)]

    if not valid_cols:
        logging.warning("⚠️ Nessuna colonna valida per la normalizzazione. Nessuna normalizzazione applicata.")
        return df
    anomalous_columns_log = []
    for col in valid_cols:
        original = df[col]
        first_invalid = original.filter(
            original.map_elements(lambda x: not np.isfinite(x) or abs(x) > 1e12, return_dtype=pl.Boolean)
        ).to_list()
        try:
            first_invalid = original.filter(
                original.map_elements(lambda x: not np.isfinite(x) or abs(x) > 1e12, return_dtype=pl.Boolean)
            ).to_list()
            if first_invalid:
                anomalous_columns_log.append(f"{col}: {first_invalid[:3]}")
        except Exception as e:
            logging.warning(f"⚠️ Errore durante controllo valori anomali per {col}: {e}")

    if anomalous_columns_log:
        with open("normalization_warnings.log", "a") as log_file:
            log_file.write(f"\n[Colonne con valori anomali] {datetime.now()}\n")
            for entry in anomalous_columns_log:
                log_file.write(f"  - {entry}\n")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select(valid_cols).to_numpy())
    for idx, col in enumerate(valid_cols):
        df = df.with_columns(pl.Series(col, scaled_data[:, idx]))

    if embedding is not None:
        df = df.with_columns([
            pl.Series(f"embedding_{i}", [val] * df.height) for i, val in enumerate(embedding)
        ])
    df = add_embedding_columns(df, symbol)
    df = df.fill_nan(0).fill_null(0)
    nulls_found = False
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            if df[col].is_nan().sum() > 0 or df[col].is_null().sum() > 0:
                print(f"⚠️ Colonna {col} contiene ancora NaN o null.")
                nulls_found = True
    if not nulls_found:
        print("✅ Tutti i valori sono stati normalizzati correttamente.")
    return df

async def process_historical_data():
    """
    Garantisce la presenza e l'aggiornamento di processed_data.zstd.parquet.
    - Se non esiste, lo crea dai dati grezzi.
    - Se esiste, aggiorna solo i simboli non aggiornati.
    """
    if not update_lock.acquire(blocking=False):
        logging.info("⏳ Aggiornamento già in corso, skip.")
        return

    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logging.warning("⚠️ processed_data.zstd.parquet non trovato. Creazione da grezzi...")
            if not os.path.exists(RAW_DATA_PATH):
                logging.warning("⚠️ File grezzi non trovato. Avvio fetch...")
                await fetch_new_data()
            else:
                logging.info("📁 File grezzi trovato: %s", RAW_DATA_PATH)
            df = pl.read_parquet(RAW_DATA_PATH)
            if df.is_empty():
                logging.error("❌ Dati grezzi vuoti. Impossibile creare processed_data.")
                return
            if "tick_volume" in df.columns and "volume" not in df.columns:
                df = df.rename({"tick_volume": "volume"})
            if "real_volume" in df.columns and "volume" not in df.columns:
                df = df.rename({"real_volume": "volume"})
            df = calculate_intraday_indicators(df)
            df = apply_all_column_features(df)
            df = apply_all_advanced_features(df)
            df = apply_all_market_structure_signals(df)
            df = ensure_all_columns(df)
            symbol = df[0, "symbol"] if "symbol" in df.columns else "UNKNOWN"
            df = normalize_data(df, symbol)
            df = deep_fill_null(df)
            save_and_sync(df, symbol=symbol)
            logging.info("✅ processed_data.zstd.parquet creato.")
            return

        # Se esiste, aggiorna solo simboli non aggiornati
        logging.info("🔄 Aggiornamento selettivo di processed_data.zstd.parquet...")
        df = pl.read_parquet(PROCESSED_DATA_PATH)
        realtime_symbols = get_realtime_symbols()
        to_update = []
        for symbol in realtime_symbols:
            for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
                df_sym = df.filter((pl.col("symbol") == symbol) & (pl.col("timeframe") == tf))
                if df_sym.is_empty() or (time.time() - df_sym["timestamp"].max()) > 180:
                    to_update.append((symbol, tf))
                    continue
                try:
                    max_ts = df_sym["timestamp"].max()
                    if isinstance(max_ts, pl.Series):
                        max_ts = max_ts.item()
                    if isinstance(max_ts, str):
                        max_ts = datetime.fromisoformat(max_ts).timestamp()
                    if isinstance(max_ts, datetime):
                        max_ts = max_ts.timestamp()
                    if not isinstance(max_ts, (int, float)):
                        raise ValueError(f"Tipo timestamp non valido: {type(max_ts)}")
                    if (time.time() - max_ts) > 180:
                        to_update.append((symbol, tf))
                except Exception as e:
                        logging.error(f"❌ Errore durante il parsing del timestamp per {symbol} [{tf}]: {e}")
                        continue
        if not to_update:
            logging.info("✅ Tutti i dati sono già aggiornati.")
            return
        for symbol, tf in to_update:
            df_new = fetch_mt5_data(symbol, tf)
            if df_new is not None and not df_new.is_empty():
                save_and_sync(df_new, symbol=symbol, timeframe=tf)
        logging.info("✅ Aggiornamento selettivo completato.")

    except Exception as e:
        logging.error(f"❌ Errore in process_historical_data: {e}")

    finally:
        update_lock.release()

def get_last_saved_timestamp(symbol, timeframe):
    """
    Recupera il timestamp più recente valido dal file processato per symbol/timeframe.
    Scarta valori nulli, infiniti o fuori range temporale (2000 - ora+1h).
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    try:
        df = pl.read_parquet(PROCESSED_DATA_PATH)
        df = df.filter(
            (pl.col("symbol") == symbol) &
            (pl.col("timeframe") == timeframe) &
            (pl.col("timestamp").is_not_null()) &
            (pl.col("timestamp") > 946684800) &   # da anno 2000
            (pl.col("timestamp") < time.time() + 3600)  # fino a 1h nel futuro
        )

        if df.is_empty():
            return None

        ts = df["timestamp"].max()

        # Conversione sicura
        if hasattr(ts, "to_python"):
            ts = ts.to_python()
        ts = float(ts)

        # Validazione finale
        if not np.isfinite(ts) or ts < 946684800 or ts > time.time() + 3600:
            logging.warning(f"❌ Timestamp ignorato (non valido): {ts}")
            return None

        return ts
    except Exception as e:
        logging.error(f"❌ Errore durante il parsing del timestamp per {symbol} [{timeframe}]: {e}")
        return None

def convert_rates_to_polars(rates, symbol, timeframe):
    df = pl.DataFrame(rates)
    if "time" in df.columns:
        df = df.rename({"time": "timestamp"})
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename({"tick_volume": "volume"})
    if "real_volume" in df.columns and "volume" not in df.columns:
        df = df.rename({"real_volume": "volume"})
    df = df.with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.lit(timeframe).alias("timeframe")
    ])
    return df

def fetch_mt5_data(symbol, timeframe="1m"):
    """
    Recupera i dati di mercato da MetaTrader5 per un simbolo e timeframe.
    Args:
    symbol (str): Il simbolo di mercato da analizzare.
    timeframe (str): Il timeframe (es. "1m", "5m", "1h").
    Returns:
    pl.DataFrame: Dati elaborati, normalizzati e arricchiti con indicatori.
    """
    try:
        if not mt5.initialize():
            raise RuntimeError("❌ MT5 non inizializzato correttamente.")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"❌ Symbol '{symbol}' non selezionabile in MT5.")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"❌ Symbol info non trovate per '{symbol}'.")
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        if timeframe not in tf_map:
            raise ValueError(f"Timeframe non valido: {timeframe}")
        tf_mt5 = tf_map[timeframe]
        print(f"📤 Chiamata MT5 → symbol={symbol} | timeframe={timeframe} ({tf_mt5})")

        last_timestamp = get_last_saved_timestamp(symbol, timeframe)
        # Controllo robusto sul timestamp
        if last_timestamp and isinstance(last_timestamp, (int, float)) and 946684800 < last_timestamp < time.time() + 3600:
            from_time = datetime.fromtimestamp(int(last_timestamp) + 1)
            rates = mt5.copy_rates_from(symbol, tf_mt5, from_time, MAX_UPDATE)
        else:
            if last_timestamp and not isinstance(last_timestamp, (int, float)):
              logging.warning(f"❌ last_timestamp non numerico: {last_timestamp}, scarico da zero.")
            elif last_timestamp and (last_timestamp < 946684800 or last_timestamp > time.time() + 3600):
                logging.warning(f"❌ last_timestamp fuori range: {last_timestamp}, scarico da zero.")
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, MAX_INITIAL)
           
        if rates is None or len(rates) == 0:
            logging.warning(f"⚠️ Nessun dato scaricato per {symbol} [{timeframe}]")
            return None

        if rates is None or len(rates) == 0:
            return None
        
        df = convert_rates_to_polars(rates, symbol, timeframe)
        df = add_realtime_orderbook_columns(df, symbol)
        df = calculate_intraday_indicators(df)
        required_indicators = get_indicators_list()
        for col in required_indicators:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))

        df = apply_all_column_features(df)
        df = apply_all_advanced_features(df)
        df = apply_all_market_structure_signals(df)
        df = ensure_all_columns(df)
        df = normalize_data(df, symbol)
        return df

    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore nel recupero dati MT5 per %s: %s", symbol, e)
        return None


def get_multi_timeframe_data(symbol, timeframes):
    """
    Restituisce un dizionario con i dati di mercato
    per ciascun timeframe specificato.
    Sceglie automaticamente se utilizzare dati normalizzati o
    recuperare dati diretti.
    """
    result = {}
    for tf in timeframes:
        try:
            # Prova a recuperare i dati normalizzati
            normalized_data = get_normalized_market_data(f"{symbol}_{tf}")
            if normalized_data is not None:
                result[tf] = normalized_data
            else:
                # Se i dati non sono disponibili, recupera i dati diretti
                result[tf] = fetch_mt5_data(symbol, timeframe=tf)
        except (KeyError, ValueError, TypeError, mt5.MetaTrader5Error) as e:
            # Log dell'errore per eventuali problemi nel recupero dei dati
            logging.error(
                "Errore nel recupero dei dati per %s con timeframe %s: %s",
                symbol, tf, e
            )
            result[tf] = None
    return result


async def get_realtime_data(symbols):
    """Ottiene dati in tempo reale da MT5 e aggiorna il database."""
    try:
        for symbol in symbols:
            if symbol is None or symbol.strip() == "":
                logging.warning("⚠️ Simbolo non valido ignorato.")
                continue
            for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
                logging.info("📡 Dati realtime %s | TF %s", symbol, tf)
                df = fetch_mt5_data(symbol, timeframe=tf)
                if df is None or df.height < 2:
                    continue
                df = ensure_all_columns(df)
                df = deep_fill_null(df)
                df = normalize_data(df, symbol)
                save_and_sync(df, symbol=symbol, timeframe=tf)
                update_embedding_in_processed_file(symbol=symbol, new_df=df, timeframe=tf)
                update_realtime_embedding(symbol, df, timeframe=tf)


            logging.info("✅ Dati realtime per %s aggiornati.", symbol)
    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore nel recupero dei dati realtime: %s", e)


def get_normalized_market_data(symbol):
    """Recupera dati normalizzati per un singolo simbolo in modo efficiente"""
    symbol = str(symbol).upper()
    if not isinstance(symbol, str):
        logging.error("❌ Simbolo non valido: atteso str, ricevuto %s", type(symbol))
        return None

    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logging.error("❌ File processato non ancora creato: %s", PROCESSED_DATA_PATH)
            return None

        df = pl.scan_parquet(PROCESSED_DATA_PATH).filter(
            pl.col("symbol").cast(pl.Utf8) == symbol
        ).collect()

        if df is None or df.is_empty():
            logging.info(f"ℹ️ Nessun dato trovato per il simbolo {symbol} (ma il file esiste).")
            return None

        return df[-1].to_dict() if df.shape[0] == 1 else df

    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore durante il recupero dei dati normalizzati per %s: %s", symbol, e)
        return None



async def main():
    try:
        await process_historical_data()
        realtime_symbols = get_realtime_symbols()
        if ENABLE_SYMBOL_STANDARDIZATION:
            auto_mapping = load_auto_symbol_mapping()
            realtime_symbols = [
                standardize_symbol(s, auto_mapping) for s in realtime_symbols
            ]
        await get_realtime_data(realtime_symbols)
    finally:
        mt5.shutdown()  # Assicura che la connessione venga chiusa alla fine


def get_available_assets():
    """
    Restituisce tutti gli asset disponibili, da preset o in modo dinamico.
    Nessuna limitazione su USD, EUR o altro.
    """
    if USE_PRESET_ASSETS:
        assets = load_preset_assets()
        return sum([block["assets"] for block in assets.values() if isinstance(block, dict) and "assets" in block], [])

    mapping = load_auto_symbol_mapping()
    return list(mapping.values())


def get_final_ai_ready_array(symbol, sequence_length=60):
    """
    Estrae i dati normalizzati in formato array/vettore per l'IA.
    Args:
        symbol (str): Simbolo dell'asset (es. 'XAUUSD')
        sequence_length (int): Numero di sequenze passate all'IA
    Returns:
        np.array: Array pronto per l'input nella rete neurale
    """
    try:
        df = get_normalized_market_data(symbol)
        if df is None or df.is_empty():
            logging.warning(f"⚠️ Nessun dato normalizzato disponibile per {symbol}.")
            return None

        # Ordina per timestamp, prendi solo le ultime righe richieste
        df = df.sort("timestamp")
        features = get_features_by_strategy_type(detect_strategy_type(df))
        data = df.select(features).to_numpy()

        if data.shape[0] < sequence_length:
            # Padding se ci sono meno righe disponibili
            padding = np.zeros((sequence_length - data.shape[0], data.shape[1]))
            data = np.vstack([padding, data])

        return data[-sequence_length:]
    except Exception as e:
        logging.error(f"❌ Errore nel creare array per IA: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
