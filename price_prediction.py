"""
Modulo per la previsione dei prezzi tramite LSTM.
- Utilizza data_handler.py per recuperare storici normalizzati con indicatori.
- Supporta più di 300 asset contemporaneamente.
- Gli asset possono essere caricati da preset_asset.json
(se attivo in `data_loader.py`) oppure selezionati dinamicamente.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import joblib
import os
import polars.selectors as cs
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data_handler import get_normalized_market_data, get_available_assets
from market_fingerprint import get_embedding_for_symbol
from smart_features import apply_all_market_structure_signals
from state_utils import safe_int, get_col_val_safe, sanitize_full_state, sanitize_columns
from constants import (
    SEQUENCE_LENGTH,
    DESIRED_STATE_SIZE,
    USE_DYNAMIC_STATE_SIZE)


print("price_prediction.py caricato ✅")

DATA_DIR = Path("D:/trading_data")
MODEL_DIR = DATA_DIR / "models"
LSTM_FOLDER = MODEL_DIR / "lstm"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LSTM_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.zstd.parquet"
TRAIN_INFO_FILE = MODEL_DIR / "train_info.parquet"
MEMORY_FILE = MODEL_DIR / "lstm_memory.parquet"
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32
ACTIVE_TRAININGS = 0
TRAIN_SEMAPHORE = asyncio.Semaphore(1)
tf.config.run_functions_eagerly(False)


def safe_int(val):
    try:
        return int(val) if not (val is None or np.isnan(val)) else 0
    except Exception:
        return 0

def load_train_info():
    if TRAIN_INFO_FILE.exists():
        return pl.read_parquet(TRAIN_INFO_FILE)
    return pl.DataFrame(
        {
            "symbol": pl.Series([], dtype=pl.Utf8),
            "data_len": pl.Series([], dtype=pl.Int64)
        }
    )

def analyze_and_select_features(df, nan_threshold=80, min_unique=2):
    """
    Analizza le feature e restituisce solo quelle utili:
    - Scarta colonne con troppi NaN o costanti.
    """
    selected = []
    stats = {}
    for col in df.columns:
        if df[col].dtype in pl.NUMERIC_DTYPES:
            values = df[col].to_numpy()
            nan_pct = np.isnan(values).mean() * 100
            n_unique = len(np.unique(values[~np.isnan(values)]))
            stats[col] = {
                "nan_pct": nan_pct,
                "n_unique": n_unique
            }
            # Criteri: meno dell'80% di NaN e almeno 2 valori unici
            if nan_pct < nan_threshold and n_unique >= min_unique:
                selected.append(col)
    # Log delle feature scartate
    for col, s in stats.items():
        if col not in selected:
            print(f"⚠️ Feature scartata: {col} (NaN%={s['nan_pct']:.1f}, unique={s['n_unique']})")
    return selected

def save_train_info(train_info):
    train_info.write_parquet(TRAIN_INFO_FILE, compression="zstd")

class PricePredictionModel:
    """
    Classe per la gestione del modello LSTM per la previsione dei prezzi.
    Include funzioni per il caricamento, l'addestramento e la previsione.
    """

    def __init__(self):
        self.memory_df = self.load_memory_table()
        # Calcola il numero di feature usando un asset di esempio
        example_assets = get_available_assets()
        if example_assets:
            example_state = self.build_full_state(example_assets[0])
            if example_state is not None:
                self.num_features = example_state.shape[1]
            else:
                self.num_features = 1
        else:
            self.num_features = 1

    @staticmethod
    def get_all_embeddings(symbol):
        """
        Restituisce la concatenazione degli embedding per tutti i timeframe.
        """
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        return np.concatenate([get_embedding_for_symbol(symbol, tf) for tf in timeframes])

    def get_model_file(self, symbol):
        """
        Restituisce il percorso del file del modello per un asset specifico.
        """
        return LSTM_FOLDER / f"lstm_model_{symbol}.keras"

    def load_memory_table(self):
        """
        Carica la tabella della memoria dal file Parquet.
        Se il file non esiste, restituisce una tabella vuota.
        """
        if MEMORY_FILE.exists():
            try:
                return pl.read_parquet(MEMORY_FILE)
            except (pl.exceptions.PolarsError, IOError, ValueError) as e:
                logging.error(
                    "❌ durante il caricamento della memoria: %s", str(e)
                )
        return pl.DataFrame(
            [[], []],
            schema={"symbol": pl.Utf8, "compressed_memory": pl.Binary}
        )

    def load_memory(self, symbol: str) -> np.ndarray:
        """
        Carica la memoria compressa per un asset specifico.
        Se non esiste, restituisce una matrice di zeri.
        """
        try:
            row = self.memory_df.filter(pl.col("symbol") == symbol)
            if row.is_empty():
                return np.zeros((SEQUENCE_LENGTH, self.num_features), dtype=np.float32)
            compressed = row[0]["compressed_memory"]
            if isinstance(compressed, pl.Series):
                compressed = compressed.item()
            return np.frombuffer(
                compressed, dtype=np.float32
            ).reshape(SEQUENCE_LENGTH, self.num_features)
        except (ValueError, KeyError, AttributeError) as e:
            logging.error(
                "❌ durante caricamento memoria per %s: %s", symbol, str(e)
            )
            return np.zeros((SEQUENCE_LENGTH, self.num_features), dtype=np.float32)

    def save_memory(self, symbol, sequence):
        """
        Salva la sequenza completa (non solo la media) per un asset specifico.
        """
        try:
            compressed = np.array(sequence, dtype=np.float32).tobytes()
            existing = self.memory_df.filter(pl.col("symbol") == symbol)
            if not existing.is_empty():
                if bytes(existing[0]["compressed_memory"]) == compressed:
                    return
                self.memory_df = self.memory_df.filter(pl.col("symbol") != symbol)

            new_row = pl.DataFrame(
                {
                    "symbol": [symbol],
                    "compressed_memory": [compressed]
                }
            )
            self.memory_df = pl.concat([self.memory_df, new_row])
            self.memory_df.write_parquet(MEMORY_FILE, compression="zstd")
        except (ValueError, IOError, pl.exceptions.PolarsError) as e:
            logging.error(
                "❌ per il salvataggio della memoria per %s: %s", symbol, str(e)
            )

    def build_lstm_model(self):
        """
        Costruisce un modello LSTM con due livelli e un livello Dense finale.
        """
        try:
            model = Sequential([
                LSTM(
                    64, activation="tanh", return_sequences=True,
                    dtype="float16"
                ),
                Dropout(0.2),
                LSTM(
                    32, activation="tanh", return_sequences=False,
                    dtype="float16"
                ),
                Dense(1, activation="linear", dtype="float16")
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model
        except (ValueError, TypeError, RuntimeError) as e:
            logging.error(
                "❌ durante la costruzione del modello LSTM: %s", str(e)
            )
            raise

    def load_or_create_model(self, symbol):
        """
        Carica un modello esistente per un asset o ne crea uno nuovo.
        """
        try:
            model_file = self.get_model_file(symbol)
            if model_file.exists():
                return load_model(model_file)
            return self.build_lstm_model()
        except (IOError, ValueError, RuntimeError) as e:
            logging.error(
                "❌ caricamento o creazione modello x %s: %s", symbol, str(e)
            )
            raise

    def preprocess_data(self, raw_data, symbol, training=False):
        """
        Normalizza i dati grezzi utilizzando MinMaxScaler.
        """
        raw_data = np.array(raw_data).reshape(-1, 1)
        scaler_file = LSTM_FOLDER / f"scaler_{symbol}.bin"
        if training:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            data = self.scaler.fit_transform(raw_data)
            joblib.dump(self.scaler, scaler_file)
            return data
        else:
            self.scaler = joblib.load(scaler_file)
            return self.scaler.transform(raw_data)

    async def train_model(self, symbol, df):
        """
        Addestra il modello LSTM per un asset specifico (versione asincrona).
        """
        global ACTIVE_TRAININGS
        async with TRAIN_SEMAPHORE:
            logging.info(f"⏳ Attendo slot disponibile per il training di {symbol}...")
            ACTIVE_TRAININGS += 1
            logging.info(f"🚀 Training attivo per {symbol}. Totale attivi: {ACTIVE_TRAININGS}")
           
            if len(df) <= SEQUENCE_LENGTH:
                logging.warning("⚠️ Dati insufficienti per l'addestramento di %s", symbol)
                return
            
            # Limita a max 1000 righe per training per evitare memory error
            if len(df) > 1000:
                df = df[-1000:]
            try:
                local_model = self.load_or_create_model(symbol)
                memory = self.load_memory(symbol)
                df = apply_all_market_structure_signals(df)
                last_row = df[-1]
                signal_score = sum([
                    safe_int(get_col_val_safe(last_row, "ILQ_Zone")),
                    safe_int(get_col_val_safe(last_row, "fakeout_up")),
                    safe_int(get_col_val_safe(last_row, "fakeout_down")),
                    safe_int(get_col_val_safe(last_row, "volatility_squeeze")),
                    safe_int(get_col_val_safe(last_row, "micro_pattern_hft")),
                ])
                raw_close = df["close"].to_numpy()
                embeddings = self.get_all_embeddings(symbol)
                extra_features = np.concatenate([[signal_score], embeddings])
                if np.isnan(extra_features).any() or np.isinf(extra_features).any():
                    logging.error(f"❌ Extra features invalide per {symbol}: {extra_features}")
                    return
                if np.all(extra_features == 0):
                    logging.warning(f"⚠️ Extra features tutte nulle per {symbol}. Possibili dati mancanti.")
                data = self.preprocess_data(raw_close, symbol=symbol, training=True)
                x, y = [], []
                for i in range(len(data) - SEQUENCE_LENGTH):
                    x.append(data[i:i+SEQUENCE_LENGTH])
                    y.append(data[i+SEQUENCE_LENGTH])
                x = np.array(x)
                if len(x) == 0 or x.shape[1:] != (SEQUENCE_LENGTH, 1):
                    logging.error(f"❌ Sequenza input non valida per {symbol} → shape: {x.shape}")
                    return
                y = np.array(y)
                memory_tiled = np.tile(memory, (len(x), 1, 1))
                context_tiled = np.tile(extra_features, (len(x), SEQUENCE_LENGTH, 1))
                full_input = np.concatenate([x, memory_tiled, context_tiled], axis=2)
                if (
                    np.isnan(full_input).any() or np.isnan(y).any() or
                    np.isinf(full_input).any() or np.isinf(y).any()
                ):
                    logging.error(f"❌ Dati invalidi (NaN o inf) per {symbol}")
                    return
                if np.all(full_input == 0) or np.all(y == 0):
                    logging.warning(f"⚠️ Dati di training vuoti o nulli per {symbol}.")
                    return
                logging.debug(f"[DEBUG] full_input shape: {full_input.shape}, y shape: {y.shape}")

                early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
                await asyncio.to_thread(
                    local_model.fit,
                    full_input, y,
                    epochs=10,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=[early_stop]
                )
                await asyncio.to_thread(local_model.save, self.get_model_file(symbol))
                print(f"✅ Modello salvato per {symbol} in {self.get_model_file(symbol)}")
                self.save_memory(symbol, raw_close[-SEQUENCE_LENGTH:])
            except (pl.exceptions.PolarsError, ValueError, IOError) as e:
                logging.error("❌ Errore addestramento modello per %s: %s", symbol, str(e))
            finally:
                ACTIVE_TRAININGS -= 1
                logging.info(f"🏁 Fine training per {symbol}. Training attivi: {ACTIVE_TRAININGS}")

    def predict_price(
            self, symbol: str, full_state: Optional[np.ndarray] = None
        ) -> Optional[float]:
        """
        Prevede il prezzo futuro per un asset specifico.
        Usa full_state se fornito, altrimenti lo ricostruisce live.
        Riporta il risultato alla scala originale se possibile.
        """
        local_model: Model = self.load_or_create_model(symbol)
        symbol = str(symbol)

        try:
            if full_state is None:
                full_state = self.build_full_state(symbol)
            if full_state is None:
                logging.error(f"❌ Impossibile creare full_state per simbolo: {symbol}")
                return None

            # Sanifica lo stato
            if np.isnan(full_state).any() or np.isinf(full_state).any():
                logging.error(f"❌ Stato contiene NaN o inf per {symbol}")
                return None
            if np.all(full_state == 0) or full_state.size == 0:
                logging.error(f"❌ Stato vuoto o nullo per {symbol}")
                return None
            if full_state.ndim != 2:
                logging.error(f"❌ Stato deve avere ndim=2, trovato: {full_state.ndim}")
                return None

            # Limita a SEQUENCE_LENGTH
            if full_state.shape[0] > SEQUENCE_LENGTH:
                full_state = full_state[-SEQUENCE_LENGTH:]
            elif full_state.shape[0] < SEQUENCE_LENGTH:
                # Pad con zeri se mancano righe
                pad_rows = SEQUENCE_LENGTH - full_state.shape[0]
                full_state = np.pad(full_state, ((pad_rows, 0), (0, 0)), mode='constant')

            # Aggiungi batch dimension: shape (1, 50, N)
            reshaped_state = np.expand_dims(full_state, axis=0)

            # Verifica finale shape
            if reshaped_state.ndim != 3 or reshaped_state.shape[1] != SEQUENCE_LENGTH:
                logging.error(f"❌ Shape errata finale per {symbol}: {reshaped_state.shape}")
                return None

            try:
                prediction = local_model.predict(reshaped_state, verbose=0)[0][0]
            except Exception as e:
                logging.warning(f"🔄 Modello incompatibile per {symbol}, riaddestro: {e}")
                latest_df = self.get_latest_data(symbol)
                if latest_df is not None:
                    asyncio.create_task(self.train_model(symbol, latest_df))
                local_model = self.load_or_create_model(symbol)
                new_state = self.build_full_state(symbol)
                if new_state is None:
                    logging.error(f"❌ Stato nullo anche dopo retraining per {symbol}")
                    return None
                reshaped_state = np.expand_dims(new_state[-SEQUENCE_LENGTH:], axis=0)
                prediction = local_model.predict(reshaped_state, verbose=0)[0][0]

            # Denormalizza se possibile
            try:
                scaler_file = LSTM_FOLDER / f"scaler_{symbol}.bin"
                if scaler_file.exists():
                    self.scaler = joblib.load(scaler_file)
                    predicted_price = self.scaler.inverse_transform([[prediction]])[0][0]
                else:
                    predicted_price = prediction
            except Exception:
                predicted_price = prediction

            logging.info("📊 Prezzo previsto per %s: %.5f", symbol, predicted_price)
            return float(predicted_price)

        except Exception as e:
            logging.error("❌ durante la previsione per %s: %s", symbol, str(e))
            return None


    def build_full_state(self, symbol) -> Optional[np.ndarray]:
        """
        Crea lo stato completo (full_state) per un asset:
        - Dati normalizzati
        - Signal score
        - Embedding da 7 timeframe
        """
        logging.info(f"📌 Creazione full_state per {symbol}")
        try:
            df = pl.DataFrame(get_normalized_market_data(symbol))
            print(f"Shape df per {symbol}: {df.shape}")
            if df.is_empty() or df.shape[0] < SEQUENCE_LENGTH:
                return None

            df = apply_all_market_structure_signals(df)
            last_row = df[-1]

            signal_score = sum([
                safe_int(get_col_val_safe(last_row, "ILQ_Zone")),
                safe_int(get_col_val_safe(last_row, "fakeout_up")),
                safe_int(get_col_val_safe(last_row, "fakeout_down")),
                safe_int(get_col_val_safe(last_row, "volatility_squeeze")),
                safe_int(get_col_val_safe(last_row, "micro_pattern_hft")),
            ])

            embeddings = self.get_all_embeddings(symbol)
            extra_features = np.concatenate([[signal_score], embeddings])

            numeric_data = df.select(cs.numeric()).to_numpy()
            if numeric_data.shape[0] > 1000:
                numeric_data = numeric_data[-1000:]
            numeric_data = np.nan_to_num(numeric_data, nan=0.0, posinf=0.0, neginf=0.0)
            recent_sequence = numeric_data[-SEQUENCE_LENGTH:]

            # comprimi ogni riga aggiungendo extra_features a ognuna
            full_state = np.concatenate([
                recent_sequence,  # shape (50, num_base_features)
                 np.tile(extra_features, (SEQUENCE_LENGTH, 1))  # shape (50, num_extra_features)
            ], axis=1)

            return np.clip(full_state, -1, 1)
        except (ValueError, KeyError, AttributeError) as e:
            logging.error(
                "❌ build_full_state fallita per %s: %s", symbol, str(e)
            )
            return None
        
    def get_latest_data(self, symbol: str) -> Optional[pl.DataFrame]:
        """
        Recupera gli ultimi dati processati per un simbolo dal file processed_data.zstd.parquet.
        """
        try:
            df = pl.read_parquet(PROCESSED_DATA_PATH)
            df = df.filter(pl.col("symbol") == symbol).sort("timestamp", descending=True).head(3000)
            return df if not df.is_empty() else None
        except Exception as e:
            logging.error("❌ Impossibile caricare ultimi dati per %s: %s", symbol, e)
            return None


if __name__ == "__main__":
    model_instance = PricePredictionModel()
    while True:
        assets = get_available_assets()
        train_info = load_train_info()

        for sym in assets:
            try:
                raw_data = get_normalized_market_data(sym)
                if raw_data is None or isinstance(raw_data, dict) or raw_data.is_empty():
                    logging.error("❌ Nessun dato disponibile per %s", sym)
                    continue

                df_asset = pl.DataFrame(raw_data)
                selected_cols = analyze_and_select_features(df_asset)
                required_signals = [
                    "spread", "ILQ_Zone", "fakeout_up", "fakeout_down",
                    "volatility_squeeze", "micro_pattern_hft"
                ]
                selected_cols = list(set(selected_cols) | {col for col in required_signals if col in df_asset.columns})
                df_asset = df_asset.select(selected_cols)
                required = {"close", "high", "low", "volume"}
                if not required.issubset(set(df_asset.columns)):
                    logging.error("❌ Colonne mancanti per %s: %s", sym, required - set(df_asset.columns))
                    continue

                model_file = model_instance.get_model_file(sym)
                retrain = False

                # Controlla se il modello esiste e se i dati sono aggiornati
                data_len = df_asset.shape[0]
                prev_info = train_info.filter(pl.col("symbol") == sym)
                prev_len = prev_info[0, "data_len"] if not prev_info.is_empty() else -1

                if not model_file.exists() or data_len > prev_len:
                    if data_len > SEQUENCE_LENGTH:
                        model_instance.train_model(sym, df_asset)
                        # aggiorna info training
                        if not prev_info.is_empty():
                            train_info = train_info.filter(pl.col("symbol") != sym)
                        new_row = pl.DataFrame({"symbol": [sym], "data_len": [data_len]})
                        train_info = pl.concat([train_info, new_row])
                        save_train_info(train_info)
                    else:
                        logging.warning("⚠️ Dati insufficienti per %s", sym)
                        continue

                full_state = model_instance.build_full_state(sym)
                pred = model_instance.predict_price(sym, full_state=full_state)
                print(f"📊 Predizione per {sym}: {pred}")

            except (ValueError, KeyError, RuntimeError) as e:
                logging.error(
                    "❌ durante l'elaborazione dell'asset %s: %s", sym, str(e)
                )
        time.sleep(3600 * 6)
