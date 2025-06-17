"""
start.py
Avvia l’intero sistema di trading AI: raccolta dati realtime, training AI, previsione prezzi, monitoraggio posizioni.
"""
import builtins
import warnings
import sys
#from ecosystem_guardian import start_guardian_in_background

import os
import gc
import psutil
import subprocess
import tempfile
import logging
import threading
import asyncio
import time

# 🔒 Forza chiusura dei file mappati in memoria prima di avviare il sistema
def cleanup_locked_files(target_folder="D:/trading_data"):
    current_process = psutil.Process(os.getpid())
    open_files = current_process.open_files()
    for f in open_files:
        if target_folder.replace("\\", "/") in f.path.replace("\\", "/"):
            try:
                os.close(f.fd)
                logging.info(f"🔓 Chiuso file lockato: {f.path}")
            except Exception as e:
                logging.warning(f"⚠️ Impossibile chiudere {f.path}: {e}")

    # Forza il garbage collector (chiude file ancora referenziati da oggetti Python)
    gc.collect()

# Esegui cleanup all'avvio
cleanup_locked_files()

from data_loader import (
    load_config, load_preset_assets, load_auto_symbol_mapping,
    dynamic_assets_loading, USE_PRESET_ASSETS
)
from data_handler import get_available_assets, get_normalized_market_data, get_realtime_data
from market_fingerprint import update_embedding_in_processed_file
from ai_model import AIModel, fetch_account_balances, background_optimization_loop
from position_manager import PositionManager
from price_prediction import PricePredictionModel


os.environ["TMP"] = "D:/temp"
os.environ["TEMP"] = "D:/temp"
os.makedirs("D:/temp", exist_ok=True)
import tempfile
tempfile.tempdir = "D:/temp"


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# def refresh_processed_data_loop():
    # from data_handler import PROCESSED_DATA_PATH, run_full_pipeline
    # import datetime

    # REFRESH_INTERVAL_HOURS = 3

    # while True:
        # try:
            # if not os.path.exists(PROCESSED_DATA_PATH):
                # logging.warning("🧩 File dati processati assente. Rigenerazione avviata.")
                # run_full_pipeline()
            # else:
                # last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(PROCESSED_DATA_PATH))
                # age = (datetime.datetime.now() - last_modified).total_seconds() / 3600.0
                # if age > REFRESH_INTERVAL_HOURS:
                    # logging.info(f"♻️ File dati processati ha più di {REFRESH_INTERVAL_HOURS}h. Rigenerazione...")
                    # run_full_pipeline()
        # except Exception as e:
            # logging.error("❌ Errore durante il refresh dati: %s", str(e))
        # time.sleep(600)

# 🔁 Thread per sincronizzare periodicamente gli embedding temporanei
def start_embedding_sync_loop():
    from embedding_sync import sync_temp_embeddings_to_main
    while True:
        try:
            sync_temp_embeddings_to_main()
            time.sleep(15)
        except Exception as e:
            logging.warning(f"⚠️ Errore nel sync loop embedding: {e}")

class TradingSystem:
    def __init__(self):
        self.config = load_config()
        logging.info("✅ Configurazione caricata: %s", self.config)

        mapping = load_auto_symbol_mapping()
        if USE_PRESET_ASSETS:
            raw_assets = load_preset_assets()
            self.assets = []
            for broker, block in raw_assets.items():
                if not isinstance(block, dict) or "assets" not in block or "enabled" not in block:
                    logging.warning(f"⚠️ Broker '{broker}' ignorato: formato invalido.")
                    continue
                if block["enabled"]:
                    assets = block["assets"]
                    if isinstance(assets, list):
                        self.assets.extend(assets)
                    logging.info(f"✅ Asset abilitati da '{broker}': {assets}")
                else:
                    logging.info(f"⛔ '{broker}' disabilitato, ignorato.")
            if not self.assets:
                raise ValueError("❌ Nessun asset abilitato trovato in preset_assets.json.")
        else:
            dynamic_assets_loading(mapping)
            self.assets = get_available_assets()
            logging.info("🔄 Asset caricati dinamicamente.")

        self.market_data = {
            symbol: data for symbol in self.assets
            if (data := get_normalized_market_data(symbol)) is not None
        }
        self.balances = fetch_account_balances()

        self.ai_model = AIModel(self.market_data, self.balances)
        self.position_manager = PositionManager()

    async def start_background_tasks(self):
        # Ottimizzazione AI Model (thread, va bene)
        threading.Thread(
            target=background_optimization_loop,
            args=(self.ai_model,), daemon=True
        ).start()
        logging.info("🔁 Ottimizzazione AI Model avviata.")

        # Monitoraggio posizioni aperte (thread, va bene)
        threading.Thread(
            target=self.monitor_positions_loop,
            daemon=True
        ).start()
        logging.info("🛡️ Monitoraggio posizioni attivo.")

        # Refresh automatico dati (thread, va bene)
        # threading.Thread(
            # target=refresh_processed_data_loop,
            # daemon=True
        # ).start()
        # logging.info("📊 Refresh automatico dati attivo.")

        # Aggiornamento dati realtime (usa asyncio.create_task)
        asyncio.create_task(get_realtime_data(self.assets))
        logging.info("📡 Aggiornamento dati realtime attivo.")
        asyncio.create_task(self.ai_model.strategy_generator.continuous_self_improvement())
        logging.info("🧠 Continuous Self-Improvement avviato.")

        # SuperAgent runner separato (subprocess, va bene)
        try:
            subprocess.Popen([sys.executable, os.path.join(os.getcwd(), "super_agent_runner.py")])
            logging.info("🚀 Super Agent Runner avviato.")
        except (FileNotFoundError, OSError) as e:
            logging.error("❌ Errore avvio Super Agent Runner: %s", e)

    def monitor_positions_loop(self):
        while True:
            self.position_manager.monitor_open_positions()
            time.sleep(10)


    async def run(self):
        await self.start_background_tasks()

        # Avvia Price Prediction
        try:
            predictor = PricePredictionModel()
            for symbol in self.assets:
                try:
                    df = get_normalized_market_data(symbol)
                    model_file = predictor.get_model_file(symbol)
                    if not model_file.exists():
                        await predictor.train_model(symbol, df)
                    embedding = predictor.predict_price(symbol)
                    if embedding is not None and not all(v == 0 for v in embedding):
                        update_embedding_in_processed_file(symbol, df, length=len(embedding))
                except Exception as e:
                    logging.warning(f"⚠️ Errore previsione {symbol}: {e}")
        except Exception as e:
            logging.warning(f"⚠️ Errore inizializzazione PricePredictionModel: {e}")

        logging.info("🏁 Trading system avviato. Inizio ciclo decisionale AI.")
        while True:
            for asset in self.ai_model.active_assets:
                await self.ai_model.decide_trade(asset)
            await asyncio.sleep(10)

if __name__ == "__main__":
    ENABLE_GUARDIAN = True
    SILENCE_SYSTEM_OUTPUT = True

    if ENABLE_GUARDIAN:
        try:
            from ecosystem_guardian import start_guardian_in_background
            start_guardian_in_background()
            if SILENCE_SYSTEM_OUTPUT:
                builtins.print = lambda *args, **kwargs: None
                warnings.filterwarnings("ignore")
                sys.stderr = open(os.devnull, 'w')
        except Exception as e:
            print(f"❌ Impossibile avviare il Guardian: {e}")
    try:
        system = TradingSystem()
        # Avvio thread di sincronizzazione embedding
        threading.Thread(target=start_embedding_sync_loop, daemon=True).start()
        logging.info("🔁 Loop di sincronizzazione embedding temporanei avviato.")
        asyncio.run(system.run())
    except KeyboardInterrupt:
        print("🛑 Arrestato manualmente.")