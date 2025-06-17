"""
Modulo AI Model.
Questo modulo definisce una classe AIModel
per il trading algoritmico basato su
intelligenza artificiale. Include funzionalità
per la gestione del rischio,
ottimizzazione del portafoglio, previsione dei prezzi,
esecuzione di trade
e monitoraggio delle prestazioni.
Include inoltre cicli di ottimizzazione in background
e strategie per migliorare
continuamente l'efficacia del modello.
"""
import threading
import time
import asyncio
import os
import logging
from pathlib import Path
import polars as pl
from polars.selectors import numeric
import numpy as np
import MetaTrader5 as mt5
from drl_agent import DRLAgent
from drl_super_integration import DRLSuperManager
from demo_module import demo_trade
#from backtest_module import run_backtest
from strategy_generator import StrategyGenerator
from price_prediction import PricePredictionModel
from optimizer_core import OptimizerCore
from data_handler import get_normalized_market_data, get_available_assets, process_historical_data
from risk_management import RiskManagement
from volatility_tools import VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer
from smart_features import (
    apply_all_market_structure_signals #apply_all_advanced_features
)
from zone_utils import get_ilq_zone_for_symbol
from market_fingerprint import get_embedding_for_symbol #load_latest_embedding_and_features
from position_manager import PositionManager
from pattern_brain import PatternBrain
from constants import USE_DYNAMIC_STATE_SIZE, SEQUENCE_LENGTH, DESIRED_STATE_SIZE
from state_utils import sanitize_full_state, safe_int, get_col_val_safe
from ai_memory_sync import AIMemory as SyncMemory
from ai_memory_compact import AIMemoryCompact as CompactMemory
from realtime_update import update_memory_after_trade, update_realtime_embedding

print("ai_model.py caricato ✅")

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


DATA_DIR = Path("D:/trading_data")
MODEL_DIR = DATA_DIR / "models"

# Crea directory se non esistono
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Percorsi file
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.zstd.parquet"
EMBEDDING_FILE = DATA_DIR / "embedding_data.zstd.parquet"
DATA_FILE = MODEL_DIR / "ai_memory.parquet"
DB_FILE = MODEL_DIR / "trades.db"
TRADE_FILE = MODEL_DIR / "trades.parquet"
PERFORMANCE_FILE = MODEL_DIR / "performance.parquet"


def initialize_mt5():
    """
    Connessione sicura a MetaTrader 5
    """
    for _ in range(3):
        if mt5.initialize():
            logging.info(
                "✅ Connessione a MetaTrader 5 stabilita con successo."
            )
            return True
        logging.warning(
            "⚠️ Tentativo di connessione a MT5 fallito, riprovo..."
        )
    return False

if not os.path.exists(PROCESSED_DATA_PATH):
    print("⚠️ File dati processati non trovato, avvio generazione dati...")
    asyncio.run(process_historical_data())

def get_metatrader_balance():
    """
    Recupero saldo da MetaTrader 5
    """
    if not initialize_mt5():
        return 0
    account_info = mt5.account_info()
    return account_info.balance if account_info else 0


def fetch_account_balances():
    """
    Recupera automaticamente il saldo per ogni utente
    """
    balances = {
        "Danny": get_metatrader_balance(),
        "Giuseppe": get_metatrader_balance()
    }
    for user, balance in balances.items():
        logging.info("💰 Saldo account %s: %.2f €", user, balance)
    return balances

def safe_get(row, key):
    val = row[key] if key in row else 0
    if val is None:
        return 0
    try:
        return val.item() if hasattr(val, "item") else float(val)
    except Exception:
        return 0

class AIModel:
    """
    Classe che rappresenta un modello di intelligenza artificiale
    per il trading.
    Questa classe gestisce il caricamento e il salvataggio della memoria,
    l'adattamento delle dimensioni del lotto,
    l'esecuzione di operazioni di trading
    e l'aggiornamento delle performance basandosi su strategie definite.
    strategy_strength (float): La forza della strategia attuale.
    strategy_generator (object): Generatore per le strategie di trading.
    """
    def __init__(self, market_data, balances):
        self.market_data = market_data
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = {acc: RiskManagement() for acc in balances}
        self.memory = self.load_memory()
        self.strategy_strength = np.mean(self.memory) + 1
        self.strategy_generator = StrategyGenerator()
        self.memory_compact_path = MODEL_DIR / "ai_memory_compact.npz"
        # Memoria compatta per apprendimento automatico (embedding, profitto, confidenza)
        if self.memory_compact_path.exists():
            self.memory_compact = CompactMemory.load(self.memory_compact_path)
        else:
            self.memory_compact = CompactMemory()
        trade_count = self.memory_compact.total_trades
        # ✅ Switch automatico: se hai almeno 20 trade → usa metriche reali
        if trade_count >= 20:
            real_profit = self.memory_compact.get_last_profit()
            real_win_rate = self.memory_compact.get_win_rate()
            real_drawdown = self.memory_compact.get_drawdown()
            real_volatility = (
                np.std(np.diff(self.memory_compact.equity_curve))
                if len(self.memory_compact.equity_curve) > 2 else 0.0
            )
            self.strategy_generator.update_knowledge(
                use_real_metrics=True,
                profit=real_profit,
                win_rate=real_win_rate,
                drawdown=real_drawdown,
                volatility=real_volatility
            )
            logging.info("📈 Strategia aggiornata con metriche reali") 
        else:
            self.strategy_generator.update_knowledge(
                memory=self.memory,
                use_real_metrics=False
            )
            logging.info("🧪 Strategia aggiornata con memoria simulata")
        logging.info("📊 Trade registrati nella memoria compatta: %d", trade_count)
        self.balances = balances
        self.portfolio_optimizer = PortfolioOptimizer(
            market_data, balances, True
        )
        self.price_predictor = PricePredictionModel()
        state_size = DESIRED_STATE_SIZE if not USE_DYNAMIC_STATE_SIZE else self.calculate_dynamic_state_size()
        self.drl_agent = DRLAgent(state_size=state_size)
        self.active_assets = self.select_best_assets(market_data)
        self.pattern_brain = PatternBrain()
        self.drl_super_manager = DRLSuperManager(state_size=state_size)
        self.drl_super_manager.load_all()
        self.drl_super_manager.start_auto_training()
        self.last_max_trade_time = 0
        self.max_trade_cooldown = 3600
        self.memory_stats = SyncMemory()
        # Memoria sincronizzata AI a lungo termine (salvataggi persistenti)
        self.memory_path = Path("D:/trading_data/ai_memory_sync.npz")
        if self.memory_path.exists():
            self.ai_memory = SyncMemory.load(self.memory_path)
        else:
            self.ai_memory = SyncMemory()
        # Dizionario per eventuali memorie asset
        self.memories = {}

    def update(self, symbol: str, result: float):
        """
        Metodo di aggiornamento AI dopo ogni operazione.
        Aggiorna la memoria, l'agente DRL, e la conoscenza strategica compressa.
        """
        # Aggiorna la memoria compatta
        self.memory_compact.update(symbol=symbol, profit=result)

        # Aggiorna la memoria AI estesa
        self.memory_stats.update(symbol=symbol, profit=result)

        # Aggiorna l'agente DRL
        if symbol in self.memories:
            self.drl_agent.update(symbol=symbol, reward=result)

        # ✅ Aggiornamento conoscenza strategica
        use_real_metrics = self.memory_stats.total_trades >= 20

        if use_real_metrics:
            profit = self.memory_stats.get_last_profit(symbol)
            win_rate = self.memory_stats.get_win_rate(symbol)
            drawdown = self.memory_stats.get_drawdown(symbol)
            volatility = self.volatility_predictor.predict(symbol)
        else:
            # fallback simulato
            profit = np.random.uniform(-0.5, 1.5)
            win_rate = np.random.uniform(0.5, 0.8)
            drawdown = np.random.uniform(0, 0.1)
            volatility = np.random.uniform(0.01, 0.05)

        self.strategy_generator.update_knowledge(
            profit=profit,
            win_rate=win_rate,
            drawdown=drawdown,
            volatility=volatility
        )

        logging.info("🔄 AIModel updated with result %.4f for %s", result, symbol)

    def calculate_dynamic_state_size(self):
        """
        calcola la dimensione dello stato in modo dinamico
        """
        try:
            sample_asset = next(iter(self.active_assets))
            market_data = self.market_data[sample_asset]
            sequence_length = 50  # Assumendo che SEQUENCE_LENGTH sia 10 come in drl_agent.py
            num_features = market_data.select(numeric()).shape[1]
            return sequence_length * num_features
        except Exception as e:
            logging.warning(f"⚠️ Errore nel calcolo dello state_size dinamico: {e}. Uso fallback {DESIRED_STATE_SIZE}.")
            return DESIRED_STATE_SIZE

    def load_memory(self):
        """
        Carica i dati di memoria compressi da un file Parquet,
        se esistente.
        numpy.ndarray: La memoria caricata come un array Numpy.
        Se il file di memoria non esiste,
        restituisce un array vuoto predefinito.
        """
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria compressa...")
            loaded_memory = pl.read_parquet(DATA_FILE)["memory"].to_numpy()
            return np.mean(loaded_memory, axis=0, keepdims=True)
        return np.zeros(1, dtype=np.float32)

    def save_memory(self, symbol, profit, embedding, confidence, decision, result, AIMemory, timeframe):
        """
        Salva un nuovo valore nella memoria compressa,
        aggiornando il file Parquet.
        new_value (numpy.ndarray):
        Il nuovo valore da aggiungere alla memoria.
        """
        key = (symbol, timeframe)
        if key not in self.memories:
            self.memories[key] = AIMemory()
        self.memories[key].update(
            profit=profit,
            embedding=embedding,
            confidence=confidence,
            decision=decision,
            result=result
        )
        self.memories[key].save(f"{MODEL_DIR}/ai_memory_{symbol}_{timeframe}.npz")
        logging.info("💾 Memoria IA compatta aggiornata.")

    def update_performance(
        self, account, symbol, action,
        lot_size, profit, strategy
    ):
        """
        Aggiorna le informazioni di performance relative a
        un'operazione di trading.
        """
        # Carica i dati esistenti
        if TRADE_FILE.exists():
            df = pl.read_parquet(TRADE_FILE)
            df = df.with_columns([
                pl.col("lot_size").cast(pl.Float64),
                pl.col("profit").cast(pl.Float64)
            ])
        else:
            df = pl.DataFrame({
                "account": pl.Series([], dtype=pl.Utf8),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "action": pl.Series([], dtype=pl.Utf8),
                "lot_size": pl.Series([], dtype=pl.Float64),
                "profit": pl.Series([], dtype=pl.Float64),
                "strategy": pl.Series([], dtype=pl.Utf8),
            })

        # Cerca se esiste già un trade per questo account e simbolo
        existing_trade = df.filter(
            (df["account"] == account) & (df["symbol"] == symbol)
        )

        if len(existing_trade) > 0:
            # Aggiorna il valore invece di creare una nuova riga
            df = df.with_columns([
                pl.when((df["account"] == account) & (df["symbol"] == symbol))
                .then(pl.lit(profit)).otherwise(df["profit"]).alias("profit")
            ])
        else:
            # Se non esiste, aggiunge una nuova entry
            new_entry = pl.DataFrame({
                "account": [account],
                "symbol": [symbol],
                "action": [action],
                "lot_size": [float(lot_size)],
                "profit": [float(profit)],
                "strategy": [strategy]
            })
            df = pl.concat([df, new_entry])

        df.write_parquet(TRADE_FILE, compression="zstd")
        logging.info(
            "📊 Trade aggiornato per %s su %s: Profit %s | Strategia: %s",
            account, symbol, profit, strategy
        )

    def adapt_lot_size(
        self, account, symbol, success_probability,
        confidence_score, predicted_volatility,
        action, current_price
    ):
        """
        Calcola la dimensione del lotto in modo dinamico, ultra-intelligente e integrato.
        Combina risk manager, confidenza AI, ILQ zone, momentum e volatilità.
        """
        risk_manager = self.risk_manager[account]
        base_lot = risk_manager.calculate_position_size(
            self.balances[account], symbol
        )

        multiplier = 1.0
        if success_probability > 0.9 and confidence_score > 0.9:
            multiplier *= 1.5

        if predicted_volatility:
            multiplier *= np.clip(1 / (1 + predicted_volatility), 0.5, 1.2)

        multiplier *= np.clip(self.strategy_strength, 0.5, 3.0)

        final_lot = base_lot * multiplier
        ilq_zone = get_ilq_zone_for_symbol(symbol)
        if ilq_zone:
            if action == "buy" and current_price <= ilq_zone["resistance"]:
                final_lot *= 1.2  # 20% in più di size
            elif action == "sell" and current_price >= ilq_zone["support"]:
                final_lot *= 1.2
        max_lot = (
            self.balances[account] * risk_manager.risk_settings["max_exposure"]
        )
        return max(0.01, min(final_lot, max_lot))

    def execute_trade(self, account, symbol, timeframe, action, lot_size, risk, strategy, sl, tp, trade_profit, embedding, confidence_score, action_rl):
        """
        Esegue un'operazione di trading su MetaTrader 5
        in base ai parametri specificati.
        """
        if not mt5.symbol_select(symbol, True):
            logging.error("❌ Simbolo %s non selezionabile o assente su MT5", symbol)
            return

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"❌ Nessun tick disponibile per {symbol}")
            return

        price = tick.ask if action == "buy" else tick.bid

        if hasattr(sl, "item"):
            sl = sl.item(-1)
        if hasattr(tp, "item"):
            tp = tp.item(-1)

        sl = float(sl) if sl is not None else 0.0
        tp = float(tp) if tp is not None else 0.0

        # Correzione automatica SL/TP se invalidi
        info = mt5.symbol_info(symbol)
        stops_level = getattr(info, "stops_level", None)

        if stops_level is None or stops_level <= 0:
            stops_level = 100  # Valore di fallback sicuro
        min_distance = stops_level * info.point
 
        if action == "buy":
            if sl >= price or tp <= price or (price - sl) < min_distance or (tp - price) < min_distance:
                sl = round(price - 10 * min_distance, 2)
                tp = round(price + 20 * min_distance, 2)
        else:  # sell
            if sl <= price or tp >= price or (sl - price) < min_distance or (price - tp) < min_distance:
                sl = round(price + 10 * min_distance, 2)
                tp = round(price - 20 * min_distance, 2)

        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(max(lot_size, 0.01), 2),
            "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": f"AI Trade ({strategy})",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(order)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error("❌ ERRORE invio ordine: %s | Codice: %s", result.comment, result.retcode)
        status = (
            "executed" if result.retcode == mt5.TRADE_RETCODE_DONE
            else "failed"
        )
        if status == "executed" and trade_profit is not None and embedding is not None:
            update_memory_after_trade(
                symbol,
                embedding,
                confidence_score,
                trade_profit,
                action_rl,
                result="success"
            )
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.risk_manager[account].max_trades -= 1

        self.update_performance(
            account, symbol, action, lot_size,
            0,  # profitto non disponibile subito
            strategy
        )

        self.save_memory(
            symbol=symbol,
            timeframe=timeframe,
            profit=trade_profit,
            embedding=embedding,
            confidence=confidence_score,
            decision=action_rl,
            result=1 if trade_profit > 0 else 0
        )
        if trade_profit is not None and embedding is not None:
            update_memory_after_trade(
                symbol,
                embedding,
                confidence_score,
                trade_profit,
                action_rl,
                result="success"
            )
        # ✅ Salva nella memoria "sync" (storico a lungo termine)
        self.ai_memory.update(trade_profit, embedding, confidence_score)
        self.ai_memory.save(self.memory_path)

        # ✅ Salva anche nella memoria "compact" (embedding compressi)
        self.memory_compact.update(trade_profit, embedding, confidence_score)
        self.memory_compact.save(self.memory_compact_path)
     
        self.strategy_generator.update_strategies(
            strategy,
            1 if status == "executed" else -10
        )

        logging.info(
            "✅ Trade %s per %s su %s: %s %.2f lotto | Strat: %s | Rischio: %.2f",
            status, account, symbol, action, lot_size, strategy, risk
        )
        self.update(symbol, trade_profit or 0.0)
        logging.info("🟢 Trade ESEGUITO su %s (%s) con lotto %.2f", symbol, action, lot_size)


    def select_best_assets(self, market_data):
        """
        Seleziona automaticamente gli asset con il miglior rendimento storico
        """
        assets_performance = {
            asset: market_data[asset]["close"].pct_change().mean()
            for asset in market_data.keys()
        }
        sorted_assets = sorted(
            assets_performance, key=assets_performance.get, reverse=True
        )
        logging.info(
            "📈 Asset selezionati per il trading: %s",
            sorted_assets[:5]
        )
        return sorted_assets[:5]  # Seleziona i 5 asset migliori

    async def decide_trade(self, symbol):
        """
        Analizza i dati di mercato per un determinato simbolo e decide se eseguire un'operazione di trading.
        """
        self.balances = fetch_account_balances()

        raw_df = get_normalized_market_data(symbol)
        if raw_df is None or raw_df.shape[0] < SEQUENCE_LENGTH:
            logging.warning("⚠️ Dati insufficienti per %s", symbol)
            return

        raw_df = apply_all_market_structure_signals(raw_df)
        last_row = raw_df[-1]

        signal_score = sum([
            safe_int(get_col_val_safe(last_row, "ILQ_Zone")),
            safe_int(get_col_val_safe(last_row, "fakeout_up")),
            safe_int(get_col_val_safe(last_row, "fakeout_down")),
            safe_int(get_col_val_safe(last_row, "volatility_squeeze")),
            safe_int(get_col_val_safe(last_row, "micro_pattern_hft")),
        ])


        embeddings = np.concatenate([
            get_embedding_for_symbol(symbol, tf)
            for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        ])
        extra_features = np.concatenate([[signal_score], embeddings])

        numeric_data = raw_df.select(numeric()).to_numpy()
        numeric_data = np.nan_to_num(numeric_data, nan=0.0, posinf=0.0, neginf=0.0)
        recent_sequence = numeric_data[-SEQUENCE_LENGTH:]
        if recent_sequence.shape[0] != SEQUENCE_LENGTH:
            logging.warning(f"⚠️ Sequenza troppo corta per {symbol}, skippo.")
            return
        if recent_sequence.shape[1] + len(extra_features) > 3000:  # Limite arbitrario per sicurezza
            logging.warning(f"⚠️ Dimensione troppo grande del full_state per {symbol}")
            return

        full_state = np.concatenate(
            [recent_sequence, np.tile(extra_features, (SEQUENCE_LENGTH, 1))],
            axis=1
        )
        full_state = np.clip(full_state, -1, 1)
        full_state = sanitize_full_state(full_state)

        update_realtime_embedding(symbol, raw_df, timeframe="1m")

        predicted_price = self.price_predictor.predict_price(symbol=symbol, full_state=full_state)

        pattern_data = [
           safe_int(get_col_val_safe(last_row, "ILQ_Zone")),
            safe_int(get_col_val_safe(last_row, "fakeout_up")),
            safe_int(get_col_val_safe(last_row, "fakeout_down")),
            safe_int(get_col_val_safe(last_row, "volatility_squeeze")),
            safe_int(get_col_val_safe(last_row, "micro_pattern_hft")),
        ]
        pattern_confidence = self.pattern_brain.predict_score(pattern_data)

        for account in self.balances:
            if self.risk_manager[account].max_trades <= 0:
                now = time.time()
                if hasattr(self, "last_max_trade_time") and now - self.last_max_trade_time < self.max_trade_cooldown:
                    continue
                self.last_max_trade_time = now
                logging.warning("❌ Max trades raggiunti per %s", account)
                continue
            if full_state.shape[0] > DESIRED_STATE_SIZE:
                full_state = full_state[:DESIRED_STATE_SIZE]
            elif full_state.shape[0] < DESIRED_STATE_SIZE:
                full_state = np.pad(full_state, (0, DESIRED_STATE_SIZE - full_state.shape[0]), mode='constant')
            action_rl, confidence_score, algo_used = self.drl_super_manager.get_best_action_and_confidence(full_state)
            ilq_zone = get_ilq_zone_for_symbol(symbol)
            current_price = raw_df["close"].to_numpy()[-1]

            if ilq_zone and "support" in ilq_zone and "resistance" in ilq_zone and current_price:
                if action_rl == "buy" and current_price > ilq_zone["resistance"]:
                    logging.info(f"🚫 BUY bloccato fuori da zona ILQ su {symbol}")
                    return
                elif action_rl == "sell" and current_price < ilq_zone["support"]:
                    logging.info(f"🚫 SELL bloccato fuori da zona ILQ su {symbol}")
                    return
                if action_rl == "buy" and current_price <= ilq_zone["resistance"]:
                    confidence_score += 0.05
                elif action_rl == "sell" and current_price >= ilq_zone["support"]:
                    confidence_score += 0.05

            success_probability = confidence_score * pattern_confidence

            if action_rl == 1:
                action = "buy"
            elif action_rl == 2:
                action = "sell"
            else:
                return

            if signal_score < 0.6:
                return

            predicted_volatility = self.volatility_predictor.predict_volatility(full_state.reshape(1, -1))[0]
            sl, ts, tp = self.risk_manager[account].adaptive_stop_loss(current_price, symbol)

            sl_val = float(sl) if isinstance(sl, (int, float)) else float(sl.to_numpy()[0])
            ts_val = float(ts) if isinstance(ts, (int, float)) else float(ts.to_numpy()[0])
            if ts_val > sl_val and (ts_val - sl_val) < (0.002 * current_price):
                logging.info("⛔ Trailing Stop troppo stretto. Nessun trade su %s", symbol)
                return

            lot_size = self.adapt_lot_size(
                account, symbol,
                success_probability, confidence_score,
                predicted_volatility
            )
            lot_size = min(max(lot_size, 0.01), 0.05)
            self.risk_manager[account].adjust_risk(symbol)

            if lot_size < 0.01:
                logging.warning("⛔ Lotto troppo piccolo, annullo trade su %s", symbol)
                return

            logging.info(
                "🤖 Azione AI: %s | Algo: %s | Confidenza: %.2f | Score: %d",
                action, algo_used, confidence_score, signal_score
            )

            trade_profit = predicted_price - current_price
            strategy, strategy_weight = self.strategy_generator.select_best_strategy(raw_df)

            self.execute_trade(account, symbol, action, lot_size, success_probability, strategy, sl, tp)

            self.strategy_strength = np.clip(
                self.strategy_strength * (1 + (strategy_weight - 0.5)),
                0.5, 3.0
            )
            vol = safe_get(last_row, "volatility")
            self.strategy_generator.update_knowledge(
                profit=trade_profit,
                win_rate=1 if trade_profit > 0 else 0,
                drawdown=abs(min(0, trade_profit)),
                volatility=vol
            )
            self.volatility_predictor.update(full_state.reshape(1, -1), vol)
            #self.volatility_predictor.update(full_state.reshape(1, -1), last_row["volatility"].item())

            if pattern_confidence < 0.2:
                return

            if success_probability > 0.3:
                self.risk_manager[account].max_trades -= 1
                self.drl_agent.update(full_state, 1 if trade_profit > 0 else 0)
                self.drl_super_manager.update_all(full_state, 1 if trade_profit > 0 else 0)

                expected_state_size = self.drl_super_manager.super_agents[algo_used].env.observation_space.shape[0]
                if full_state.shape[0] == expected_state_size:
                    self.drl_super_manager.reinforce_best_agent(full_state, 1)
                else:
                    logging.warning(
                        f"⚠️ Stato RL non coerente: atteso {expected_state_size}, ricevuto {full_state.shape[0]}. Reinforcement saltato."
                    )
            else:
                logging.info("🚫 Nessun trade su %s per %s. Avvio Demo.", symbol, account)
                demo_trade(symbol, raw_df)
                self.drl_agent.update(full_state, 1 if trade_profit > 0 else 0)
                self.drl_super_manager.update_all(full_state, 1 if trade_profit > 0 else 0)


def background_optimization_loop(
    ai_model_instance, interval_seconds=43200
):
    """
    Esegue un ciclo continuo per ottimizzare le strategie e il modello AI.
    Args:ai_model_instance (AIModel):Istanza del modello AI da ottimizzare.
    interval_seconds (int, opzionale):Intervallo di tempo in secondi tra
    due cicli di ottimizzazione. Default: 43200 secondi (12 ore).
    """
    optimizer = OptimizerCore(
        strategy_generator=ai_model_instance.strategy_generator,
        ai_model=ai_model_instance
    )
    while True:
        optimizer.run_full_optimization()
        time.sleep(interval_seconds)


def loop_position_monitor(position_manager_instance):
    """
    Controlla e gestisce tutte le posizioni aperte in autonomia.
    """
    while True:
        position_manager_instance.monitor_open_positions()
        time.sleep(10)


if __name__ == "__main__":
    # 🔄 Recupera tutti gli asset disponibili (preset o dinamici)
    assets = get_available_assets()
    print(f"📦 Asset caricati da get_available_assets(): {assets}")

    # 📊 Crea un dizionario con i dati normalizzati per ciascun asset
    all_market_data = {
        symbol: data
        for symbol in assets
        if (data := get_normalized_market_data(symbol)) is not None
    }

    ai_model = AIModel(all_market_data, fetch_account_balances())

    #threading.Thread(
        #target=ai_model.strategy_generator.continuous_self_improvement,
        #daemon=True
    #).start()

    thread = threading.Thread(
        target=background_optimization_loop,
        args=(ai_model,), daemon=True
    )
    thread.start()

    pm = PositionManager()

    threading.Thread(
        target=lambda: loop_position_monitor(pm), daemon=True
    ).start()

    while True:
        for asset in ai_model.active_assets:
            asyncio.run(ai_model.decide_trade(asset))
        time.sleep(10)