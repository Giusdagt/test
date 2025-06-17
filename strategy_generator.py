"""
Modulo Strategy Generator
Questo modulo definisce la classe StrategyGenerator per la generazione
e l'ottimizzazione di strategie di trading algoritmico.
Include funzionalità
per rilevare anomalie di mercato, generare strategie basate su indicatori
tecnici e migliorare continuamente le strategie in base alle prestazioni.
"""
import inspect
import asyncio
import logging
import threading
import time
from pathlib import Path
import numpy as np
import polars as pl
from indicators import TradingIndicators

print("strategy_generator.py caricato✅")

# Configura il logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

MODEL_DIR = Path("D:/trading_data/models")
STRATEGY_FILE = MODEL_DIR / "strategies_compressed.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class StrategyGenerator:
    """
    Classe per la generazione e l'ottimizzazione
    di strategie di trading algoritmico.
    Questa classe include metodi per:
    - Rilevare anomalie di mercato.
    - Generare strategie basate su indicatori tecnici.
    - Aggiornare la conoscenza compressa basandosi
    sulle performance delle strategie.
    - Migliorare continuamente le strategie tramite un processo iterativo.
    """
    def __init__(self,):
        self.indicators = None
        self.all_indicators = {}
        self.compressed_knowledge = self.load_compressed_knowledge()
        self.market_anomalies = []
        self.generated_strategies = {}
        self.latest_market_data = None
        logger.info("StrategyGenerator initialized")

    def get_all_indicators(self):
        indicator_data = self.indicators.calculate_all_indicators()
        return {
            "RSI": lambda: indicator_data["RSI"].item(-1),
            "MACD": lambda: indicator_data["MACD"].item(-1),
            "MACD_Signal": lambda: indicator_data["MACD_Signal"].item(-1),
            "MACD_Hist": lambda: indicator_data["MACD_Hist"].item(-1),
            "ADX": lambda: indicator_data["ADX"].item(-1),
            "BB_Upper": lambda: indicator_data["BB_Upper"].item(-1),
            "BB_Middle": lambda: indicator_data["BB_Middle"].item(-1),
            "BB_Lower": lambda: indicator_data["BB_Lower"].item(-1),
            "EMA_50": lambda: indicator_data["EMA_50"].item(-1),
            "EMA_200": lambda: indicator_data["EMA_200"].item(-1),
            "SMA_100": lambda: indicator_data["SMA_100"].item(-1),
            "VWAP": lambda: indicator_data["VWAP"].item(-1) if "VWAP" in indicator_data else 0,
            "STOCH_K": lambda: indicator_data["STOCH_K"].item(-1),
            "STOCH_D": lambda: indicator_data["STOCH_D"].item(-1),
            "momentum": lambda: indicator_data["ROC_10"].item(-1),
            "Donchian_Upper": lambda: indicator_data["Donchian_Upper"].item(-1),
            "volatility": lambda: indicator_data["volatility"].item(-1) if "volatility" in indicator_data else 1.0,
            "OBV": lambda: indicator_data["OBV"].item(-1),
            "Ichimoku_Tenkan": lambda: indicator_data["Ichimoku_Tenkan"].item(-1),
            "Ichimoku_Kijun": lambda: indicator_data["Ichimoku_Kijun"].item(-1),
            "Senkou_Span_A": lambda: indicator_data["Senkou_Span_A"].item(-1),
            "Senkou_Span_B": lambda: indicator_data["Senkou_Span_B"].item(-1),
            "SuperTrend_Upper": lambda: indicator_data["SuperTrend_Upper"].item(-1),
            "SuperTrend_Lower": lambda: indicator_data["SuperTrend_Lower"].item(-1),
            "ATR": lambda: indicator_data["ATR"].item(-1),
            "CCI": lambda: indicator_data["CCI"].item(-1),
            "MFI": lambda: indicator_data["MFI"].item(-1),
            "VWAP": lambda: indicator_data["VWAP"].item(-1) if "VWAP" in indicator_data else 0,
        }
    
    def ensure_knowledge_column(self, df: pl.DataFrame) -> pl.DataFrame:
        if "knowledge" not in df.columns:
            logging.warning("⚠️ Colonna 'knowledge' mancante, viene creata.")
            df = df.with_columns(
                pl.Series("knowledge",
                        [np.zeros(100, dtype=np.float32).tobytes()] * df.height,
                        dtype=pl.Binary)
            )
        return df

    def load_compressed_knowledge(self):
        """
        Carica la conoscenza compressa da un file Parquet.
        """
        if STRATEGY_FILE.exists():
            df = pl.read_parquet(STRATEGY_FILE)
            df = self.ensure_knowledge_column(df)
            value = df["knowledge"][0]
            if isinstance(value, bytes):
                logging.info("✅ Loaded compressed knowledge from file")
                return np.frombuffer(value, dtype=np.float32)
            else:
                logging.warning("⚠️ Contenuto 'knowledge' non in formato bytes, uso fallback.")
        else:
            logging.info("ℹ️ Nessun file di conoscenza esistente trovato")

        # fallback in ogni caso
        return np.zeros(100, dtype=np.float32)


    def save_compressed_knowledge(self):
        """
        Salva la conoscenza compressa in un file Parquet.
        Questo metodo converte l'array di conoscenza compressa
        in un formato binario
        e lo salva in un file Parquet utilizzando la compressione Zstandard.
        Questo file può essere usato per recuperare
        la conoscenza in esecuzioni successive.
        Updates:
        - Scrive il contenuto di `self.compressed_knowledge`
        in `STRATEGY_FILE`.
        """
        df = pl.DataFrame({"knowledge": [self.compressed_knowledge.tobytes()]})
        df.write_parquet(STRATEGY_FILE, compression="zstd")
        logging.info("Saved compressed knowledge to file")

    def detect_market_anomalies(self, market_data):
        """
        Rileva anomalie nel mercato basandosi su volatilità e volumi.
        Questo metodo analizza i dati di mercato
        per identificare condizioni anomale,
        come un'elevata volatilità o un improvviso picco nei volumi di scambio.
        Le anomalie rilevate vengono
        aggiunte alla lista `self.market_anomalies`.
        """
        high_volatility = market_data["volatility"].item(-1) > 2.0
        sudden_volume_spike = (
            market_data["volume"].item(-1) >
            market_data["volume"].mean() * 3
        )
        if high_volatility or sudden_volume_spike:
            self.market_anomalies.append("Manipolazione Rilevata")
            logging.warning(
                "Rilevata anomalia di mercato: elevata volatilità o "
                "picco improvviso del volume"
            )

    def update_knowledge(self, memory=None, use_real_metrics=None):
        """
        Aggiorna la conoscenza compressa usando AIMemory.
        Se use_real_metrics è False, usa valori simulati.
        """
        if use_real_metrics is None:
            use_real_metrics = memory and getattr(memory, "total_trades", 0) >= 20
        if use_real_metrics and memory:
            profit = memory.total_profit
            win_rate = memory.winrate
            drawdown = memory.max_drawdown
            volatility = memory.profit_std
        else:
            profit = np.random.uniform(-50, 50)
            win_rate = np.random.uniform(0.2, 0.8)
            drawdown = np.random.uniform(0, 30)
            volatility = np.random.uniform(0, 20)

        efficiency_score = (
            (profit * 0.5) + (win_rate * 0.3) -
            (drawdown * 0.1) - (volatility * 0.1)
        )
        self.compressed_knowledge = np.clip(
            self.compressed_knowledge + (efficiency_score / 1000),
            0, 1
        )

        logging.info(
            f"🔄 Knowledge updated: profit={profit:.2f}, winrate={win_rate:.2f}, "
            f"drawdown={drawdown:.2f}, volatility={volatility:.2f}"
        )
        self.save_compressed_knowledge()

        # Compressione incrementale
        if len(self.compressed_knowledge) > 50:
            self.compressed_knowledge = np.mean(
                self.compressed_knowledge.reshape(-1, 2), axis=1
            )
            self.save_compressed_knowledge()
            logging.info("📉 Knowledge compressed incrementally")

    def generate_new_strategies(self, market_data):
        """
        Genera nuove strategie di trading basate sui
        dati di mercato più recenti.
        Questo metodo utilizza i valori degli
        indicatori tecnici calcolati
        dai dati di mercato per creare nuove strategie di trading.
        vengono salvate nel dizionario `self.generated_strategies`.
        """
        self.indicators = TradingIndicators(data=market_data)
        self.all_indicators = self.get_all_indicators()
        indicator_values = (
            {name: func() for name,
            func in self.all_indicators.items()}
        )
        new_strategies = {
            "strategy_1": (
                indicator_values["RSI"] < 30 and
                indicator_values["MACD"] > indicator_values["MACD_Signal"] and
                indicator_values["ADX"] > 25
            ),
            "strategy_2": (
                indicator_values["RSI"] > 70 and
                indicator_values["MACD"] < indicator_values["MACD_Signal"] and
                indicator_values["BB_Upper"] > market_data["close"].item(-1)
            ),
            "strategy_3": (
                indicator_values["EMA_50"] > indicator_values["EMA_200"] and
                indicator_values["VWAP"] > market_data["close"].item(-1)
            )
        }
        self.generated_strategies.update(new_strategies)
        logging.info("Generated new strategies")

    def select_best_strategy(self, market_data):
        """
        Seleziona la migliore strategia di trading basata sui dati
        di mercato attuali e sulla conoscenza compressa.
        Questo metodo analizza gli indicatori tecnici
        calcolati dai dati di mercato
        e le condizioni delle strategie generate
        per determinare quale strategia applicare.
        Se non viene soddisfatta alcuna condizione,
        viene selezionata una strategia predefinita.
        """
        self.indicators = TradingIndicators(data=market_data)
        self.all_indicators = self.get_all_indicators()
        self.detect_market_anomalies(market_data)
        self.generate_new_strategies(market_data)
        indicator_values = (
            {name: func() for name,
            func in self.all_indicators.items()}
        )

        strategy_conditions = {
            **self.generated_strategies,
            "scalping": (
                indicator_values["RSI"] < 30 and
                indicator_values["MACD"] > indicator_values["MACD_Signal"] and
                indicator_values["ADX"] > 25 and
                self.compressed_knowledge.mean() > 0.6,
            ),
            "mean_reversion": (
                indicator_values["RSI"] > 70 and
                indicator_values["MACD"] < indicator_values["MACD_Signal"] and
                indicator_values["BB_Upper"] >
                market_data["close"].item(-1) and
                self.compressed_knowledge.mean() > 0.5,
            ),
            "trend_following": (
                indicator_values["EMA_50"] > indicator_values["EMA_200"] and
                indicator_values["VWAP"] > market_data["close"].item(-1) and
                self.compressed_knowledge.mean() > 0.4,
            ),
            "swing": (
                indicator_values["STOCH_K"] < 20 and
                indicator_values["STOCH_D"] < 20 and
                self.compressed_knowledge.mean() > 0.3,
            ),
            "momentum": (
                indicator_values["momentum"] > 100 and
                indicator_values["ADX"] > 20 and
                self.compressed_knowledge.mean() > 0.2,
            ),
            "breakout": (
                indicator_values["Donchian_Upper"] <
                market_data["high"].item(-1) and
                indicator_values["volatility"] > 1.5 and
                self.compressed_knowledge.mean() > 0.7,
            ),
            "ai_generated": self.compressed_knowledge.mean() > 0.75,
        }

        for strategy, condition in strategy_conditions.items():
            if condition:
                logging.info("Best strategy selected: %s", strategy)
                return strategy, self.compressed_knowledge.mean()

        logging.info("Default strategy selected")
        return "default_strategy", self.compressed_knowledge.mean()

    def fuse_top_strategies(self, top_n=5):
        """
        Combina le migliori strategie generate
        in una singola strategia "super".
        Questo metodo seleziona le migliori `top_n`
        strategie basandosi sui loro
        punteggi di performance, e le fonde in una nuova strategia chiamata
        "super_strategy". Ogni indicatore è mediato
        tra le strategie selezionate.
        """
        sorted_strategies = sorted(
            self.generated_strategies.items(),
            key=lambda x: x[1].get('performance_score', 0.5),
            reverse=True
        )[:top_n]

        super_strategy = {}
        for _, strategy in sorted_strategies:
            for indicator, condition in strategy.items():
                super_strategy[indicator] = super_strategy.get(
                    indicator, 0
                ) + condition

        for indicator in super_strategy:
            super_strategy[indicator] /= top_n

        self.generated_strategies["super_strategy"] = super_strategy
        logging.info("Fused top strategies into super strategy")

    def exploit_market_anomalies(self, market_data):
        """
        Identifica e sfrutta le anomalie di mercato per
        generare strategie dedicate.
        Questo metodo analizza i dati di mercato per rilevare anomalie, come
        spread elevati o alta latenza. Per ogni anomalia rilevata,
        una strategia
        specifica viene generata e aggiunta al dizionario
        self.generated_strategies.
        """
        anomalies = []
        if market_data["spread"].item(-1) > market_data["spread"].mean() * 5:
            anomalies.append("Buco di Liquidità")
        if market_data["latency"].item(-1) > 200:
            anomalies.append("Lag nei Dati")
        for anomaly in anomalies:
            name = f"exploit_{anomaly.lower().replace(' ', '_')}"
            self.generated_strategies[name] = {"anomaly_detected": True}
            logging.warning("Exploiting market anomaly: %s", anomaly)

    async def continuous_self_improvement(self, interval_seconds=1800):
        """
        Miglioramento continuo delle strategie:
        - Dati simulati SOLO in fase iniziale e controllata.
        
        """
        from data_loader import load_preset_assets
        from ai_memory_compact import AIMemoryCompact as CompactMemory
        from ai_memory_sync import AIMemory as SyncMemory

        compact_path = Path("D:/trading_data/ai_memory_compact.npz")
        sync_path = Path("D:/trading_data/ai_memory_sync.npz")

        # Creazione file vuoti se non esistono
        if not compact_path.exists():
            CompactMemory().save(compact_path)
        if not sync_path.exists():
            SyncMemory().save(sync_path)

        # Loop asincrono continuo
        while True:
            try:
                compact = CompactMemory.load(compact_path)
                sync = SyncMemory.load(sync_path)
                if self.latest_market_data is not None:
                # Strategie e anomalie
                    self.generate_new_strategies(self.latest_market_data)
                    self.fuse_top_strategies()
                    self.exploit_market_anomalies(self.latest_market_data)
                    assets = load_preset_assets()
                    knowledge_updated = False

                    # 🔹 PRIORITÀ: Memoria sincronizzata (per simboli con almeno 20 trade)
                    for symbol in assets:
                        if sync.get_total_trades(symbol) >= 20:
                            try:
                                profit = sync.get_last_profit(symbol)
                                win_rate = sync.get_win_rate(symbol)
                                drawdown = sync.get_drawdown(symbol)
                                volatility = self.volatility_predictor.predict(symbol)

                                if all(0 <= x <= 1 for x in [win_rate, drawdown]) and abs(profit) < 10:
                                    self.update_knowledge(profit, win_rate, drawdown, volatility)
                                    logging.info(f"📈 Conoscenza aggiornata [SYNC] da {symbol}")
                                    knowledge_updated = True
                                    break
                                else:
                                    logging.warning(f"⚠️ Valori anomali ignorati [SYNC] {symbol}")
                            except Exception as e:
                                logging.warning(f"⚠️ Errore memoria sync [{symbol}]: {e}")

                    # 🔹 ALTERNATIVA: Memoria compatta
                    if not knowledge_updated and compact.total_trades >= 60:
                        try:
                            profit = compact.get_last_profit()
                            win_rate = compact.get_win_rate()
                            drawdown = compact.get_drawdown()
                            volatility = compact.sharpe_ratio

                            if all(0 <= x <= 1 for x in [win_rate, drawdown]) and abs(profit) < 10:
                                self.update_knowledge(profit, win_rate, drawdown, volatility)
                                logging.info("📈 Conoscenza aggiornata [COMPACT]")
                                knowledge_updated = True
                            else:
                                logging.warning("⚠️ Valori anomali ignorati [COMPACT]")
                        except Exception as e:
                            logging.warning(f"⚠️ Errore memoria compact: {e}")

                    # 🔹 FALLBACK: Solo se nessun dato reale disponibile e forzato
                    if not knowledge_updated and self.force_simulated_knowledge:
                        if compact.total_trades < 20 and all(sync.get_total_trades(sym) < 20 for sym in assets):
                            simulated_profit = np.random.uniform(-0.2, 0.5)
                            simulated_win_rate = np.random.uniform(0.52, 0.6)
                            simulated_drawdown = np.random.uniform(0.02, 0.1)
                            simulated_volatility = np.random.uniform(0.01, 0.03)
                            self.update_knowledge(simulated_profit, simulated_win_rate, simulated_drawdown, simulated_volatility)
                            logging.info("🤖 Conoscenza aggiornata con dati simulati (fase iniziale)")
                        else:
                            logging.info("⛔ Dati simulati disabilitati: già presenti dati reali sufficienti")
                    elif not knowledge_updated:
                        logging.info("⚠️ Nessuna conoscenza aggiornata: dati insufficienti")
                else:
                    logging.info("🕒 Attesa: dati di mercato non disponibili")
            except Exception as general_error:
                logging.error(f"💥 Errore nel ciclo di apprendimento asincrono: {general_error}")
            await asyncio.sleep(interval_seconds)

    def update_strategies(self, strategy_name, result):
        """
        Aggiorna internamente lo stato delle strategie dopo un trade.
        """
        logging.info(
            "📈 Strategia aggiornata: %s con profit: %s",
            strategy_name,
            result,
        )
        self.compressed_knowledge = np.clip(
            self.compressed_knowledge + (result / 1000), 0, 1
        )
        self.save_compressed_knowledge()


# ✅ Test rapido e avvio
if __name__ == "__main__":

    sg = StrategyGenerator()
    sg.save_compressed_knowledge()
    threading.Thread(
        target=sg.continuous_self_improvement, daemon=True
    ).start()
    logger.info(
        "📊 Conoscenza strategica caricata: %s", sg.compressed_knowledge.mean()
    )