"""
optimizer_core.py

Questo modulo contiene la classe OptimizerCore responsabile dell'ottimizzazione
delle strategie di trading,
delle anomalie di mercato, della conoscenza compressa e della memoria AI.
Include anche funzionalità per
l'ottimizzazione dei trade e delle performance.

Classi:
- OptimizerCore: Gestisce le varie ottimizzazioni per migliorare le
performance del sistema di trading.

Funzioni:
- optimize_strategies: Ottimizza le strategie di trading.
- optimize_anomalies: Ottimizza le anomalie di mercato.
- optimize_knowledge: Comprime e ottimizza la conoscenza.
- optimize_ai_memory: Ottimizza la memoria AI.
- optimize_trades: Ottimizza i trade.
- optimize_performance: Ottimizza le performance.
- evaluate_evolution: Valuta l'evoluzione strategica.
- clean_ram: Pulisce la RAM.
- run_full_optimization: Esegue l'ottimizzazione completa.
"""

import logging
import gc
from pathlib import Path
import polars as pl
import numpy as np

print("optimizer_core.py caricato ✅")

MODEL_DIR = Path("D:/trading_data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_FILE = MODEL_DIR / "strategies_compressed.parquet"
ANOMALY_FILE = MODEL_DIR / "market_anomalies.parquet"
KNOWLEDGE_FILE = MODEL_DIR / "compressed_knowledge.parquet"
AI_MEMORY_FILE = MODEL_DIR / "ai_memory.parquet"
TRADE_FILE = MODEL_DIR / "trades.parquet"
PERFORMANCE_FILE = MODEL_DIR / "performance.parquet"


class OptimizerCore:
    """
    Gestisce l'ottimizzazione delle strategie di trading,
    anomalie di mercato e altri dati.
    """
    def __init__(self, strategy_generator=None, ai_model=None):
        self.sg = strategy_generator
        self.ai = ai_model

    def optimize_strategies(self):
        """
        Ottimizza le strategie di trading e le salva su file.
        """
        if not self.sg:
            logging.warning("⚠️ Nessun strategy_generator collegato.")
            return

        top_strategies = dict(list(self.sg.generated_strategies.items())[:5])
        self.sg.generated_strategies.clear()
        self.sg.generated_strategies.update(top_strategies)

        df = pl.DataFrame({"strategies": [str(self.sg.generated_strategies)]})
        df.write_parquet(STRATEGY_FILE, compression="zstd")
        logging.info("✅ Strategie ottimizzate e salvate.")

    def optimize_anomalies(self):
        """
        Ottimizza le anomalie di mercato mantenendo solo le ultime 50.
        """
        if not self.sg:
            return
        limited_anomalies = (
            self.sg.market_anomalies[-50:]  # Mantiene solo le ultime 50
        )
        self.sg.market_anomalies = limited_anomalies
        df = pl.DataFrame({"anomalies": [limited_anomalies]})
        df.write_parquet(
            ANOMALY_FILE, compression="zstd"
        )
        logging.info("✅ Anomalie salvate e ottimizzate.")

    def optimize_knowledge(self):
        """
        Comprime e ottimizza la conoscenza accumulata nel sistema.
        """
        if not self.sg:
            return

        ck = self.sg.compressed_knowledge
        while len(ck) > 25:
            ck = np.mean(ck.reshape(-1, 2), axis=1)
        self.sg.compressed_knowledge = ck

        df = pl.DataFrame({"knowledge": [ck.tobytes()]})
        df.write_parquet(KNOWLEDGE_FILE, compression="zstd")
        logging.info("🧠 Conoscenza compressa e salvata.")

    def optimize_ai_memory(self):
        """
        Ottimizza la memoria AI consolidando le ultime 10 entry.
        """
        if not self.ai:
            return

        if AI_MEMORY_FILE.exists():
            mem = pl.read_parquet(AI_MEMORY_FILE)["memory"].to_numpy()
            if len(mem) > 10:
                mem = mem[-10:]  # Mantieni solo le ultime 10 entry
            mean_mem = np.mean(mem)
            df = pl.DataFrame({"memory": [mean_mem]})
            df.write_parquet(
                AI_MEMORY_FILE, compression="zstd"
            )
            logging.info("🧠 Memoria AI consolidata.")

    def optimize_trades(self):
        """
        Ottimizza i trade ordinandoli per profitto e mantenendo i migliori 100.
        """
        if not TRADE_FILE.exists():
            return
        df = pl.read_parquet(TRADE_FILE)
        df = df.sort("profit", descending=True).head(100)  # top 100 trade
        df.write_parquet(TRADE_FILE, compression="zstd")
        logging.info("📊 Trade compressi e ottimizzati.")

    def optimize_performance(self):
        """
        Ottimizza le performance mantenendo le ultime 100 righe di dati.
        """
        if not PERFORMANCE_FILE.exists():
            return
        df = pl.read_parquet(PERFORMANCE_FILE)
        df = df.tail(100)  # Ultime 100 righe di performance
        df.write_parquet(
            PERFORMANCE_FILE, compression="zstd"
        )
        logging.info("📈 Performance ottimizzate.")

    def evaluate_evolution(
        self, profit, win_rate, drawdown, volatility, strategy_strength
    ):
        """
        Valuta l'evoluzione strategica basata su vari parametri di performance.
        """
        score = (
            (profit * 0.5) +
            (win_rate * 0.3) -
            (drawdown * 0.1) -
            (volatility * 0.1) +
            (strategy_strength * 0.5)
        )
        evolution_score = max(0, min(100, score * 10))
        logging.info("📊 Evoluzione Strategica: %.2f / 100", evolution_score)
        return evolution_score

    def clean_ram(self):
        """
        Libera la memoria RAM eseguendo il garbage collection.
        """
        gc.collect()
        logging.info("🧹 RAM pulita.")

    def run_full_optimization(self):
        """
        Esegue l'ottimizzazione completa delle strategie, anomalie, conoscenza,
        memoria AI, trade e performance.
        """
        self.optimize_strategies()
        self.optimize_anomalies()
        self.optimize_knowledge()
        self.optimize_ai_memory()
        self.optimize_trades()
        self.optimize_performance()
        self.clean_ram()
        logging.info("✅ Ottimizzazione completa eseguita.")
