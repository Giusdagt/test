"""
Memoria compatta per apprendimento automatico 
per strategia AI
self.memory_compact_path = MODEL_DIR / "ai_memory_compact.npz"
"""
import numpy as np
from pathlib import Path

print("ai_memory_compact.py caricato ✅")

class AIMemoryCompact:
    def __init__(self):
        self.total_trades = 0
        self.total_profit = 0.0
        self.embedding_sum = None
        self.embedding_count = 0
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.last_embedding = None
        self.last_confidence = None
        self.equity_curve = []

    def update(self, profit: float, embedding: np.ndarray, confidence: float = None):
        self.total_trades += 1
        self.total_profit += profit

        # Embedding
        emb = np.array(embedding, dtype=np.float32)
        if self.embedding_sum is None:
            self.embedding_sum = emb
        else:
            self.embedding_sum += emb
        self.embedding_count += 1
        self.last_embedding = emb

        # Confidence
        if confidence is not None:
            self.confidence_sum += confidence
            self.confidence_count += 1
            self.last_confidence = confidence

        # Equity curve
        if not self.equity_curve:
            self.equity_curve = [self.total_profit]
        else:
            self.equity_curve.append(self.total_profit)

    @property
    def avg_embedding(self):
        if self.embedding_count == 0:
            return None
        return self.embedding_sum / self.embedding_count

    @property
    def avg_confidence(self):
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count

    @property
    def sharpe_ratio(self):
        returns = np.diff(self.equity_curve)
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns))

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        np.savez_compressed(
            filepath,
            total_trades=self.total_trades,
            total_profit=self.total_profit,
            embedding_sum=self.embedding_sum,
            embedding_count=self.embedding_count,
            confidence_sum=self.confidence_sum,
            confidence_count=self.confidence_count,
            last_embedding=self.last_embedding,
            last_confidence=self.last_confidence,
            equity_curve=np.array(self.equity_curve, dtype=np.float32)
        )

    @staticmethod
    def load(filepath: str | Path):
        filepath = Path(filepath)
        if not filepath.exists():
            return AIMemoryCompact()

        data = np.load(filepath, allow_pickle=True)
        mem = AIMemoryCompact()
        mem.total_trades = int(data["total_trades"])
        mem.total_profit = float(data["total_profit"])
        mem.embedding_sum = data["embedding_sum"]
        mem.embedding_count = int(data["embedding_count"])
        mem.confidence_sum = float(data["confidence_sum"])
        mem.confidence_count = int(data["confidence_count"])
        mem.last_embedding = data["last_embedding"]
        mem.last_confidence = data["last_confidence"]
        mem.equity_curve = data["equity_curve"].tolist()
        return mem
    
    def get_last_profit(self) -> float:
        """Restituisce l'ultimo profitto registrato (variazione tra ultimi due punti equity)."""
        if len(self.equity_curve) < 2:
            return 0.0
        return float(self.equity_curve[-1] - self.equity_curve[-2])

    def get_win_rate(self) -> float:
        """Stima il win rate basandosi sull'equity curve (profit positivo = vincente)."""
        if len(self.equity_curve) < 2:
            return 0.0
        wins = 0
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i] > self.equity_curve[i - 1]:
                wins += 1
        return wins / (len(self.equity_curve) - 1)

    def get_drawdown(self) -> float:
        """
        Calcola il drawdown massimo come percentuale relativa al profitto cumulato.
        """
        if len(self.equity_curve) == 0:
            return 0.0
        equity = np.array(self.equity_curve, dtype=np.float32)
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.maximum(peak, 1e-6)
        return float(np.max(drawdowns))

