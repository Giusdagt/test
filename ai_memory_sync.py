"""
ai_memory_sync.py
Memoria AI sincronizzata a lungo termine
Salva embedding, performance e confidenza per ogni trade.
Compatibile con AI, DRL, LSTM, realtime e recovery post-crash.
self.memory_path = Path("D:/trading_data/ai_memory_sync.npz")
"""
import numpy as np

print("ai_memory_sync.py caricato ✅")

class AIMemory:
    def __init__(self):
        self.n_trades = 0
        self.total_profit = 0.0
        self.win_trades = 0
        self.loss_trades = 0
        self.embedding_sum = None
        self.embedding_count = 0
        self.max_drawdown = 0.0
        self.equity_curve = []
        self.last_embedding = None

        # Confidence
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.confidence_max = None
        self.confidence_min = None
        self.last_confidence = None

        # Profit statistici
        self.profit_sum = 0.0
        self.profit_squared_sum = 0.0
        self.max_profit = None
        self.min_profit = None

        # Ultima azione IA
        self.last_decision = None
        self.last_result = None

    def update(self, profit, embedding, confidence=None, decision=None, result=None):
        self.n_trades += 1
        self.total_profit += profit
        self.profit_sum += profit
        self.profit_squared_sum += profit**2
        if self.max_profit is None or profit > self.max_profit:
            self.max_profit = profit
        if self.min_profit is None or profit < self.min_profit:
            self.min_profit = profit

        if profit > 0:
            self.win_trades += 1
        else:
            self.loss_trades += 1

        # Embedding
        if self.embedding_sum is None:
            self.embedding_sum = np.array(embedding)
        else:
            self.embedding_sum += np.array(embedding)
        self.embedding_count += 1
        self.last_embedding = np.array(embedding)

        # Equity curve e drawdown
        if not self.equity_curve:
            self.equity_curve = [self.total_profit]
        else:
            self.equity_curve.append(self.total_profit)
        peak = max(self.equity_curve)
        drawdown = peak - self.total_profit
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Confidence
        if confidence is not None:
            self.confidence_sum += confidence
            self.confidence_count += 1
            if self.confidence_max is None or confidence > self.confidence_max:
                self.confidence_max = confidence
            if self.confidence_min is None or confidence < self.confidence_min:
                self.confidence_min = confidence
            self.last_confidence = confidence

        # Ultima azione IA
        if decision is not None:
            self.last_decision = decision
        if result is not None:
            self.last_result = result

    @property
    def winrate(self):
        return self.win_trades / self.n_trades if self.n_trades else 0

    @property
    def lossrate(self):
        return self.loss_trades / self.n_trades if self.n_trades else 0

    @property
    def avg_profit(self):
        return self.total_profit / self.n_trades if self.n_trades else 0

    @property
    def profit_std(self):
        if self.n_trades < 2:
            return 0.0
        mean = self.profit_sum / self.n_trades
        variance = (self.profit_squared_sum / self.n_trades) - mean**2
        return np.sqrt(variance) if variance > 0 else 0.0

    @property
    def avg_embedding(self):
        if self.embedding_count == 0:
            return None
        return self.embedding_sum / self.embedding_count

    @property
    def sharpe_ratio(self):
        returns = np.diff(self.equity_curve)
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns))

    @property
    def avg_confidence(self):
        return self.confidence_sum / self.confidence_count if self.confidence_count else 0.0

    @property
    def max_confidence(self):
        return self.confidence_max if self.confidence_max is not None else 0.0

    @property
    def min_confidence(self):
        return self.confidence_min if self.confidence_min is not None else 0.0

    @property
    def last_drawdown(self):
        if not self.equity_curve:
            return 0.0
        peak = max(self.equity_curve)
        return peak - self.equity_curve[-1]

    def save(self, path):
        np.savez_compressed(
            path,
            n_trades=self.n_trades,
            total_profit=self.total_profit,
            win_trades=self.win_trades,
            loss_trades=self.loss_trades,
            embedding_sum=self.embedding_sum,
            embedding_count=self.embedding_count,
            max_drawdown=self.max_drawdown,
            equity_curve=np.array(self.equity_curve, dtype=np.float32),
            last_embedding=self.last_embedding,
            confidence_sum=self.confidence_sum,
            confidence_count=self.confidence_count,
            confidence_max=self.confidence_max,
            confidence_min=self.confidence_min,
            last_confidence=self.last_confidence,
            profit_sum=self.profit_sum,
            profit_squared_sum=self.profit_squared_sum,
            max_profit=self.max_profit,
            min_profit=self.min_profit,
            last_decision=self.last_decision,
            last_result=self.last_result
        )

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=True)
        mem = AIMemory()
        mem.n_trades = int(data["n_trades"])
        mem.total_profit = float(data["total_profit"])
        mem.win_trades = int(data["win_trades"])
        mem.loss_trades = int(data["loss_trades"])
        mem.embedding_sum = data["embedding_sum"]
        mem.embedding_count = int(data["embedding_count"])
        mem.max_drawdown = float(data["max_drawdown"])
        mem.equity_curve = data["equity_curve"].tolist()
        mem.last_embedding = data["last_embedding"]
        mem.confidence_sum = float(data.get("confidence_sum", 0.0))
        mem.confidence_count = int(data.get("confidence_count", 0))
        mem.confidence_max = data.get("confidence_max", None)
        mem.confidence_min = data.get("confidence_min", None)
        mem.last_confidence = data.get("last_confidence", None)
        mem.profit_sum = float(data.get("profit_sum", 0.0))
        mem.profit_squared_sum = float(data.get("profit_squared_sum", 0.0))
        mem.max_profit = data.get("max_profit", None)
        mem.min_profit = data.get("min_profit", None)
        mem.last_decision = data.get("last_decision", None)
        mem.last_result = data.get("last_result", None)
        return mem