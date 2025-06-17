import polars as pl
import numpy as np
from data_handler import get_normalized_market_data

print("smc_manager.py caricato ✅")

class SMCManager:
    """
    Gestisce la logica completa Smart Money Concept (SMC):
    - Rilevamento zone di liquidità (ILQ Zones)
    - Calcolo Delivery Zones
    - Identificazione BOS (Break of Structure) e CHoCH (Change of Character)
    - Calcolo Stop Loss e Take Profit intelligenti
    """

    def __init__(self, volume_factor=2.0, spread_thresh=0.02):
        self.volume_factor = volume_factor
        self.spread_thresh = spread_thresh

    def get_ilq_zone(self, symbol, lookback=200):
        df = get_normalized_market_data(symbol)
        if df is None or len(df) < lookback:
            return None

        df = self._add_ilq_zone(df)
        ilq_df = df.filter(pl.col("ILQ_Zone") == 1)

        if ilq_df.is_empty():
            return None

        support = ilq_df["low"].min()
        resistance = ilq_df["high"].max()

        return {
            "support": float(support),
            "resistance": float(resistance)
        }

    def get_delivery_zone(self, symbol: str, action: str, lookback: int = 200):
        if not isinstance(symbol, str):
            raise ValueError(f"Simbolo non valido: {symbol}")
        df = get_normalized_market_data(symbol)
        if df is None or df.height < lookback:
            return None  # Dati insufficienti

        ilq_df = df.filter(pl.col("ILQ_Zone") == 1)
        if ilq_df.is_empty():
            return None  # Nessuna ILQ rilevata

        if action == "buy":
            delivery_zone = ilq_df["high"].max()
        else:  # Sell
            delivery_zone = ilq_df["low"].min()

        return float(delivery_zone)


    def calculate_delivery_zone(self, symbol, extension_factor=1.5):
        ilq_zone = self.get_ilq_zone(symbol)
        if not ilq_zone:
            return None

        return {
            "buy_tp": ilq_zone['resistance'] * extension_factor,
            "sell_tp": ilq_zone['support'] / extension_factor
        }

    def detect_bos(self, df):
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        bos = [0]

        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                bos.append(1)
            elif lows[i] < lows[i-1]:
                bos.append(-1)
            else:
                bos.append(0)

        df = df.with_columns(pl.Series("BOS", bos))
        return df

    def detect_choch(self, df):
        closes = df["close"].to_numpy()
        choch = [0]

        for i in range(2, len(closes)):
            if closes[i-1] > closes[i-2] and closes[i] < closes[i-1]:
                choch.append(1)  # Cambio da rialzo a ribasso
            elif closes[i-1] < closes[i-2] and closes[i] > closes[i-1]:
                choch.append(-1)  # Cambio da ribasso a rialzo
            else:
                choch.append(0)

        choch = [0] + choch  # Allinea lunghezza
        df = df.with_columns(pl.Series("CHoCH", choch))
        return df

    def get_stop_loss_tp(self, symbol, action):
        ilq_zone = self.get_ilq_zone(symbol)
        delivery_zone = self.calculate_delivery_zone(symbol)

        if not ilq_zone or not delivery_zone:
            return None, None

        if action == "buy":
            sl = ilq_zone['support'] * 0.99  # 1% sotto supporto ILQ
            tp = delivery_zone['buy_tp']
        else:
            sl = ilq_zone['resistance'] * 1.01  # 1% sopra resistenza ILQ
            tp = delivery_zone['sell_tp']

        return sl, tp

    def _add_ilq_zone(self, df):
        avg_volume = df["volume"].mean()
        ilq = (
            (df["spread"] < self.spread_thresh) &
            (df["volume"] > avg_volume * self.volume_factor)
        ).cast(pl.Int8)
        return df.with_columns(ilq.alias("ILQ_Zone"))

    def enforce_entry_zone(self, symbol, current_price, action):
        ilq_zone = self.get_ilq_zone(symbol)
        if not ilq_zone:
            return False

        if action == "buy":
            return ilq_zone['support'] <= current_price <= ilq_zone['resistance']
        else:
            return ilq_zone['support'] <= current_price <= ilq_zone['resistance']

    def apply_smc_signals(self, df):
        df = self._add_ilq_zone(df)
        df = self.detect_bos(df)
        df = self.detect_choch(df)
        return df
