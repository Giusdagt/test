# zone_utils.py

import polars as pl
import math
from data_handler import get_normalized_market_data
from smart_features import add_ilq_zone

def get_ilq_zone_for_symbol(symbol, lookback=200, volume_factor=2.0):
    """
    Calcola i livelli precisi di supporto e resistenza della zona ILQ.
    """
    df = get_normalized_market_data(symbol)
    if df is None or df.is_empty() or len(df) < lookback:
        return None

    try:
        df = add_ilq_zone(df, volume_factor=volume_factor)
        if "ILQ_Zone" not in df.columns:
            return None

        ilq_df = df.filter(pl.col("ILQ_Zone") == 1)
        if ilq_df.is_empty():
            return None

        support = ilq_df["low"].min()
        resistance = ilq_df["high"].max()

        if not math.isfinite(support) or not math.isfinite(resistance):
            return None

        return {
            "support": float(support),
            "resistance": float(resistance)
        }

    except Exception as e:
        print(f"⚠️ Errore calcolo ILQ Zone per {symbol}: {e}")
        return None
