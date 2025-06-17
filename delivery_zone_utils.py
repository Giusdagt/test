"""delivery_zone_utils.py"""

import polars as pl
import logging

def add_delivery_zone_columns(df, symbol):
    try:
        # Controllo sicuro: valore "close" sull'ultima riga
        if "close" not in df.columns or df["close"].is_empty():
            logging.error(f"❌ Colonna close mancante o vuota per {symbol}")
            return df

        last_close = df["close"][-1] if df.height > 0 else None
        if last_close is None or not isinstance(last_close, (int, float)):
            logging.error(f"❌ Valore close non valido per {symbol}: {last_close}")
            return df

        from delivery_zone_manager import DeliveryZoneManager
        dzm = DeliveryZoneManager()
        delivery_price_buy = dzm.get_delivery_zone(symbol, action="buy")
        delivery_price_sell = dzm.get_delivery_zone(symbol, action="sell")

        delivery_price_buy = float(delivery_price_buy) if delivery_price_buy is not None else 0.0
        delivery_price_sell = float(delivery_price_sell) if delivery_price_sell is not None else 0.0

        df = df.with_columns([
            pl.Series("delivery_zone_buy", [delivery_price_buy] * df.height),
            pl.Series("delivery_zone_sell", [delivery_price_sell] * df.height),
        ])


    except Exception as e:
        logging.error(f"Errore calcolo delivery zone per {symbol}: {e}")

    return df

