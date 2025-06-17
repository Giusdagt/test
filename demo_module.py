"""
Modulo demo_module.py
Questo modulo fornisce funzionalità per simulare operazioni di trading
senza inviarle realmente. È utile per test, training automatico
o quando le credenziali di trading non sono disponibili.
Funzionalità principali:
- Simulazione di trade con profitti casuali.
- Salvataggio dei dati delle operazioni simulate in un file Parquet.
"""
import logging
import random
from pathlib import Path
import datetime
import polars as pl

print("demo_module.py caricato ✅")

MODEL_DIR = (
    Path("/mnt/usb_trading_data/models")
    if Path("/mnt/usb_trading_data").exists()
    else Path("D:/trading_data/models")
)
TRADE_FILE = MODEL_DIR / "demo_trades.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def demo_trade(symbol, market_data):
    if market_data is None or market_data.height == 0:
        logging.warning("⚠️ Nessun dato per il demo trade su %s.", symbol)
        return

    fake_profit = round(random.uniform(-5, 10), 2)
    new_row = pl.DataFrame({
        "symbol": pl.Series([symbol], dtype=pl.Utf8),
        "profit": pl.Series([fake_profit], dtype=pl.Float64),
        "timestamp": pl.Series([datetime.datetime.now()], dtype=pl.Datetime)
    })

    df = (
        pl.read_parquet(TRADE_FILE)
        if TRADE_FILE.exists()
        else pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.Utf8),
            "profit": pl.Series([], dtype=pl.Float64),
            "timestamp": pl.Series([], dtype=pl.Datetime)
        })
    )
    df = pl.concat([df, new_row], how="vertical")
    df.write_parquet(TRADE_FILE, compression="zstd")
