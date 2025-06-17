"""
Definizioni delle colonne richieste per il DataFrame.
"""
import polars as pl
from datetime import datetime, timezone
import MetaTrader5 as mt5

print("column_definitions.py caricato ✅")

required_columns = [
    'symbol', 'volatility', 'last_updated', "timeframe",
    'historical_prices', 'timestamp', "close", "open", "high",
    "ask", "bid", "spread", "ask_volume", "bid_volume", "market_depth",
    "low", "volume", "market_cap", "market_cap_rank",
    "fully_diluted_valuation", "circulating_supply",
    "total_supply", "max_supply", "price_change_24h",
    "price_change_percentage_24h", "market_cap_change_24h",
    "market_cap_change_percentage_24h", "ath",
    "ath_change_percentage", "ath_date", "atl",
    "atl_change_percentage", "atl_date", "roi",
    "order_book", "trade_volume",
    "price_change_7d", "price_change_1h", "price_change_4h",
    "latency", "signal", "stoch",
    "spread_volatility", "spread_ma"
]

def add_roi(df):
    if "close" in df.columns and "open" in df.columns:
        df = df.with_columns([
            (
                ((pl.col("close") - pl.col("open")) / (pl.col("open") + 1e-9)) * 100
            ).alias("roi")
        ])
    return df

def add_price_change_1h(df):
    if "close" in df.columns:
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(1)).alias("price_change_1h")
        ])
    return df

def add_price_change_4h(df):
    if "close" in df.columns:
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(4)).alias("price_change_4h")
        ])
    return df

def add_volatility(df, window=5):
    if "close" in df.columns:
        df = df.with_columns([
            pl.col("close").rolling_std(window_size=window).alias("volatility")
        ])
    return df

def add_price_change_7d(df):
    if "close" in df.columns:
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(7)).alias("price_change_7d")
        ])
    return df

def calculate_stochastic_pl(df, k_period=14, d_period=3):
    """
    Calcola Stochastic Oscillator (%K) e Signal Line (%D) su un DataFrame Polars.
    Restituisce un nuovo DataFrame con colonne 'stoch' e 'signal'.
    """
    stoch = (
        100 * (
            df['close'] - df['low'].rolling_min(k_period)
        ) / (
            df['high'].rolling_max(k_period) - df['low'].rolling_min(k_period) + 1e-9
        )
    )
    signal = stoch.rolling_mean(d_period)
    return df.with_columns([
        stoch.alias("stoch"),
        signal.alias("signal")
    ])

def add_spread_features(df, window=5):
    if "spread" in df.columns:
        df = df.with_columns([
            pl.col("spread").rolling_mean(window).alias("spread_ma"),
            pl.col("spread").rolling_std(window).alias("spread_volatility"),
        ])
    return df

def add_order_book_features(df):
    if "order_book" in df.columns:
        df = df.with_columns([
            pl.col("order_book").map_elements(
                lambda book: float(sum(level['volume'] for level in book)) if book else 0.0,
                return_dtype=pl.Float64, skip_nulls=False
            ).alias("market_depth"),
            pl.col("order_book").map_elements(
                lambda book: float(sum(level['volume'] for level in book if level.get('type') == 1)) if book else 0.0,
                return_dtype=pl.Float64, skip_nulls=False
            ).alias("ask_volume"),
            pl.col("order_book").map_elements(
                lambda book: float(sum(level['volume'] for level in book if level.get('type') == -1)) if book else 0.0,
                return_dtype=pl.Float64, skip_nulls=False
            ).alias("bid_volume"),
        ])
    return df

def add_market_cap(df):
    if "close" in df.columns and "circulating_supply" in df.columns:
        df = df.with_columns([
            (pl.col("close") * pl.col("circulating_supply")).alias("market_cap")
        ])
    return df

def add_bid_ask_price(df):
    if "bid" in df.columns:
        df = df.with_columns([pl.col("bid").alias("bid_price")])
    if "ask" in df.columns:
        df = df.with_columns([pl.col("ask").alias("ask_price")])
    return df

def add_fully_diluted_valuation(df):
    # FDV = close * max_supply
    if "close" in df.columns and "max_supply" in df.columns:
        df = df.with_columns([
            (pl.col("close") * pl.col("max_supply")).alias("fully_diluted_valuation")
        ])
    return df

def add_market_cap_rank(df):
    # Rank basato su market_cap (1 = più grande)
    if "market_cap" in df.columns:
        df = df.with_columns([
            (-pl.col("market_cap")).rank("dense").alias("market_cap_rank")
        ])
    return df

def add_ath_atl(df):
    # All Time High/Low e relative date
    if "close" in df.columns and "timestamp" in df.columns:
        ath = df["close"].max()
        atl = df["close"].min()

        ath_rows = df.filter(pl.col("close") == ath)
        atl_rows = df.filter(pl.col("close") == atl)

        ath_date = ath_rows["timestamp"].min() if ath_rows.height > 0 and ath_rows["timestamp"].null_count() < ath_rows.height else None
        atl_date = atl_rows["timestamp"].min() if atl_rows.height > 0 and atl_rows["timestamp"].null_count() < atl_rows.height else None

        df = df.with_columns([
            pl.lit(ath).alias("ath"),
            pl.lit(atl).alias("atl"),
            pl.lit(ath_date).alias("ath_date"),
            pl.lit(atl_date).alias("atl_date"),
            ((pl.col("close") - ath) / ath * 100).alias("ath_change_percentage"),
            ((pl.col("close") - atl) / atl * 100).alias("atl_change_percentage")
        ])

        # Correzione valori infiniti o NaN
        df = df.with_columns([
            pl.when(pl.col("atl_change_percentage").is_infinite() | pl.col("atl_change_percentage").is_nan())
              .then(0.0)
              .otherwise(pl.col("atl_change_percentage"))
              .alias("atl_change_percentage"),

            pl.when(pl.col("ath_change_percentage").is_infinite() | pl.col("ath_change_percentage").is_nan())
              .then(0.0)
              .otherwise(pl.col("ath_change_percentage"))
              .alias("ath_change_percentage"),
        ])
    return df


def add_market_cap_change_24h(df):
    # Variazione market cap 24h e percentuale
    if "market_cap" in df.columns:
        df = df.with_columns([
            (pl.col("market_cap") - pl.col("market_cap").shift(24)).alias("market_cap_change_24h"),
            ((pl.col("market_cap") - pl.col("market_cap").shift(24)) / (pl.col("market_cap").shift(24) + 1e-9) * 100).alias("market_cap_change_percentage_24h"),
        ])
    return df

def add_price_change_percentage_24h(df):
    # Percentuale variazione prezzo 24h
    if "close" in df.columns:
        df = df.with_columns([
            ((pl.col("close") - pl.col("close").shift(24)) / (pl.col("close").shift(24) + 1e-9) * 100).alias("price_change_percentage_24h")
        ])
    return df

def add_last_updated(df):
    """
    Aggiunge una colonna 'last_updated' con timestamp UTC corrente.
    """
    timestamp_utc = datetime.now(timezone.utc).timestamp()
    return df.with_columns([
        pl.lit(timestamp_utc).alias("last_updated")
    ])


def add_realtime_orderbook_columns(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """
    Aggiunge dinamicamente colonne realtime ask, bid, spread, volumi e profondità.
    Solo dove disponibili, senza forzare i valori.
    """
    ask_list = []
    bid_list = []
    spread_list = []
    ask_vol_list = []
    bid_vol_list = []
    depth_list = []

    for _ in df.iter_rows(named=True):
        tick = mt5.symbol_info_tick(symbol)
        book = mt5.market_book_get(symbol)

        ask = tick.ask if tick and tick.ask > 0 else 0.0
        bid = tick.bid if tick and tick.bid > 0 else 0.0
        spread = round(ask - bid, 8) if ask > 0 and bid > 0 else 0.0

        ask_volume = 0.0
        bid_volume = 0.0
        depth = 0
        if book:
            for entry in book:
                if entry.type == mt5.BOOK_TYPE_SELL:
                    ask_volume += entry.volume
                elif entry.type == mt5.BOOK_TYPE_BUY:
                    bid_volume += entry.volume
            depth = len(book)

        ask_list.append(ask)
        bid_list.append(bid)
        spread_list.append(spread)
        ask_vol_list.append(ask_volume)
        bid_vol_list.append(bid_volume)
        depth_list.append(depth)

    return df.with_columns([
        pl.Series("ask", ask_list),
        pl.Series("bid", bid_list),
        pl.Series("spread", spread_list),
        pl.Series("ask_volume", ask_vol_list),
        pl.Series("bid_volume", bid_vol_list),
        pl.Series("market_depth", depth_list),
    ])


def apply_all_column_features(df):
    """
    Applica tutte le funzioni di arricchimento colonne in sequenza.
    """
    df = add_price_change_1h(df)
    df = add_price_change_4h(df)
    df = add_price_change_7d(df)
    df = add_volatility(df)
    df = calculate_stochastic_pl(df)
    df = add_spread_features(df)
    df = add_order_book_features(df)
    df = add_roi(df)
    df = add_market_cap(df)
    df = add_fully_diluted_valuation(df)
    df = add_market_cap_rank(df)
    df = add_ath_atl(df)
    df = add_market_cap_change_24h(df)
    df = add_price_change_percentage_24h(df)
    df = add_bid_ask_price(df)
    df = add_last_updated(df)
    return df