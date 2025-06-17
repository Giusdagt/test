"""
ai_features.py
Definizione dei set di feature AI
usati per SCALPING, SWING e MACRO trading.
Il sistema può scegliere automaticamente
il set corretto in base al tipo di strategia.
"""

print("ai_features.py caricato ✅")

AI_FEATURES_SCALPING = [
    "open", "high", "low", "close", "volume",
    "spread", "roi", "bid_volume", "ask_volume", "market_depth",
    "price_change_percentage_24h", "market_cap_change_percentage_24h",
    "bid", "ask", "tick_volume", "signal", "stoch",
    "market_cap", "market_cap_rank",
    "volatility", "spread_volatility", "latency", "spread_ma",
    "trade_volume", "order_book",
    "RSI", "STOCH_K", "STOCH_D", "MACD", "MACD_Signal", "MACD_Hist",
    "EMA_50", "EMA_200", "SMA_100", "SMA_14", "SMA_50", "VWAP",
    "ATR", "CCI", "MFI", "SuperTrend_Upper", "SuperTrend_Lower", "ROC_10",
    "cvd", "bullish_fvg", "bearish_fvg", "liquidity_zone_active",
    "body_size", "upper_wick", "lower_wick", "body_ratio", "is_indecision",
    "engulfing_bullish", "engulfing_bearish", "inside_bar",
    "fakeout_up", "fakeout_down", "volatility_squeeze", "micro_pattern_hft",
    "momentum", "local_volatility", "avg_volume_3",
    "signal_score", "weighted_signal_score",
    "BOS", "CHoCH",
    "delivery_zone_buy",
    "delivery_zone_sell"
]

AI_FEATURES_SWING = [
    "open", "high", "low", "close", "volume",
    "market_cap", "spread", "roi", "signal", "stoch",
    "market_cap_rank", "price_change_1h", "price_change_4h", "price_change_7d",
    "price_change_percentage_24h", "market_cap_change_percentage_24h",
    "ath", "ath_change_percentage", "ath_date",
    "atl", "atl_change_percentage", "atl_date",
    "circulating_supply", "total_supply", "max_supply",
    "volatility", "trade_volume", "spread_ma",
    "RSI", "BB_Upper", "BB_Middle", "BB_Lower", "ADX", "OBV",
    "Ichimoku_Tenkan", "Ichimoku_Kijun", "Senkou_Span_A", "Senkou_Span_B",
    "Donchian_Lower", "Donchian_Upper", "ATR", "CCI", "MFI",
    "cvd", "bullish_fvg", "bearish_fvg", "liquidity_zone_active",
    "body_size", "upper_wick", "lower_wick", "body_ratio",
    "engulfing_bullish", "engulfing_bearish", "inside_bar",
    "ILQ_Zone", "fakeout_up", "fakeout_down", "volatility_squeeze",
    "momentum", "signal_score", "weighted_signal_score",
    "BOS", "CHoCH",
    "delivery_zone_buy", "delivery_zone_sell"
]


AI_FEATURES_MACRO = [
    "open", "close", "volume", "roi",
    "market_cap_rank", "fully_diluted_valuation", "price_change_7d",
    "market_cap", "price_change_percentage_24h", "market_cap_change_percentage_24h",
    "ath", "ath_change_percentage", "ath_date",
    "atl", "atl_change_percentage", "atl_date",
    "circulating_supply", "total_supply", "max_supply",
    "last_updated", "historical_prices", "trade_volume", "order_book",
    "RSI", "ADX", "OBV", "ATR", "VWAP", "MACD",
    "Ichimoku_Tenkan", "Ichimoku_Kijun",
    "macro_event_near", "macro_event_impact", "liquidity_zone_active",
    "ILQ_Zone", "signal_score", "weighted_signal_score",
    "BOS", "CHoCH",
    "delivery_zone_buy", "delivery_zone_sell"
]

# Dizionario centrale per selezione automatica
FEATURE_SET_MAP = {
    "scalping": AI_FEATURES_SCALPING,
    "swing": AI_FEATURES_SWING,
    "macro": AI_FEATURES_MACRO
}


def get_features_by_strategy_type(strategy_type: str):
    """
    Restituisce l'elenco di feature
    corretto in base al tipo di strategia.
    Args:
    strategy_type (str):
    Tipo di strategia ("scalping", "swing", "macro")
    Returns:
    list[str]: Elenco di colonne da usare per IA/DRL
    """
    features = FEATURE_SET_MAP.get(str(strategy_type).lower(), AI_FEATURES_SWING)
    if isinstance(features, list):
        return features
    elif isinstance(features, str):
        return [features]
    return AI_FEATURES_SWING

def get_ai_features_from_df(df, strategy_type: str):
    """
    Filtra un DataFrame mantenendo
    solo le colonne AI in base al tipo di strategia.
    Args:
    df (polars.DataFrame or pandas.DataFrame): DataFrame completo
    strategy_type (str): "scalping", "swing" o "macro"
    Returns:
    DataFrame filtrato con solo le colonne AI disponibili
    """
    features = get_features_by_strategy_type(strategy_type)
    available = [col for col in features if col in df.columns]
    if hasattr(df, "select"):
        return df.select(available)
    else:
        return df[available]
