"""
Modulo per il calcolo avanzato degli indicatori di trading.
"""

import logging
import numpy as np
import polars as pl
import requests
import talib

print("indicators.py caricato ✅")

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 URL SENTIMENT_API_URLAPI per Sentiment Analysis
ENABLE_SENTIMENT_API = False  # Imposta a True per attivare la chiamata all'API
SENTIMENT_API_URL = "https://your-sentiment-api.com/analyze"


def align_indicator_length(indicator, data_len):
    """
    Allinea la lunghezza di un array/serie di indicatori a quella dei dati.
    Se più corto, aggiunge np.nan in testa.
    """
    indicator = np.asarray(indicator)
    if len(indicator) < data_len:
        pad = [np.nan] * (data_len - len(indicator))
        indicator = np.concatenate([pad, indicator])
    elif len(indicator) > data_len:
        indicator = indicator[-data_len:]
    return indicator

def calculate_scalping_indicators(data):
    """Indicatori ottimizzati per Scalping."""
    if data.shape[0] < 60:
        logging.warning("⚠️ Dataset troppo corto per calcolare tutti gli indicatori. Righe disponibili: %d", data.shape[0])
        return data

    data = data.with_columns([
        pl.col("close").rolling_mean(window_size=14).alias("SMA_14"),
        pl.col("close").ewm_mean(span=14).alias("EMA_14"),
        pl.col("close").rolling_mean(window_size=50).alias("SMA_50"),
        pl.col("close").ewm_mean(span=50).alias("EMA_50"),
        pl.col("close").rolling_mean(window_size=100).alias("SMA_100"),
        pl.col("close").ewm_mean(span=200).alias("EMA_200"),
        ((pl.col("close") / pl.col("close").shift(10)) - 1).alias("ROC_10"),
    ])

    rsi = align_indicator_length(talib.RSI(data["close"], timeperiod=14), len(data))
    stoch_k, stoch_d = talib.STOCH(data["high"], data["low"], data["close"])
    stoch_k = align_indicator_length(stoch_k, len(data))
    stoch_d = align_indicator_length(stoch_d, len(data))

    macd, macdsignal, macdhist = talib.MACD(
        data["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    macd = align_indicator_length(macd, len(data))
    macdsignal = align_indicator_length(macdsignal, len(data))
    macdhist = align_indicator_length(macdhist, len(data))
    atr = align_indicator_length(talib.ATR(data["high"], data["low"], data["close"], timeperiod=14), len(data))
    cci = align_indicator_length(talib.CCI(data["high"], data["low"], data["close"], timeperiod=20), len(data))
    mfi = align_indicator_length(talib.MFI(data["high"], data["low"], data["close"], data["volume"], timeperiod=14), len(data))
    # Calcolo VWAP in stile Polars
    data = data.with_columns([
        (pl.col("close") * pl.col("volume")).alias("pv")
    ])
    data = data.with_columns([
        pl.col("pv").cum_sum().alias("cum_pv"),
        pl.col("volume").cum_sum().alias("cum_vol")
    ])
    data = data.with_columns([
        (pl.col("cum_pv") / pl.col("cum_vol")).alias("VWAP")
    ])
    data = data.drop(["pv", "cum_pv", "cum_vol"])

    data = data.with_columns(
        pl.Series("RSI", rsi),
        pl.Series("STOCH_K", stoch_k),
        pl.Series("STOCH_D", stoch_d),
        pl.Series("MACD", macd),
        pl.Series("MACD_Signal", macdsignal),
        pl.Series("MACD_Hist", macdhist),
        pl.Series("SuperTrend_Upper", data["close"] + (2 * atr)),
        pl.Series("SuperTrend_Lower", data["close"] - (2 * atr)),
        pl.Series("ATR", atr),
        pl.Series("CCI", cci),
        pl.Series("MFI", mfi),
    )
    return data


def calculate_historical_indicators(data):
    """Indicatori ottimizzati per analisi storiche."""
    if data.shape[0] < 60:
        logging.warning("⚠️ Dataset troppo corto per calcolare tutti gli indicatori. Righe disponibili: %d", data.shape[0])
        return data

    upper_band, middle_band, lower_band = talib.BBANDS(
        data["close"], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    upper_band = align_indicator_length(upper_band, len(data))
    middle_band = align_indicator_length(middle_band, len(data))
    lower_band = align_indicator_length(lower_band, len(data))
    adx = align_indicator_length(talib.ADX(data["high"], data["low"], data["close"], timeperiod=14), len(data))
    obv = align_indicator_length(talib.OBV(data["close"], data["volume"]), len(data))

    data = data.with_columns(
        pl.Series("BB_Upper", upper_band),
        pl.Series("BB_Middle", middle_band),
        pl.Series("BB_Lower", lower_band),
        pl.Series("ADX", adx),
        pl.Series("OBV", obv),
    )

    # Ichimoku Cloud - rolling min/max
    data = data.with_columns([
        pl.col("high").rolling_max(window_size=9).alias("nine_high"),
        pl.col("low").rolling_min(window_size=9).alias("nine_low"),
        pl.col("high").rolling_max(window_size=26).alias("twenty_six_high"),
        pl.col("low").rolling_min(window_size=26).alias("twenty_six_low"),
        pl.col("high").rolling_max(window_size=52).alias("fifty_two_high"),
        pl.col("low").rolling_min(window_size=52).alias("fifty_two_low"),
    ])

    # Ichimoku Cloud - linee principali
    data = data.with_columns([
        ((pl.col("nine_high") + pl.col("nine_low")) / 2).alias("Ichimoku_Tenkan"),
        ((pl.col("twenty_six_high") + pl.col("twenty_six_low")) / 2).alias("Ichimoku_Kijun"),
    ])

    # Ichimoku Cloud - Senkou Span (richiedono le linee già calcolate)
    data = data.with_columns([
        (((pl.col("Ichimoku_Tenkan") + pl.col("Ichimoku_Kijun")) / 2).shift(26)).alias("Senkou_Span_A"),
        (((pl.col("fifty_two_high") + pl.col("fifty_two_low")) / 2).shift(26)).alias("Senkou_Span_B"),
    ])

    donchian_period = 20  # puoi cambiare il periodo se vuoi
    data = data.with_columns([
        pl.col("high").rolling_max(window_size=donchian_period).alias("Donchian_Upper"),
        pl.col("low").rolling_min(window_size=donchian_period).alias("Donchian_Lower"),
    ])

    return data


def calculate_intraday_indicators(data):
    """Indicatori ottimizzati per il trading intra-day."""
    data = calculate_scalping_indicators(data)
    data = calculate_historical_indicators(data)
    return data


def fetch_sentiment_data():
    """Recupera dati di sentiment analysis."""
    if not ENABLE_SENTIMENT_API:
        logging.info("Sentiment API disattivata.")
        return None
    try:
        response = requests.get(SENTIMENT_API_URL)
        response.raise_for_status()
        sentiment_data = response.json()
        return sentiment_data["sentiment_score"]
    except requests.exceptions.RequestException as e:
        logging.error("❌ Errore API Sentiment Analysis: %s", e)
        return None


def get_indicators_list():
    """Restituisce una lista di tutti gli indicatori disponibili."""
    return [
        "RSI", "STOCH_K", "STOCH_D", "BB_Upper", "BB_Middle", "BB_Lower",
        "MACD", "MACD_Signal", "MACD_Hist", "EMA_50", "EMA_200", "SMA_100",
        "ADX", "OBV", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Senkou_Span_A",
        "Senkou_Span_B", "SuperTrend_Upper", "SuperTrend_Lower",
        "Donchian_Lower", "Donchian_Upper", "ATR", "CCI", "MFI", "VWAP"
    ]


class TradingIndicators:
    """Classe per calcolare e gestire gli indicatori di trading."""

    def __init__(self, data):
        self.data = data
        self.indicators = {}

    def calculate_all_indicators(self):
        """Calcola tutti gli indicatori disponibili."""
        return calculate_intraday_indicators(self.data)

    def fetch_sentiment(self):
        """Ottiene dati di sentiment analysis per il mercato."""
        return fetch_sentiment_data()

    def list_available_indicators(self):
        """Restituisce la lista di tutti gli indicatori disponibili."""
        return get_indicators_list()
