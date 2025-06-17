import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from data_handler import get_normalized_market_data

print("market_event_manager.py caricato ✅")

class MarketEventManager:
    def __init__(self):
        self.last_checked = None
        self.important_events = []  # Cache eventi macro
        self.patience_level = 0.7  # 0-1, maggiore = più selettivo

    def fetch_macro_events(self):
        """Simula il recupero eventi macroeconomici da un'API gratuita."""
        today = datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%d')
        if self.last_checked == today:
            return self.important_events

        try:
            # Simuliamo con eventi statici per il test
            self.important_events = [
                {"time": "14:30", "event": "NFP", "impact": "High"},
                {"time": "20:00", "event": "FOMC", "impact": "High"}
            ]
            self.last_checked = today
        except Exception as e:
            logging.warning(f"⚠️ Errore recupero eventi macro: {e}")
        return self.important_events

    def is_macro_event_near(self):
        now_utc = datetime.now(tz=datetime.timezone.utc)
        events = self.fetch_macro_events()
        for event in events:
            event_time = datetime.strptime(f"{now_utc.date()} {event['time']}", "%Y-%m-%d %H:%M")
            if now_utc <= event_time <= now_utc + timedelta(minutes=30):
                logging.info(f"📅 Evento macro imminente: {event['event']} - Impatto: {event['impact']}")
                return True
        return False

    def calculate_cvd(self, df):
        """Calcola il Cumulative Volume Delta semplificato."""
        buys = (df['close'] > df['open']).sum()
        sells = (df['close'] < df['open']).sum()
        cvd = buys - sells
        return cvd

    def detect_fvg_zones(self, df):
        """Rileva Fair Value Gaps (FVG) semplificati."""
        gaps = []
        for i in range(2, len(df) - 1):
            if df['low'][i] > df['high'][i - 2]:
                gaps.append({"type": "Bullish FVG", "price": df['low'][i]})
            if df['high'][i] < df['low'][i - 2]:
                gaps.append({"type": "Bearish FVG", "price": df['high'][i]})
        return gaps

    def trade_filter(self, signal_confidence, volatility):
        """Applica un filtro di qualità trade basato su pazienza, volatilità e confidenza AI."""
        required_confidence = self.patience_level * (1 + volatility)
        if signal_confidence >= required_confidence:
            logging.info("✅ Trade consentito | Confidenza: %.2f | Richiesta: %.2f", signal_confidence, required_confidence)
            return True
        logging.info("❌ Trade scartato | Confidenza: %.2f | Richiesta: %.2f", signal_confidence, required_confidence)
        return False

    def is_liquidity_zone_active(self, symbol):
        """Controlla se il prezzo attuale si trova vicino a zone di liquidità."""
        df = get_normalized_market_data(symbol)
        if df is None or df.height == 0:
            return False
        ilq_zones = df.filter(df["ILQ_Zone"] == 1)
        if ilq_zones.is_empty():
            return False
        current_price = df['close'][-1]
        support = ilq_zones['low'].min()
        resistance = ilq_zones['high'].max()
        return support * 0.99 <= current_price <= resistance * 1.01

    def should_trade(self, symbol, signal_confidence, volatility):
        """Valutazione finale se procedere o meno al trade."""
        if self.is_macro_event_near():
            logging.info("⛔ Evento macroeconomico imminente. Evito il trade su %s.", symbol)
            return False
        if not self.is_liquidity_zone_active(symbol):
            logging.info("📉 Nessuna zona di liquidità attiva su %s.", symbol)
            return False
        return self.trade_filter(signal_confidence, volatility)
