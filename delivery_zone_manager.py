"""
Modulo completo per la gestione delle Delivery Zone.
Calcola in modo intelligente le zone di presa profitto in base a volumi, ILQ zone,
reazioni storiche e previsione dei prezzi.
"""
import traceback
import numpy as np
import polars as pl
from data_handler import get_normalized_market_data
from smart_features import add_ilq_zone
from price_prediction import PricePredictionModel


print("delivery_zone_manager.py caricato ✅")


class DeliveryZoneManager:
    def __init__(self):
        self.price_predictor = PricePredictionModel()

    def calculate_delivery_zone(self, symbol, action, lookback=300, volume_factor=2.0):
        df = get_normalized_market_data(symbol)
        if df is None or df.height < lookback:
            return None  # Dati insufficienti
        
        if "embedding" not in df.columns and not any(col.startswith("embedding_") for col in df.columns):
            print(f"⚠️ Colonna 'embedding' non trovata per {symbol}")
            return None
        numeric_cast = ["volume", "high", "low", "close", "open"]
        for col in numeric_cast:
            if col in df.columns:
                try:
                    df = df.with_columns(df[col].cast(pl.Float64).alias(col))
                except Exception as e:
                    print(f"⚠️ Impossibile castare {col} in float64 per {symbol}: {e}")
        # Applica ILQ Zone
        df = add_ilq_zone(df, volume_factor=volume_factor)
        ilq_df = df.filter(pl.col("ILQ_Zone") == 1)

        if ilq_df.is_empty():
            return None  # Nessuna zona liquida rilevata

        # Analisi dei volumi per la zona di Delivery
        high_volume_zones = ilq_df.filter(pl.col("volume") > ilq_df["volume"].mean() * volume_factor)
        if high_volume_zones.is_empty():
            return None

        # Calcola livelli di prezzo target
        if action == "buy":
            delivery_level = float(high_volume_zones["high"].max())
        elif action == "sell":
            delivery_level = float(high_volume_zones["low"].min())
        else:
            print(f"⚠️ Azione non valida: {action}")
            return None

        # Raffinamento con la previsione del prezzo
        numeric_cols = [col for col, dtype in df.schema.items() if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        market_data_array = df.select(numeric_cols).to_numpy().flatten()
        full_state = np.clip(market_data_array, -1, 1)
        try:
            full_state = full_state.reshape(1, -1)
        except Exception as e:
            print(f"❌ Errore nel reshape dello stato per {symbol}: {e}")
            return None
        prediction = self.price_predictor.predict_price(symbol=symbol, full_state=full_state)
        if prediction is None or not isinstance(prediction, (list, np.ndarray)) or len(prediction) == 0:
            print(f"⚠️ Predict_price ha restituito None o vuoto per {symbol}")
            return None
        predicted_price = prediction[0]


        if not np.isfinite(delivery_level) or not np.isfinite(predicted_price):
            print(f"⚠️ Valori non validi per fusione Delivery Zone per {symbol}: "
                  f"delivery_level={delivery_level}, predicted_price={predicted_price}")
            return None
        final_delivery_zone = delivery_level * 0.7 + predicted_price * 0.3
        return round(float(final_delivery_zone), 5)

    def get_delivery_zone(self, symbol, action):
        if symbol is None:
            print("⚠️ Nessuna Delivery Zone trovata: simbolo non specificato (None)")
            traceback.print_stack(limit=3)
            return None
        zone = self.calculate_delivery_zone(symbol, action)
        if zone is None:
            print(f"⚠️ Nessuna Delivery Zone trovata per {symbol} [{action}]")
        return zone


if __name__ == "__main__":
    dzm = DeliveryZoneManager()
    test_zone = dzm.get_delivery_zone("EURUSD", "buy")
    print(f"Delivery Zone suggerita: {test_zone}")
