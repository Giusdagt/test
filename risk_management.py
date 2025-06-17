"""
risk_management.py
Modulo definitivo per la gestione intelligente del rischio e del capitale.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e trading adattivo su
più account. Supporta trading multi-strategia, gestione avanzata del rischio
e allocazione ottimale del capitale. Configurazione dinamica tramite
config.json per automazione totale.
"""

import logging
import numpy as np
from volatility_tools import VolatilityPredictor
from data_loader import (
    load_config,
    load_auto_symbol_mapping,
    USE_PRESET_ASSETS,
    load_preset_assets
)
from data_handler import get_normalized_market_data
from zone_utils import get_ilq_zone_for_symbol

print("risk_management.py caricato ✅")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Carica configurazione globale
config = load_config()


def get_tradable_assets():
    """Restituisce gli asset per il trading in base alle impostazioni."""
    auto_mapping = load_auto_symbol_mapping()
    return sum(load_preset_assets().values(), []) if USE_PRESET_ASSETS \
        else list(auto_mapping.values())


class RiskManagement:
    """Gestisce rischio, allocazione del capitale e trailing stop"""
    def __init__(self, max_drawdown=None):
        """Starta il sistema di gestione del rischio dalla configurazione"""
        settings = config["risk_management"]
        self.risk_settings = {
            "max_drawdown": (
                max_drawdown
                if max_drawdown is not None
                else settings["max_drawdown"]
            ),
            "trailing_stop_pct": settings["trailing_stop_pct"],
            "risk_per_trade": settings["risk_per_trade"],
            "max_exposure": settings["max_exposure"]
        }
        self.balance_info = {'min': float('inf'), 'max': 0}
        self.kill_switch_activated = False
        self.volatility_predictor = VolatilityPredictor()
        self.recovery_counter = 0
        self.adaptive_enabled = (
            config["risk_management"].get("adaptive_risk_management", True)
        )
        self.take_profit_pct = (
            config["risk_management"].get("take_profit_pct", 0.05)
        )
        self.max_trades = config["risk_management"].get("max_trades", 5)

    def calculate_dynamic_risk(self, market_data):
        """
        Calcola il rischio dinamico in base ai dati di mercato.
        """
        if market_data is None or market_data.is_empty() or "volatility" not in market_data:
            return self.risk_settings["risk_per_trade"]

        volatility = market_data["volatility"]
        momentum = market_data.get("momentum", 1.0)

        if volatility < 0.3 and momentum > 1:
            return min(0.03, self.risk_settings["risk_per_trade"] * 1.5)
        if volatility > 0.5:
            return max(0.005, self.risk_settings["risk_per_trade"] * 0.5)
        return self.risk_settings["risk_per_trade"]

    def check_drawdown(self, current_balance):
        """
        Monitora il drawdown e attiva il kill switch se necessario.
        """
        if self.balance_info['max'] == 0:
            self.balance_info['max'] = current_balance
            self.balance_info['min'] = current_balance

        if current_balance < self.balance_info['min']:
            self.balance_info['min'] = current_balance

        if current_balance > self.balance_info['max']:
            self.balance_info['max'] = current_balance

        drawdown = (
            (self.balance_info['max'] - current_balance)
            / self.balance_info['max']
        ) if self.balance_info['max'] > 0 else 0

        if drawdown > self.risk_settings["max_drawdown"]:
            self.kill_switch_activated = True
            logging.error(
                "🛑 KILL SWITCH ATTIVATO: drawdown %.2f%%", drawdown * 100
            )

    def adaptive_stop_loss(self, entry_price, symbol):
        """ Calcola Stop Loss (SL), Trailing Stop (TS) e Take Profit (TP)
        usando la ILQ Zone se disponibile, altrimenti fallback su volatilità.
        """
        ilq = get_ilq_zone_for_symbol(symbol)
        if ilq and isinstance(ilq, dict):
            sl = ilq["support"] if entry_price > ilq["support"] else entry_price - 0.002
            tp = ilq["resistance"] if entry_price < ilq["resistance"] else entry_price + 0.004
            ts = sl + (tp - sl) * 0.4  # Trailing stop al 40% tra SL e TP
            return sl, ts, tp
        try:
            market_data = get_normalized_market_data(symbol)
            if market_data is None or market_data.is_empty():
                raise ValueError("Dati non disponibili")
            if "volatility" not in market_data.columns:
                raise ValueError("Colonna 'volatility' mancante")
            
            volatility = market_data["volatility"].to_numpy()[-1]
            stop_loss = entry_price * (1 - (volatility * 1.5))
            trailing_stop = entry_price * (1 - (volatility * 0.8))
            take_profit = entry_price * (1 + self.take_profit_pct)
            return stop_loss, trailing_stop, take_profit
        except Exception as e:
            logging.warning("⚠️ Fallback SL/TP per %s: %s", symbol, str(e))
            default_sl = entry_price * 0.98
            default_ts = entry_price * 0.99
            default_tp = entry_price * 1.02
            return default_sl, default_ts, default_tp

    def adjust_risk(self, symbol):
        """Adatta trailing stop e il capitale usando dati normalizzati."""
        market_data = get_normalized_market_data(symbol)
        required_keys = ["volume", "price_change", "rsi", "bollinger_width"]
        if market_data is None or market_data.is_empty() or any(
            key not in market_data for key in required_keys
        ):
            logging.warning(
                "⚠️ Dati incompleti X %s, resto invariato", symbol
            )
            return  # Non modifica il rischio se i dati non sono completi

        future_volatility = self.volatility_predictor.predict_volatility(
            np.array([
                [
                    market_data["volume"],
                    market_data["price_change"],
                    market_data["rsi"],
                    market_data["bollinger_width"]
                ]
            ])
        )
        atr = future_volatility[0] * 100  # Previsione volatilità futura
        if atr > 15:
            self.risk_settings["trailing_stop_pct"] = 0.15
            self.risk_settings["risk_per_trade"] = 0.01
        elif atr > 10:
            self.risk_settings["trailing_stop_pct"] = 0.1
            self.risk_settings["risk_per_trade"] = 0.015
        else:
            self.risk_settings["trailing_stop_pct"] = 0.05
            self.risk_settings["risk_per_trade"] = 0.02

    def calculate_position_size(self, balance, symbol):
        """Dimensione ottimale della posizione in base al saldo e ai dati"""
        market_data = get_normalized_market_data(symbol)

        if self.adaptive_enabled:
            self.adjust_risk(symbol)

        if balance <= 0:
            logging.warning(
                "⚠️ Saldo non valido (%s) per %s, imposta 0.",
                balance,
                symbol
            )
            return 0

        self.check_drawdown(balance)
        if self.kill_switch_activated:
            logging.warning("🛑 Kill switch attivo. Nessuna posizione.")
            return 0

        if market_data is None or market_data.is_empty() or "momentum" not in market_data:
            logging.warning(
                "⚠️ Momentum non disponibile per %s, uso valore base.",
                symbol
            )
            momentum_factor = 1  # Default
        else:
            momentum_factor = 1 + float(market_data["momentum"].item(-1))

        base_position_size = balance * self.risk_settings["risk_per_trade"]
        adjusted_position_size = base_position_size * momentum_factor
        max_allowed = balance * self.risk_settings["max_exposure"]
        if (
            market_data is not None
            and "ILQ_Zone" in market_data
            and "volatility" in market_data
            and market_data["ILQ_Zone"].item(-1) == 1
            and float(market_data["volatility"].item(-1)) < 0.3
        ):
            adjusted_position_size *= 1.5

        return float(min(float(adjusted_position_size), float(max_allowed)))
