"""
ai_utils.py
"""
import logging
from data_handler import (
    get_normalized_market_data, get_available_assets
)
from ai_model import AIModel, fetch_account_balances

print("ai_utils.py caricato ✅")

logging.basicConfig(level=logging.INFO)


def prepare_ai_model():
    """
    Prepara il modello AI recuperando i bilanci dell'account,
    gli asset disponibili e i relativi dati di mercato normalizzati.
    Restituisce:
    tuple: Una tupla contenente l'istanza di AIModel e
    il dizionario dei dati di mercato.
    """
    balances = fetch_account_balances()
    all_assets = get_available_assets()
    market_data = {
        symbol: get_normalized_market_data(symbol)
        for symbol in all_assets
    }
    return AIModel(market_data, balances), market_data
