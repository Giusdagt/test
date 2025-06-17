"""
data_loader.py
Gestione avanzata, ultra-intelligente e dinamica degli asset.
Supporta preset, caricamento dinamico e trading reale da config.json.
"""

import json
import os
import logging
import requests
from universal_symbol_manager import get_provider_symbol, get_internal_symbol

print("data_loader.py caricato ✅")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"
PRESET_ASSETS_FILE = "preset_assets.json"
AUTO_MAPPING_FILE = "auto_symbol_mapping.json"

USE_PRESET_ASSETS = True  # True preset_assets.json, False dinamico illimitato
ENABLE_AUTO_MAPPING = False  # ✅ False = Usa solo i simboli manuali
ENABLE_SYMBOL_STANDARDIZATION = False # O True se vuoi attivarla

SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD",
    "NOK", "SEK", "DKK", "MXN", "ZAR", "TRY", "CNY", "RUB", "PLN", "HUF",
    "INR", "IDR", "VND", "THB", "KRW", "PHP"
]


def load_json_file(json_file, default=None):
    """Carica un file JSON e restituisce i dati."""
    if not os.path.exists(json_file):
        logging.warning("⚠️ File %s non trovato, ne creo uno nuovo", json_file)
        return {} if default is None else default
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, json_file):
    """Salva i dati in un file JSON."""
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logging.info("✅ Dati salvati in %s", json_file)


def load_config():
    """Carica il file config.json per trading reale."""
    return load_json_file(CONFIG_FILE)


def load_market_data_apis():
    """Carica configurazioni delle API di mercato."""
    return load_json_file(MARKET_API_FILE)


def load_preset_assets():
    """Carica gli asset predefiniti per il trading da preset_assets.json."""
    with open("preset_assets.json", "r") as f:
        all_assets = json.load(f)
    enabled_assets = {}
    for category, info in all_assets.items():
        if info.get("enabled", False):
            enabled_assets[category] = info
    return enabled_assets


def load_auto_symbol_mapping():
    """Carica la mappatura automatica simboli."""
    return load_json_file(AUTO_MAPPING_FILE, default={})


def save_auto_symbol_mapping(mapping):
    """Salva la mappatura automatica simboli."""
    save_json_file(mapping, AUTO_MAPPING_FILE)


def standardize_symbol(symbol, mapping, provider="default"):
    """
    Standardizza i simboli in base al mapping e al provider.
    Rispetta il flag ENABLE_AUTO_MAPPING.
    """
    if not ENABLE_AUTO_MAPPING:
        return mapping.get(symbol, symbol)

    standardized = mapping.get(symbol)
    if not standardized:
        adapted_symbol = adapt_symbol_for_provider(symbol, provider)
        logging.warning(
            "⚠️ Simbolo sconosciuto per il provider %s: %s. Usato come: %s",
            provider, symbol, adapted_symbol
        )
        mapping[symbol] = adapted_symbol
        save_auto_symbol_mapping(mapping)
        return adapted_symbol
    return standardized


def adapt_symbol_for_provider(symbol, provider):
    """
    Adatta il simbolo al formato richiesto dal provider.
    """
    if provider == "yfinance":
        # Adatta il simbolo per yfinance (esempio: BTCUSD → BTC-USD)
        return f"{symbol[:3].upper()}-{symbol[3:].upper()}"
    # Aggiungi altre logiche per altri provider, se necessario
    return symbol


def adapt_symbol_for_broker(symbol, broker):
    """Adatta il simbolo al formato richiesto dal broker."""
    return get_provider_symbol(symbol, broker)


def categorize_tradable_assets(assets, mapping):
    """Categorizza automaticamente gli asset forniti."""
    try:
        for category, asset_list in assets.items():
            TRADABLE_ASSETS[category] = [
                get_internal_symbol(
                    adapt_symbol_for_broker(
                        standardize_symbol(asset, mapping), "MetaTrader5"
                    ),
                    "MetaTrader5"
                )
                for asset in asset_list
            ]
        logging.info("✅ Asset organizzati e normalizzati con successo.")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        logging.error("❌ Errore categorizzazione asset: %s", e)


# Ultra-avanzato: priorità no-api, fallback API solo se necessario
def dynamic_assets_loading(mapping):
    """Caricamento dinamico intelligente degli asset."""
    market_data_apis = load_market_data_apis()
    assets = {"crypto": [], "forex": [], "indices": [], "commodities": []}

    no_api_sources = market_data_apis.get("data_sources", {}).get("no_api", {})
    for source_name, base_url in no_api_sources.items():
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            for item in data:
                api_symbol = get_provider_symbol(item["symbol"], source_name)
                symbol = get_internal_symbol(api_symbol, source_name)
                standardized_symbol = standardize_symbol(
                    symbol, mapping, provider=source_name
                )
                asset_type = exchange_asset_type(standardized_symbol)
                if asset_type:
                    assets[asset_type].append(standardized_symbol)
            logging.info("✅ Dati no-api da %s caricati.", source_name)
        except requests.RequestException as e:
            logging.warning("⚠️ Fonte no-api '%s' fallita: %s", source_name, e)

    categorize_tradable_assets(assets, mapping)


def exchange_asset_type(symbol):
    """Determina tipo di asset basato su simbolo."""
    if symbol.endswith(tuple(SUPPORTED_CURRENCIES)):
        return "forex"
    if symbol.startswith(("BTC", "ETH", "BNB")):
        return "crypto"
    if symbol.startswith(("XAU", "XAG", "WTI")):
        return "commodities"
    if symbol in ("US30", "NAS100", "SPX"):
        return "indices"
    return None

TRADABLE_ASSETS = load_preset_assets()