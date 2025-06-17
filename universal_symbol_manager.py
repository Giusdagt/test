import json
import os
import logging

print("universal_symbol_manager.py caricato ✅")

AUTO_MAPPING_FILE = "auto_symbol_mapping.json"

PROVIDER_MAPPINGS = {
    "yahoo": {
        "crypto_format": lambda base, quote: f"{base}-{quote}",
        "forex_format": lambda base, quote: f"{base}{quote}=X",
        "index_format": lambda symbol: symbol,
        "commodity_format": lambda symbol: symbol
    },
    "alphavantage": {
        "crypto_format": lambda base, quote: f"{base}",
        "forex_format": lambda base, quote: f"{base}",
        "index_format": lambda symbol: symbol,
        "commodity_format": lambda symbol: symbol
    },
    "stooq": {
        "crypto_format": lambda base, quote: f"{base.lower()}{quote.lower()}.us",
        "forex_format": lambda base, quote: f"{base.lower()}{quote.lower()}.fx",
        "index_format": lambda symbol: symbol.lower(),
        "commodity_format": lambda symbol: symbol.lower()
    },
    "mt5": {
        "crypto_format": lambda base, quote: f"{base}{quote}",
        "forex_format": lambda base, quote: f"{base}{quote}",
        "index_format": lambda symbol: symbol,
        "commodity_format": lambda symbol: symbol
    },
    "default": {
        "crypto_format": lambda base, quote: f"{base}{quote}",
        "forex_format": lambda base, quote: f"{base}{quote}",
        "index_format": lambda symbol: symbol,
        "commodity_format": lambda symbol: symbol
    }
}

def load_auto_symbol_mapping():
    if not os.path.exists(AUTO_MAPPING_FILE):
        return {}
    with open(AUTO_MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_auto_symbol_mapping(mapping):
    with open(AUTO_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    logging.info("✅ Mappatura simboli aggiornata in %s", AUTO_MAPPING_FILE)

def is_crypto(symbol):
    return "USD" in symbol and len(symbol) == 6  # Es: BTCUSD

def is_forex(symbol):
    return len(symbol) == 6 and symbol.isalpha()

def is_index(symbol):
    return any(keyword in symbol for keyword in ["SPX", "NAS", "DJI", "DAX", "FTSE", "CAC", "NIKKEI"])

def is_commodity(symbol):
    return symbol.startswith(("XAU", "XAG", "OIL", "GOLD", "SILVER", "WTI", "BRENT"))

def get_provider_symbol(internal_symbol, provider="default"):
    """
    Traduce un simbolo interno nel formato richiesto dal provider.
    Aggiorna automaticamente la mappatura se il simbolo non è presente.
    """
    provider = provider.lower()
    mapping = load_auto_symbol_mapping()

    # Se il simbolo è già mappato, restituiscilo
    if internal_symbol in mapping:
        return mapping[internal_symbol]

    # Determina il tipo di asset
    base = internal_symbol[:3].upper()
    quote = internal_symbol[3:].upper()

    try:
        if is_crypto(internal_symbol):
            provider_symbol = PROVIDER_MAPPINGS.get(provider, PROVIDER_MAPPINGS["default"])["crypto_format"](base, quote)
        elif is_forex(internal_symbol):
            provider_symbol = PROVIDER_MAPPINGS.get(provider, PROVIDER_MAPPINGS["default"])["forex_format"](base, quote)
        elif is_index(internal_symbol):
            provider_symbol = PROVIDER_MAPPINGS.get(provider, PROVIDER_MAPPINGS["default"])["index_format"](internal_symbol)
        elif is_commodity(internal_symbol):
            provider_symbol = PROVIDER_MAPPINGS.get(provider, PROVIDER_MAPPINGS["default"])["commodity_format"](internal_symbol)
        else:
            logging.warning("⚠️ Simbolo sconosciuto: %s. Nessuna traduzione applicata.", internal_symbol)
            provider_symbol = internal_symbol  # Fallback
    except Exception as e:
        logging.error("❌ Errore durante la traduzione del simbolo %s: %s", internal_symbol, e)
        provider_symbol = internal_symbol  # Fallback

    # Se il simbolo tradotto è vuoto, usa un fallback
    if not provider_symbol:
        provider_symbol = f"UNKNOWN_{internal_symbol}"

    # Aggiorna la mappatura e salva
    mapping[internal_symbol] = provider_symbol
    save_auto_symbol_mapping(mapping)

    logging.info("🔁 Simbolo tradotto e salvato: %s → %s per provider %s", internal_symbol, provider_symbol, provider)
    return provider_symbol

def get_internal_symbol(provider_symbol, provider="default"):
    """
    Traduce un simbolo specifico del provider nel formato interno standard.
    """
    provider = provider.lower()

    try:
        if provider == "yahoo":
            if "-" in provider_symbol:
                base, quote = provider_symbol.split("-")
                return f"{base.upper()}{quote.upper()}"
            elif provider_symbol.endswith("=X"):
                return provider_symbol[:-2].upper()
        elif provider == "stooq":
            if provider_symbol.endswith(".us") or provider_symbol.endswith(".fx"):
                return provider_symbol[:-3].upper()
        elif provider == "mt5":
            return provider_symbol.upper()
        elif provider == "alphavantage":
            return provider_symbol.upper()
    except Exception as e:
        logging.error("❌ Errore durante la traduzione del simbolo %s per il provider %s: %s", provider_symbol, provider, e)

    # Fallback per provider sconosciuti
    logging.warning("⚠️ Simbolo sconosciuto per il provider %s: %s", provider, provider_symbol)
    return provider_symbol.upper() if provider_symbol else "UNKNOWN"


def standardize_symbol(symbol, mapping, provider="default"):
    """
    Standardizza i simboli in modo compatibile con il provider specificato.
    """
    symbol_internal = get_internal_symbol(symbol, provider)

    if not symbol_internal or symbol_internal.startswith("UNKNOWN"):
        logging.error("❌ Simbolo interno non valido: %s", symbol)
        return None

    if symbol_internal in mapping:
        return mapping[symbol_internal]

    provider_symbol = get_provider_symbol(symbol_internal, provider)
    if not provider_symbol or provider_symbol.startswith("UNKNOWN"):
        logging.error("❌ Simbolo del provider non valido: %s", symbol_internal)
        return None

    mapping[symbol_internal] = provider_symbol
    save_auto_symbol_mapping(mapping)

    logging.info("⚠️ Simbolo sconosciuto aggiunto: %s → %s", symbol_internal, provider_symbol)
    return provider_symbol
