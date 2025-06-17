"""
Modulo ponte per evitare import ciclici tra ai_model,
drl_agent, drl_super_integration e position_manager
"""
import asyncio
import logging
import sys
from stable_baselines3.common.vec_env import DummyVecEnv
from drl_agent import GymTradingEnv, DRLSuperAgent
from ai_utils import prepare_ai_model
from data_handler import process_historical_data

print("rl_bridge.py caricato ✅")

logging.basicConfig(level=logging.INFO)


async def load_data():
    """
    Elabora i dati storici richiesti per
    il sistema di trading.
    Questa funzione è asincrona e utilizza
    `process_historical_data`
    per preparare i dati richiesti da altri moduli.
    Attualmente restituisce un valore placeholder
    `True` per future espansioni.
    Returns:bool: Sempre `True`, come placeholder.
    """
    await process_historical_data()
    return True  # placeholder per future espansioni

if __name__ == "__main__":
    try:
        # Carica i dati elaborati e i bilanci
        asyncio.run(load_data())
        ai_model, market_data = prepare_ai_model()

        for symbol in ai_model.active_assets:
            try:
                env_raw = GymTradingEnv(
                    data=market_data[symbol],
                    symbol=symbol
                )
                env = DummyVecEnv([lambda env_raw=env_raw: env_raw])

                agent_discrete = DRLSuperAgent(
                    state_size=512, env=env
                )
                agent_discrete.train(steps=200_000)

                agent_continuous = DRLSuperAgent(
                    state_size=512, env=env
                )
                agent_continuous.train(steps=200_000)

            except (ValueError, FileNotFoundError) as e:
                logging.error("⚠️ Errore su %s: %s", symbol, e)

        print("✅ Agenti DRL addestrati e salvati")

    except (ValueError, FileNotFoundError) as e:
        logging.error("Errore nel main: %s", e)
        sys.exit(1)
