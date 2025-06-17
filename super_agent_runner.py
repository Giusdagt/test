"""
super_agent_runner.py
Addestramento continuo di DRLSuperAgent su dati reali.
Utilizza dati di mercato veri da get_normalized_market_data
e asset attivi da AIModel.
"""

import asyncio
import threading
import time
import logging
from stable_baselines3.common.vec_env import DummyVecEnv
from ai_utils import prepare_ai_model
from data_handler import process_historical_data
from drl_agent import GymTradingEnv, DRLSuperAgent

print("super_agent_runner.py caricato ✅")

logging.basicConfig(level=logging.INFO)


async def load_data():
    """Carica ed elabora i dati storici richiesti."""
    await process_historical_data()
    logging.info("✅ Dati storici elaborati correttamente.")


def auto_train_super_agent():
    """
    Addestra continuamente il DRLSuperAgent su asset reali.
    Ripete l'addestramento ogni 6 ore sugli asset attivi.
    """
    asyncio.run(load_data())
    ai_model, market_data = prepare_ai_model()

    while True:
        for symbol in ai_model.active_assets:
            try:
                data = market_data[symbol]
                env_raw = GymTradingEnv(data=data, symbol=symbol)
                env = DummyVecEnv([lambda env_raw=env_raw: env_raw])
                agent = DRLSuperAgent(state_size=512, env=env)

                logging.info("🎯 Addestramento avviato su %s", symbol)
                agent.train(steps=50_000)
                logging.info("✅ Addestramento completato su %s", symbol)

            except (KeyError, ValueError) as e:
                logging.error("⚠️ Errore su %s: %s", symbol, e)

        logging.info("⏳ Pausa 6 ore prima del prossimo ciclo.")
        time.sleep(6 * 3600)


if __name__ == "__main__":
    thread = threading.Thread(
        target=auto_train_super_agent,
        daemon=True
    )
    thread.start()
