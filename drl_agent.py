"""
drl_agent.py
class DRLAgent + class DRLSuperAgent:
DRL avanzato + PPO/DQN/A2C/SAC con ReplayBuffer,
DummyVecEnv, Environment autonomo
DRLSuperAgent - Agente di Decisione Reinforcement Learning
Auto-Migliorante
"""
import os
from pathlib import Path
import logging
import numpy as np
import polars as pl
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, SAC
from sklearn.decomposition import PCA
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
from data_handler import process_historical_data
from constants import USE_DYNAMIC_STATE_SIZE, SEQUENCE_LENGTH, DESIRED_STATE_SIZE


print("drl_agent.py caricato ✅")
MODEL_PATH = Path("D:/trading_data/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PCA_FOLDER = MODEL_PATH / "pca"
DRL_FOLDER = MODEL_PATH / "drl"
PCA_FOLDER.mkdir(parents=True, exist_ok=True)
DRL_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)

async def load_data():
    """
    Carica i dati elaborati dal modulo data_handler.
    """
    await process_historical_data()
    if not Path("D:/trading_data/processed_data.zstd.parquet").exists():
        raise FileNotFoundError("Il file dei dati elaborati non esiste.")
    data = pl.read_parquet("D:/trading_data/processed_data.zstd.parquet")
    if data.is_empty():
        raise ValueError("Il file dei dati elaborati è vuoto.")
    return data


class GymTradingEnv(gym.Env):
    """
    Ambiente compatibile con Gym per simulazioni di trading.
    """
    def __init__(self, data, symbol, initial_balance=100, sequence_length=SEQUENCE_LENGTH):
        super().__init__()
        self.data = data.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy()
        self.total_steps = len(self.data)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.current_step = 0
        self.sequence_length = min(sequence_length, len(self.data))
        self.num_features = self.data.shape[1]
        self.state_size = self.sequence_length * self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

    def reset(self, seed=None):
        """
        Reimposta l'ambiente di simulazione di trading.
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        end_step = self.current_step + self.sequence_length
        obs = self.data[self.current_step:end_step]
        obs_flat = obs.flatten()

        # Padding o troncamento per garantire la dimensione corretta
        if obs_flat.shape[0] > self.state_size:
            obs_flat = obs_flat[:self.state_size]
        elif obs_flat.shape[0] < self.state_size:
            padding = np.zeros(self.state_size - obs_flat.shape[0])
            obs_flat = np.concatenate([obs_flat, padding])

        return obs_flat

    def step(self, action):
        """
        Esegue un passo nell'ambiente di simulazione di trading.
        """
        if self.current_step >= self.total_steps - self.sequence_length:
            obs = self._next_observation()
            return obs, 0.0, True, False, {}

        price_now = self.data[self.current_step][0]
        price_next = self.data[self.current_step + 1][0]

        # Aggiorna la posizione in base all'azione
        if action == 1:  # Buy
            self.position = 1
        elif action == 2:  # Sell
            self.position = -1
        else:  # Hold
            self.position = 0

        # Calcola il reward basato sulla posizione aggiornata
        reward = (price_now - price_next) * self.position
        self.balance += reward
        self.current_step += 1

        done = self.current_step >= self.total_steps - self.sequence_length or self.balance <= 0
        truncated = False
        obs = self._next_observation()
        return obs, reward, done, truncated, {}

    def render(self, mode='human'):
        """
        Mostra lo stato corrente dell'ambiente.
        """
        if mode != 'human':
            raise NotImplementedError(
                f"La modalità '{mode}' non è supportata."
            )
        print(
            f"""Step: {self.current_step},
            Balance: {self.balance},
            Position: {self.position}"""
            )


class DRLAgent:
    """
    DRLAgent - Agente di Decisione Reinforcement Learning Compresso
    """
    def __init__(self, state_size, max_memory=5000):
        self.state_size = state_size
        self.memory = []
        self.weights = np.random.normal(0, 0.1, state_size).astype(np.float32)
        self.max_memory = max_memory
        self.learning_rate = 0.01
        logging.info(
            "🧠 DRLAgent attivo | stato: %d, memoria max: %d",
            state_size,
            max_memory
        )

    def predict(self, state: np.ndarray) -> float:
        """
        Calcola una previsione basata sullo stato fornito.
        Args:
        state (np.ndarray): Lo stato attuale dell'ambiente.
        Returns:
        float: Valore previsto normalizzato tra 0 e 1.
        """
        value = np.dot(state, self.weights)
        return float(np.clip(1 / (1 + np.exp(-value)), 0, 1))

    def get_confidence(self, state: np.ndarray) -> float:
        """
        Restituisce la confidenza sulla predizione attuale.
        Basata su similarità tra stato attuale e memoria recente.
        Non cresce nel tempo.
        """
        if len(self.memory) < 10:
            return 0.1  # minima confidenza iniziale

        # Seleziona ultimi 10 stati
        recent_states = np.array([s for s, _ in self.memory[-10:]])
        dot_products = np.dot(recent_states, state)
        similarity = np.mean(dot_products)

        # Calcola varianza dei valori predetti
        predictions = [np.dot(s, self.weights) for s in recent_states]
        variance = np.var(predictions) + 1e-6

        # Formula di confidenza
        confidence = 1 / (1 + variance * (1 - similarity))
        return float(np.clip(confidence, 0.1, 1.0))

    def update(self, state: np.ndarray, outcome: float):
        """
        Aggiorna i pesi dell'agente DRL in base allo
        stato attuale e al risultato.
        Args:
        state (np.ndarray): Lo stato attuale dell'ambiente.
        outcome (float): Il risultato osservato
        (ad esempio, il reward).
        """
        self.memory.append((state, outcome))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        gradient = (outcome - np.dot(state, self.weights)) * state
        self.weights += self.learning_rate * gradient
        self.weights = np.clip(self.weights, -2, 2).astype(np.float32)

    def compress_memory(self):
        """
        Riduce la memoria dell'agente mantenendo solo gli ultimi stati
        fino al limite massimo definito.
        """
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def save(self):
        """
        Salva i pesi dell'agente DRL in un file compresso.
        """
        np.savez_compressed(
            str(MODEL_PATH / "agent_weights.npz"), weights=self.weights
        )

    def load(self):
        """
        Carica i pesi dell'agente DRL da un file salvato.
        """
        data = np.load(str(MODEL_PATH / "agent_weights.npz"))
        self.weights = data["weights"]


class DRLSuperAgent:
    """
    Agente DRL avanzato che supporta PPO, DQN, A2C, SAC con ambiente autonomo.
    """
    def __init__(self, num_features, env=None):

        dummy_data = pl.DataFrame(
            np.zeros((SEQUENCE_LENGTH, DESIRED_STATE_SIZE // SEQUENCE_LENGTH))
        )
        dummy_env = GymTradingEnv(
            data=dummy_data,
            symbol="DUMMY",
            initial_balance=100,
            sequence_length=SEQUENCE_LENGTH
        )
        print("🔍 OBS SPACE:", dummy_env.observation_space.shape)

        self.state_size = DESIRED_STATE_SIZE if not USE_DYNAMIC_STATE_SIZE else dummy_env.state_size
        if self.state_size != DESIRED_STATE_SIZE:
            logging.warning(f"⚠️ State size impostato a {self.state_size} anziché {DESIRED_STATE_SIZE}.")
        self.env = env or DummyVecEnv([lambda: dummy_env])

        self.drl_agent = DRLAgent(state_size=self.state_size)

        # Seleziona automaticamente l'algoritmo
        self.algo = self._select_algorithm()
        self.model = self._init_model(self.algo)
        logging.info("Algoritmo selezionato: %s", self.algo)

    def _select_algorithm(self):
        """
        Seleziona automaticamente l'algoritmo più adatto
        in base al tipo di spazio di azione.
        Returns:
        str: Nome dell'algoritmo selezionato.
        """
        space = self.env.envs[0].action_space
        logging.info("Tipo di spazio di azione: %s", type(space))

        if isinstance(space, spaces.Box):
            if self.state_size > 256:
                return "SAC"
            return "A2C"
        if isinstance(space, spaces.Discrete):
            if self.state_size < 256:
                return "DQN"
            if self.state_size <= 512:
                return "A2C"
            return "PPO"

        raise ValueError(
            f" spazio azione non supportato: {type(self.env.action_space)}"
        )

    def _init_model(self, algo):
        if algo == "PPO":
            return PPO(
                "MlpPolicy", self.env, verbose=0,
                n_steps=8192, # per uso meno ram 4096
                batch_size=128,
                policy_kwargs={
                    "net_arch": {
                        "pi": [128, 64], "vf": [128, 64]
                    }
                }
            )
        if algo == "DQN":
            return DQN(
                "MlpPolicy", self.env, verbose=0,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=128,
                policy_kwargs={"net_arch": [128, 64]}
            )
        if algo == "A2C":
            return A2C(
                "MlpPolicy", self.env, verbose=0,
                n_steps=8192, # per uso meno ram 4096
                batch_size=128,
                policy_kwargs={
                    "net_arch": {
                        "pi": [128, 64], "vf": [128, 64]
                    }
                }
            )
        if algo == "SAC":
            return SAC(
                "MlpPolicy", self.env, verbose=0,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=128,
                policy_kwargs={"net_arch": [128, 64]}
            ) 
        raise ValueError("Algoritmo non supportato")

    def train(self, steps=5000):
        """
        Addestra il modello DRL per un numero specificato di passi.
        Args:
        steps (int): Numero di passi di addestramento. Default è 5000.
        """
        n_steps = getattr(self.model, "n_steps", 2048)
        steps = max(steps, n_steps)
        steps = (steps // n_steps) * n_steps
        try:
            self.model.learn(total_timesteps=steps, reset_num_timesteps=True)
        except Exception as e:
            logging.error("❌ Errore nell'addestramento di %s: %s", self.algo, e)
            return
        model_path = DRL_FOLDER / f"{self.algo}_model"
        self.model.save(str(model_path))
        self.drl_agent.compress_memory()
        self.drl_agent.save()
        logging.info("💪 %s aggiornato e salvato in %s", type(self.model).__name__, model_path)

    def predict(self, state):
        if np.isnan(state).any() or np.isinf(state).any():
            logging.error("❌ Stato contiene NaN o valori infiniti. Applico correzione preventiva.")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        logging.info(f"📐 Stato ricevuto con shape: {state.shape}")

        if state.ndim != 2 or state.shape[1] == 0:
            logging.error("⚠️ Stato non valido, verifica i dati in ingresso.")
            np.save("debug_state_failed.npy", state)
            raise ValueError("⚠️ Stato non valido: shape errata.")

        if USE_DYNAMIC_STATE_SIZE:

            symbol = getattr(self.env.envs[0], "symbol", "UNKNOWN")
            timeframe = getattr(self.env.envs[0], "timeframe", "1m")
            state = apply_dynamic_pca(state, symbol, timeframe)
        else:
            if state.shape[1] != DESIRED_STATE_SIZE:
                logging.warning(f"⚠️ Dimensione stato errata: {state.shape[1]}. Adatto a {DESIRED_STATE_SIZE}.")
                if state.shape[1] > DESIRED_STATE_SIZE:
                    state = state[:, :DESIRED_STATE_SIZE]
                elif state.shape[1] < DESIRED_STATE_SIZE:
                    padding = np.zeros((state.shape[0], DESIRED_STATE_SIZE - state.shape[1]))
                    state = np.concatenate([state, padding], axis=1)

        action, _ = self.model.predict(state, deterministic=True)
        confidence = self.drl_agent.predict(state.flatten())
        return int(action[0]), float(confidence)

    def save(self):
        """
        Salva il modello DRL e i pesi dell'agente DRL su file.
        """
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.model.save(str(DRL_FOLDER / f"{self.algo}_model"))
        self.drl_agent.save()
        logging.info(f"💾 Modello {self.algo} salvato correttamente.")

    def load(self):
        """
        Carica il modello DRL e i pesi dell'agente DRL
        da file salvati.
        """
        model_path = DRL_FOLDER / f"{self.algo}_model"
        if model_path.exists():
            self.model = self.model.load(str(model_path))
            self.drl_agent.load()
            logging.info(f"📥 Modello {self.algo} caricato da {model_path}")
        else:
            logging.warning(f"⚠️ Modello {self.algo} non trovato. Creo e salvo un nuovo modello.")
            self.model = self._init_model(self.algo)
            self.drl_agent.initialize()
            self.save()
            logging.info(f"✅ Nuovo modello {self.algo} creato e salvato.")

def save_pca_model(pca, symbol: str, timeframe: str = "1m"):
    """
    Salva il modello PCA addestrato per uno specifico simbolo e timeframe.
    """
    path = PCA_FOLDER / f"{symbol}_{timeframe}_pca.pkl"
    try:
        # Assicurati che la cartella esista
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(pca, f)
        logging.info(f"💾 PCA salvato: {path}")
    except Exception as e:
        logging.warning(f"❌ Errore salvataggio PCA {symbol}: {e}")

def load_pca_model(symbol: str, timeframe: str = "1m"):
    """
    Carica il modello PCA salvato per uno specifico simbolo e timeframe.
    Returns:
    PCA model se esiste, altrimenti None.
    """
    path = PCA_FOLDER / f"{symbol}_{timeframe}_pca.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                pca = pickle.load(f)
            logging.info(f"📥 PCA caricato correttamente da: {path}")
            return pca
        except Exception as e:
            logging.warning(f"⚠️ Errore nel caricamento del PCA ({symbol}, {timeframe}): {e}")
    else:
        logging.info(f"ℹ️ PCA non trovato per {symbol} [{timeframe}]")
    return None

def apply_dynamic_pca(state, symbol, timeframe):
    try:
        pca = load_pca_model(symbol, timeframe)
        if not pca:
            pca = PCA(n_components=DESIRED_STATE_SIZE)
            if state.shape[0] < 2:
                logging.warning(f"⚠️ PCA saltato: solo {state.shape[0]} campioni disponibili.")
                return state
            state = pca.fit_transform(state)
            save_pca_model(pca, symbol, timeframe)
            logging.info("✅ PCA addestrata e salvata.")
        else:
            state = pca.transform(state)
        return state.reshape(state.shape[0], -1)
    except Exception as e:
        logging.warning(f"⚠️ PCA fallita: {e}. Uso padding.")
        pad_size = DESIRED_STATE_SIZE - state.shape[1]
        if pad_size < 0:
            logging.error(f"❌ State troppo grande per PCA padding. Dimensione: {state.shape}")
            return state[:, :DESIRED_STATE_SIZE]
        padding = np.zeros((state.shape[0], pad_size))
        return np.concatenate([state, padding], axis=1)

