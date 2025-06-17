"""
pattern_brain.py
Modulo per Pattern AI - Rete neurale secondaria
che valuta segnali tecnici (engulfing, fakeouts, squeeze, ecc.)
e supporta la decisione della rete principale con un punteggio.
"""
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

print("pattern_brain.py caricato ✅")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Dove salviamo il modello
MODEL_PATH = Path("D:/trading_data/pattern_model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_PATH / "pattern_model.keras"


class PatternBrain:
    """
    Classe per la gestione di una rete neurale secondaria
    che valuta segnali tecnici.
    La rete neurale calcola un punteggio di confidenza
    basato su pattern tecnici
    (es. engulfing, fakeouts, squeeze)
    per supportare le decisioni della rete principale.
    """
    def __init__(self, input_size=5):
        """
        Inizializza un'istanza della classe PatternBrain.
        Args:
        input_size (int):
        Dimensione dell'input per il modello. Default è 5.
        """
        self.input_size = input_size
        self.model = self.build_model()
        self.load()

    def build_model(self):
        """
        Costruisce e restituisce il modello della rete neurale.
        Returns:
        tf.keras.Sequential: Modello della rete neurale compilato.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def predict_score(self, pattern_array):
        """
        Calcola il punteggio di confidenza per un array di pattern.
        Args:
        pattern_array (array-like):
        Array di input che rappresenta i pattern tecnici.
        Returns:
        float: Punteggio di confidenza calcolato dal modello.
        """
        pattern_array = np.array(pattern_array).reshape(1, -1)
        score = float(self.model.predict(pattern_array, verbose=0)[0][0])
        return score

    def train(self, x_train, y_train, epochs=5):
        """
        Addestra il modello con i dati forniti.
        Args:
        x_train (array-like): Dati di input per l'addestramento.
        y_train (array-like): Etichette corrispondenti ai dati di input.
        epochs (int): Numero di epoche per l'addestramento.
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1)
        self.save()

    def save(self):
        """Salva il modello addestrato nel file specificato."""
        self.model.save(MODEL_FILE)
        logging.info("📦 PatternBrain salvato in %s", MODEL_FILE)

    def load(self):
        """Carica il modello addestrato dal file specificato."""
        if MODEL_FILE.exists():
            self.model = tf.keras.models.load_model(MODEL_FILE)
            logging.info("🧠 PatternBrain caricato da %s", MODEL_FILE)


# Utilizzo tipico:
# pattern_data = [ilq, fakeout_up, fakeout_down, squeeze, micro_pattern]
# confidence = pattern_brain.predict_score(pattern_data)
# if confidence > 0.7: potenzia l'azione suggerita dalla rete principale
