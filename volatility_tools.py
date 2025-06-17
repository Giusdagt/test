"""
volatility_tools.py
Modulo per la previsione della volatilità.
Utilizza memoria compressa e regressione lineare leggera.
Nessun file, nessuna crescita infinita.
"""

import numpy as np

print("volatility_tools.py caricato✅")


class VolatilityPredictor:
    """
    Classe per la previsione della volatilità.
    Questa classe utilizza una memoria compressa per memorizzare i dati
    più recenti (massimo 1000 esempi) e apprende un modello di previsione
    tramite regressione lineare leggera. Se il modello non è addestrato,
    utilizza la deviazione standard come fallback.
    Attributi:
        memory (list): Memoria interna per memorizzare le feature e i target.
        weights (np.ndarray): Pesi del modello appresi tramite regressione.
    """
    def __init__(self):
        # Memoria interna per apprendimento (massimo 1000 elementi)
        self.memory = []
        # Pesi del modello (imparati via regressione)
        self.weights = None

    def update(self, features: np.ndarray, target_volatility: float):
        """
        Aggiorna la memoria e ricalcola i pesi se ci sono dati sufficienti.
        Args:
        features (np.ndarray): 1D o 2D array di input (full_state).
        target_volatility (float): Volatilità reale osservata (ultimo valore).
        """
        # Assicura che sia 2D per consistenza
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Aggiunge il nuovo esempio alla memoria
        self.memory.append((features[0], target_volatility))

        # Compressione: mantiene solo gli ultimi 1000 esempi
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]

        # Prepara i dati per l'addestramento
        feature_matrix = np.array([f for f, _ in self.memory])
        y = np.array([v for _, v in self.memory])

        # Solo se abbastanza dati (almeno tanti quanto le feature)
        if len(feature_matrix) >= feature_matrix.shape[1]:
            # Pseudo-regressione lineare compressa
            self.weights = np.linalg.pinv(feature_matrix) @ y

    def predict_volatility(self, features: np.ndarray):
        """
        Predice la volatilità. Se non addestrato, fallback con std().
        Args:
            features (np.ndarray): input 2D
        Returns:
            np.ndarray: valori di volatilità previsti
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.weights is None or features.shape[1] != self.weights.shape[0]:
            return np.std(features, axis=1)  # fallback

        return features @ self.weights
