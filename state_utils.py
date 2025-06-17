import numpy as np
from constants import DESIRED_STATE_SIZE

print("state_utils.py caricato ✅")

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def get_col_val_safe(row, col):
    try:
        return row[col].item()
    except:
        return 0

def sanitize_full_state(full_state, desired_length=1000):
    """
    Garantisce che full_state abbia sempre la stessa lunghezza in righe (es: 1000).
    Se mancano righe, aggiunge padding. Se in eccesso, tronca.
    """
    current_length = full_state.shape[0]

    if current_length == desired_length:
        return full_state
    elif current_length > desired_length:
        return full_state[-desired_length:]
    else:
        n_missing = desired_length - current_length
        padding = np.zeros((n_missing, full_state.shape[1]))  # padding 2D
        return np.concatenate([padding, full_state], axis=0)
    
def sanitize_columns(full_state, desired_width=DESIRED_STATE_SIZE):
    """
    Garantisce che full_state abbia sempre DESIRED_STATE_SIZE colonne.
    Applica padding o taglia se necessario.
    """
    current_width = full_state.shape[1]

    if current_width == desired_width:
        return full_state
    elif current_width > desired_width:
        return full_state[:, :desired_width]
    else:
        padding = np.zeros((full_state.shape[0], desired_width - current_width))
        return np.concatenate([full_state, padding], axis=1)


