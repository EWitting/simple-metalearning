import numpy as np


def compute_regret(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute regret: difference between best true value and the true value of the item with highest predicted score.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Regret value
    """
    if len(y_true) == 0:
        return np.nan

    best_true = np.max(y_true)
    optimal_indices = np.where(y_pred == np.max(y_pred))[0]
    predicted_best_idx = np.random.choice(optimal_indices)
    selected_true = y_true[predicted_best_idx]

    return best_true - selected_true
