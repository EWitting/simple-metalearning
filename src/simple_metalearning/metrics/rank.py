import numpy as np
import pandas as pd

def weighted_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute weighted rank correlation.

    Formula: r_w(R̃, R) = 1 - (6 Σ(R̃ᵢ - Rᵢ)²((n - R̃ᵢ + 1) + (n - Rᵢ + 1))) / (n⁴ + n³ - n² - n)

    where:
    - R̃ = predicted ranks (highest score = rank 1)
    - R = true ranks (highest value = rank 1)
    - n = number of items

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Weighted rank correlation coefficient (0 if either input is constant)
    """
    n = len(y_true)
    if n < 2:
        return np.nan

    # If either input is constant, correlation is 0 (no ranking information)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    # Compute ranks (highest = 1, using average for ties)
    # scipy.stats.rankdata with method='average' gives 1 to lowest, so we need to reverse
    true_ranks = (
        n + 1 - pd.Series(y_true).rank(method="average", ascending=False).values
    )
    pred_ranks = (
        n + 1 - pd.Series(y_pred).rank(method="average", ascending=False).values
    )

    # Compute weighted rank correlation
    numerator = 6 * np.sum(
        (pred_ranks - true_ranks) ** 2 * ((n - pred_ranks + 1) + (n - true_ranks + 1))
    )
    denominator = n**4 + n**3 - n**2 - n

    if denominator == 0:
        return np.nan

    return 1 - (numerator / denominator)

