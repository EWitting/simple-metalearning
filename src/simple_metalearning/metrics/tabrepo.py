from typing import List, Union
import numpy as np


def compute_normalized_regret(
    ground_truth: np.ndarray,
    revealed_performances: Union[List[float], np.ndarray],
) -> float:
    """
    Compute normalized regret: same as regret but on TabRepo-normalized values.

    Normalization maps the best ground truth value to 1.0 and the median to 0.0
    (clipped at 0). The normalization is fixed to the full set of ground truth values.

    Monotonically decreases as better flows are found, just like regular regret.
    """
    optimal = np.max(ground_truth)
    median = np.median(ground_truth)
    best_so_far = np.max(revealed_performances) if len(revealed_performances) > 0 else -np.inf

    if optimal == median:
        return 0.0

    if best_so_far == -np.inf:
        return 1.0

    normalized_best = np.clip((best_so_far - median) / (optimal - median), 0, None)
    return float(1.0 - normalized_best)


def compute_improvability(
    ground_truth: np.ndarray,
    revealed_performances: Union[List[float], np.ndarray],
) -> float:
    """
    Compute improvability: (err_best_so_far - err_optimal) / err_best_so_far * 100%

    Measures how much better the best possible result is relative to what has been found.
    Monotonically decreases as better flows are found.

    - Negative error (all values <= 0): err = -val
    - Accuracy (all values in [0, 1]): err = 1 - val
    - Otherwise: returns NaN (without warning).
    """
    is_negative_error = np.all(ground_truth <= 1e-9)
    is_accuracy = np.all((ground_truth >= -1e-9) & (ground_truth <= 1.0 + 1e-9))

    if not is_negative_error and not is_accuracy:
        return np.nan

    optimal = np.max(ground_truth)
    best_so_far = np.max(revealed_performances) if len(revealed_performances) > 0 else -np.inf

    if is_negative_error:
        err_current = -best_so_far if best_so_far != -np.inf else np.inf
        err_optimal = -optimal
    else:
        err_current = 1.0 - best_so_far if best_so_far != -np.inf else np.inf
        err_optimal = 1.0 - optimal

    if not np.isfinite(err_current):
        return 100.0

    if err_current <= 1e-9:
        return 0.0

    return float(np.clip((err_current - err_optimal) / err_current * 100.0, 0.0, 100.0))
