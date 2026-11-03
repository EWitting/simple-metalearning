import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr


# Returns 0 when either input is constant — undefined correlation gets score 0.
# Returns None when fewer than 2 values are available — caller should skip logging.
def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = pearsonr(y_true, y_pred)[0]
    return 0.0 if not np.isfinite(result) else float(result)


def safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(y_true, y_pred)[0]
    return 0.0 if not np.isfinite(result) else float(result)
