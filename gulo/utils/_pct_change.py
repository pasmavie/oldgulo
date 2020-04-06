import numpy as np

def pct_change(x: np.array) -> np.array:
    """
    Consider using pd.DataFrame().pct_change()...
    """
    return np.divide(np.diff(x), x[:-1])
