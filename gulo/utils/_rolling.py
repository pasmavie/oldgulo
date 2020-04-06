import numpy as np
from gulo.utils import deprecated


@deprecated("Slow on large arrays. Use pd.Series(y).rolling(window=lookback).mean()")
def sma(y: np.array, lookback: int) -> np.array:
    """
    Simple Moving Average
    """
    ws = np.repeat(1, lookback) / lookback
    return np.convolve(y, ws, "valid")


def fast_sma(x: np.array, l: int) -> float:
    """
    x: input array
    l: lookback
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[l:] - cumsum[:-l]) / float(l)


@deprecated("Slow on large arrays. Use pd.Series(y).rolling(window=lookback).std(ddof=1)")
def smstd(y: np.array, lookback: int) -> np.array:
    """
    Simple Moving Standard Deviation
    """
    stds = np.array([np.sqrt(np.sum(np.square(y[i - lookback : i] - np.mean(y[i - lookback : i]))) / (lookback - 1)) for i in np.arange(lookback, len(y) + 1)])
    return stds
