import numpy as np
import pandas as pd
# import pymc3 as pm


def expected_sd(hist_data: pd.Series, window=25, ewma_window=None):
    """
    Use EWMA to estimate future standard deviation
    Suggested value for window=25 and for ewma_window=36
    """
    if ewma_window:
        # https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
        # For alternatives only, beware of numpy floating point precision problems when the input is too large.
        # This is because (1-alpha)**(n+1) -> 0 when n -> inf and alpha -> 1, leading to divide-by-zero's and NaN values popping up in the calculation.
        sd = hist_data.ewm(span=ewma_window).std().as_matrix()[-1]
    else:
        sd = np.std(hist_data[-window:])
    return sd
