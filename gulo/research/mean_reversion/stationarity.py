# E.Chan p.62
# Stationarity means that prices diffuse slower than a geometric random walk
# Mean reversion means that the change in price is proportional to the difference between the mean price and the current price.

# The ADF test is designed to test for mean reversion.
# The Hurst exponent and Variance Ratio tests are designed to test for stationarity.
# Half-life of mean reversion measures how quickly a price series reverts to its mean, and is a good predictor of the profitability or Sharpe ratio of a mean- reverting trading strategy when applied to this price series.
# A linear trading strategy here means the number of units or shares of a unit portfolio we own is proportional to the negative Z-Score of the price series of that portfolio.
# If we can combine two or more nonstationary price series to form a stationary portfolio, these price series are called cointegrating.
# Cointegration can be measured by either CADF test or Johansen test.
# The eigenvectors generated from the Johansen test can be used as hedge ratios to form a stationary portfolio out of the input price series, and the one with the largest eigenvalue is the one with the shortest half-life.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa import stattools
from typing import Tuple, Callable, Dict, Union
from gulo.utils import sign


def adf(
    y, maxlag: int = None, regression: str = "c", autolag: int = None, t_cv: int = 5
) -> Tuple[bool, float, Dict[str, float]]:
    """
    EChan AlgoTr p.43
     dy[t] = lambda*y[t-1] + mu + beta[t] + alpha[1]*dy[t-1] + ... + alpha[k]*dy[t-k] + t
    ADF test that lambda != 0 (lambda=0 is the null hypothesis).
    If lambda is significantly negative, the series is mean reverting.
    :param y: ts in ascending order
    :maxlag: check tsa.stattools.adfuller, defaults to 12 * (len(y) / 100) ** .25
    :regression: check tsa.stattools.adfuller
    :autolag: check tsa.stattools.adfuller
    :param t_cv: T-stat critical value level, determines significance
    :return: True if stationary
    """
    if not maxlag:
        maxlag = int(12 * (len(y) / 100) ** 0.25)
    t, _, _, _, cvs = stattools.adfuller(
        x=y, maxlag=maxlag, regression=regression, autolag=autolag
    )
    cv = cvs[f"{t_cv}%"]  # critical value
    condition_1 = t < 0  # must have negative lambda
    condition_2 = sign(cvs["1%"], cvs["10%"])(
        t, cv
    )  # t-stat must be significant at the chosen level
    return (condition_1 & condition_2, t, cvs)


def hurst_exp(y: Union[np.array, pd.Series], lags: int = None) -> Tuple[float, float]:
    """
    Hurst Exponent as per EChan AlgoTr p.45: intuitively speaking, a “stationary” price series means that
    the prices diffuse from its initial value more slowly than a geometric random walk would.
    Mathematically, we can determine the nature of the price series by measuring this speed of diffusion.
    The speed of diffusion can be characterized by the variance:
      Var(tau) = np.std(z[tau:]-z[:-tau])
    where z is the log prices (z = log(y)) and tau is an arbitrary time lag.
    For a generic random walk we know that the speed is proportional to tau, for large taus: Speed(tau) ~ tau.
    For a mean-reverting or a trending time series we can write speed as:
      Var(tau) ~ tau**2H
    where H is the Hurst Exponent and it's =.5 for a geometric rw, <.5 for a mean reverting series and >.5 for a trending one.
    The more towards the extremes and the more trending/mean-reverting.

    Important notes on estimating the Hurst Exponent using the R/S method here:
    http://bearcave.com/misl/misl_tech/wavelets/hurst/index.html#EstimatingHurstWithRs
    The link above is the best resource I've found covering the topic.
    :param y: time series of LOG(prices)
    :param lags: as stated here https://robotwealth.com/demystifying-the-hurst-exponent-part-2/
        the number of lags used matters: if Hurst is a measure of the memory in a time series 
        then, depending on the time scale, this memory can be both mean-reverting and trending.
    :return: H=0.5 indicates a random-walk, H>0.5 a trend, H<0.5 stationarity. 
    """
    # Here the cleanest implementation by Dr.E.P.Chan
    z = pd.DataFrame(y)
    taus = np.arange(np.round(len(z) / 10)).astype(
        int
    )  # We cannot use tau that is of same magnitude of time series length
    X = np.log(taus)
    Y = np.empty(len(taus))  # log variance
    for tau in taus:
        Y[tau] = np.log(z.diff(tau).var(ddof=0))
    # Y = Y[:len(taus)] <- don't need it
    X = X[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    H = results.params[1] / 2
    pvalue = results.pvalues[1]
    return H, pvalue


def halflife(y: Union[pd.Series, np.array]) -> float:
    """
    How long it would take a time series to revert back to half it’s initial deviation away from the mean
    
    To estimate half life you (H) first need to find the speed of mean reversion (lambda).
    Perform a linear regression with the prices as the independent variable and returns as the dependent.
    Lambda is equal to the slope of the linear regression line. Thus H = log(2)/lambda as seen here: http://marcoagd.usuarios.rdc.puc-rio.br/half-life.html
    """
    if type(y)==pd.Series:
        y = y.values
    y_1 = y[:-1]  # y[t-1]
    dy = y[1:] - y_1  # y[t] - y[t-1] 
    res = sm.OLS(dy, sm.add_constant(y_1)).fit()
    # half-life
    hl = -np.log(2) / res.params[1]
    return hl
