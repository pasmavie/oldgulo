import abc
import numpy as np
import pandas as pd
from gulo.backtesting import RollingValidation
from gulo.utils import expected_sd
from gulo.config import *


class BaseStrategy(object, metaclass=abc.ABCMeta):
    """
    Can have:
    - 1 or more instruments associated
    - different variations depending on various parameters
    Must have:
    - a fit() method
    - a predict() method
    """

    def __init__(self):
        self._fs = None  # forecast scalar
        self._esd = None  # expected standard deviation of asset returns

    @abc.abstractmethod
    def _decision_f(self, X: pd.DataFrame):
        """
        This method will use the rules determined by _fit
        to output a prediction
        """

    @abc.abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series):
        """
        This method will determine the rules to be used
        by _decision_f to output a prediction
        """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        +  Refer to _fit for the actual method to be implemented in
            child classes to train the decision function

        + Forecast scalar: if you’re going to create your own trading rules
        you need to rescale them, so that the average absolute value
        of the forecast is around 1 (Sorry Rob, I prefer it to 10)
        To do this you need to run a back-test of the trading strategy,
        although you only require forecast values and you don’t need to check performance.
        TODO: You should also average across as many instruments as possible.
        Excerpt From: “Systematic Trading”. Apple Books.

        """
        sd_preds = []
        for train, test in RollingValidation().split(X):
            self._fit(X[train], y[train])
            self._esd = expected_sd(y)
            sd_preds.append(
                # remember the recent vol standardisation
                self._decision_f(X[test])
                / self._esd
            )
        self._fs = EXPECTED_ABSOLUTE_FORECAST / np.mean(
            sd_preds
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Scale the return between -2 and 2, with an avg exp value of 1.
        Here a forecast=1 will correspond to allocate your volatility-scalar,
        a forecast of, let's say -2, will correspond to -2*vol-scalar.
        Use self._fs estimated during the fit
        and also self._expected_sd_asset_returns
        """
        preds = np.array([self._decision_f(x) for x in X])
        return np.clip(
            -MAX_FORECAST,
            MAX_FORECAST,
            self._fs * preds / self._esd,
        )
