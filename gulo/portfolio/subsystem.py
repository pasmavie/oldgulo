import numpy as np
import pandas as pd
from copy import copy  # or deepcopya?
from typing import *
from gulo.strategies import BaseStrategy
from gulo.backtesting import RollingValidation, SequentialValidation
from gulo.utils import performance
from gulo.config import *


class Subsystem(object):
    def __init__(self, input_strategies: Dict[str:BaseStrategy]):
        self.input_strategies = input_strategies
        self.strategies = None
        self._forecast_weights = None
        self._forecast_diversification_multiplier = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Step 1) BACKTEST
        You'll need to run a backtest for all the strategies first.
        TODO: test raw strategy skew

        Step 3) STRATEGY SELECTION
        Drop strategies with correlation > 95% and reset the forecast weights
        TODO: from here we calc the forecast diversification multiplier. Understand if you wanna use raw or scaled strategy predictions for that...

        Step 4) DIVERSIFICATION MULTIPLIER
        Since the final strategies won't be perfectly correlated, you'll need to multiply the final
        forecast by a factor, similar to the forecast scalar.
        "Given N trading rule variations with a correlation matrix of forecast values H
        and forecast weights W summing to 1, the diversification multiplier will be 1 ÷ [√(W × H × WT)]"
        Excerpt From: “Systematic Trading”. Apple Books.

        Step 5) FIT ALL THE STRATEGIES
        The final fit after the backtesting iter

        :param X: containst input for all the strategies
        :param y: is always the same
        """
        # 1) BACKTEST
        ys_hat = {}
        for train, test in RollingValidation().split(X, y):
            for strat_name, strat in self.input_strategies.items():
                strat.fit(X[train], y[train])
                pred = strat.predict(X[test])
                ys_hat[strat_name].append(pred)

        # 3) STRATEGY SELECTION
        ys_hat = pd.DataFrame(ys_hat)
        corr_matrix = ys_hat.corr().abs()  # Create correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )  # Select the upper triangle of correlation matrix
        to_drop = ys_hat.columns[
            [column for column in upper.columns if any(upper[column] > 0.95)]
        ]  # Find strategies with correlation greater than 0.95
        ys_hat.drop(columns=to_drop, inplace=True)  # Drop strategies if corr > .95
        self.strategies = {
            k: v for k, v in self.input_strategies if k in ys_hat.columns
        }  # Update self
        self._reset_forecast_weights()

        # 4) DIVERSIFICATION MULTIPLIER
        corr_matrix = ys_hat.corr().abs()  # Re-create correlation matrix
        corr_matrix = np.clip(
            0, corr_matrix
        )  # and floor negative correlations to zero (p.129)
        self._forecast_diversification_multiplier = np.divide(
            1, np.sqrt(corr_matrix @ self._forecast_weights @ np.transpose(corr_matrix))
        )  # consider limiting this like R.C. does in its book (2.5 for an avg forecast of 10)

        # 5) FIT ALL THE STRATEGIES
        for strat in self.strategies.values():
            strat.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Combine the forecasts, apply the diversification multiplier to get back to
        the expected absolute forecast. Then clip between max and min
        :param X: contains inputs for all the strategies
        """
        forecast = np.zeros(len(X))
        for strat_name, strat in self.strategies.items:  # TODO 1-liner here?
            forecast += strat.predict(X) * self._forecast_weights.get(strat_name)
        forecast = np.clip(
            -MAX_FORECAST,
            MAX_FORECAST,
            forecast * self._forecast_diversification_multiplier,
        )
        return forecast

    def _reset_forecast_weights(self, equal_weights=True):
        if equal_weights:
            new_forecast_weights = {
                k: 1 / len(self.strategies) for k in self.strategies.keys()
            }
            self._forecast_weights = new_forecast_weights
        else:
            # TODO: equal_weights is the easy choice but
            #  what shall I really do? Chapter 8 - Choosing the forecast weights
            pass

    def update_strategy_holding_period(self):
        """
        Make sure that the holding period of a strategy doesn't get excessively long
        :Law Of Active Management:
        The Sharpe Ratio of a given strategy will be proportional
        to the square root of the number of independent bets made per year.
        """
        return min(self.strategies.values().holding_period)

    def backtest(self, X, y, folds_generator: SequentialValidation):
        syscopy = copy(self)
        results = (performance(syscopy.fit(X[train], y[train]).predict(X[test]), y[test]) for train, test in folds_generator.split(X))
        return results
