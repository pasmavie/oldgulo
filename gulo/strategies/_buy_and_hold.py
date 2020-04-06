import pandas as pd
from gulo.strategies import BaseStrategy
from gulo.config import *

class BuyAndHold(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.pred = None

    def _decision_f(self, X: pd.DataFrame):
        """
        This method will use the rules determined by _fit
        to output a prediction
        """
        return self.pred

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        """
        This method will determine the rules to be used
        by _decision_function to output a prediction
        """
        self.pred = MAX_FORECAST
