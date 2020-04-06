import pandas as pd
import warnings
from scipy import stats


class BaseAsset(object):
    def __init__(self, hist_asset_returns: pd.DataFrame):
        self.hist_asset_returns = hist_asset_returns
        if stats.skew(self.hist_asset_returns.values) < 0:
            warnings.warn(
                "This asset returns have negative skew. Check that volatility is not too low and consider using a positive skew strategy",
                UserWarning,
            )

        self.instrument_currency_volatility = None  # p.158
        self.instrument_value_volatility = None  # p.158
