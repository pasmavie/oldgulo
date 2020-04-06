import warnings
from scipy import stats

class Metrics:
    
    @staticmethod
    def test_skew(x):
        return stats.skew(x)


class AssetMetrics(Metrics):   
    
    @staticmethod
    def test_skew(x):
        skew = super().test_skew(x)
        if skew < 0:
            warnings.warn("This asset returns have negative skew. Check that volatility is not too low and consider using a positive skew strategy", UserWarning)

