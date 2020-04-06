import numpy as np
from typing import Dict
from gulo.portfolio.subsystem import Subsystem
from gulo.config import MAX_SR, MAX_VOL_TGT


class BasePortfolio(object):
    def __init__(self, k: float, ss: Dict[str, Subsystem], vol_tgt: float):
        self.k = k  # trading capital
        self.ss = ss
        self.vol_tgt = vol_tgt
        self.cash_vol_tgt = None
        self.sr = None
        
        # TODO: Beware! The annualised vol target is NOT the maximum,
        #  NOR the average, you might expect to loose in a year.
        #  There are several reasons why your expected average annual loss
        #  wouldn't be equal to the annualised expected daily standard deviation.
        #  Firstly, if your Sharpe ratio is greater than zero then
        #  your expected average annual loss will be smaller than one sigma.
        #  Secondly, in the event of losses you’d probably reduce your positions
        #  if you’re using trend following rules or stop losses.
        #  Thirdly, as you’ll see in the next chapter ‘Position Sizing’,
        #  you should reduce your positions when price volatility rises,
        #  which reduces losses in periods of rising risk.
        #  Also, as you’ll see later in this chapter,
        #  you’d reduce your risk target throughout the year if you made losses,
        #  and vice versa. More technically if consecutive returns aren’t independent,
        #  and have time series autocorrelation, then annualising
        #  by multiplying by 16 (sqrt(days_in_one_year)) is a poor approximation.
        # TODO: (PART 2) Following the above, make sure an ordinary movement
        #  against you doesn't wipe your position. Reconnects to CHF peg and low vol assets

    def _update_annualised_cash_vol_tgt(self):
        """
        At least on a daily basis.
        """
        self.cash_vol_tgt = self.vol_tgt * self.k
