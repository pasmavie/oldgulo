import abc
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
import warnings
from sklearn.exceptions import NotFittedError
from typing import Dict, List, Optional, Union
from gulo.utils import sign
from gulo.research.mean_reversion.stationarity import adf, hurst_exp, halflife


class AbstractMrStrat(metaclass=abc.ABCMeta):
    def __init__(self, max_hl: float):
        self.max_hl = max_hl  # max half life
        self.hl = None
        self._fitted = False

    def fit(self, y: Union[np.array, pd.Series]):
        """
        1) Test for mean reversion
        ADF and Hurst Exponent are better than a MR-strat backtest because they depend on less parameters
        and if we find out that a ts mean reverts, we can 
        2) Calculate Half-life of mean reversion and verify it's short enough for our trading horizon
        """
        ADF = adf(y)
        HE = hurst_exp(np.log(y))
        if not ADF[0]:
            warnings.warn(f"This series is not stationary. Augmented Dickey-Fuller: {ADF}")
        if HE[0] >= 0.5:
            warnings.warn(f"This series does not mean revert. Hurst Exp.: {HE}")

        self.hl = int(round(halflife(y.values)))
        if self.hl > self.max_hl:
            warnings.warn(f"Halflife {self.hl} exceeds limit")
        self._fitted = True
        return self

    def predict(self, y: Union[np.array, pd.Series]) -> float:
        """
        1) Calculate the Z-score, i.e. the normalized (by moving std) deviation of the price from its moving avg
        2) Use the Z-score as a signal to decide entry and exit.
            A simple strategy would be to just mantain the nr lots negatively proportional to the Z-score.* 
            * valid if we're using time series of prices but it requires adjustment if it's log returns etc.
        Of course this is not practical as it'd require constant rebalancing.
        """
        if not self._fitted:
            raise NotFittedError
        if type(y) != pd.Series:
            y = pd.Series(y)
        wdw = y.rolling(window=self.hl)
        zscore = -(y - wdw.mean()) / wdw.std(ddof=1)

        return zscore

    @abc.abstractmethod
    def trade(self, y):
        raise NotImplementedError


class SimpleMrStrat(AbstractMrStrat):
    def __init__(self, entry=None, exit=None, **kwargs):
        super().__init__(**kwargs)
        self.lots = None
        self.entry: Optional[float] = entry
        if self.entry is None:
            self.entry = 1
        self.exit: Optional[float] = exit
        if self.exit is None:
            self.exit = 0

    def trade(self, y: Union[np.array, pd.Series]) -> int:

        z = self.predict(y)
        z = z.values[-1]
        absz = abs(z)
        lots = self.lots
        if absz <= self.exit:
            self.lots = 0
        elif absz > self.entry:
            self.lots = -z / absz  # = +/- 1
        return self.lots


class SimpleHedgeRatioStrat:
    """
    Suitable for two cointegrating time series
    Not a very practical strategy due to the constant infinitesimal rebalancing and the demand of unlimited buying power. 
    """

    @staticmethod
    def trade(x: np.array, y: np.array) -> (int, float):
        """
        The Cointegration ADF test produces different results depending on which of the two time series is chosen as the independent variable.
        So, we repeat the test two times [f(x, y), f(y, x)] and check which one produces the most negative t-statistic.
        Then we calculate the hedge ratio to combine the two time series into a stationary one.
        
        Long x (the independent series), short y by a number of units equivalent to the hedge ratio.
        NOTE: This is valid if we're using time series of prices but it needs adjustments if it's log(prices), returns etc etc
        """
        xy_t, xy_pv, _ = ts.coint(x, y)[0]
        yx_t, yx_pv, _ = ts.coint(y, x)[0]
        if (yx_t < xy_t) & (yx_pv < 0.05):
            raise ValueError("Y should be the independent variable here, not X")
        xy = sm.OLS(x, sm.add_constant(y)).fit()
        hedge_ratio = xy.results[0]
        return (1, hedge_ratio)


class CointegratingPortfolioStrat:
    """
    Suitable for n cointegrating ts
    Not a very practical strategy due to the constant infinitesimal rebalancing and the demand of unlimited buying power. 
    """

    def __init__(self):
        self.ws = None
        self.hl = None
        self._fitted = False

    def fit(self, ts: pd.DataFrame):
        """
        Use the Johansen test to calculate the portfolio shares for each instrument.
        This test uses some nice linear algebra to test wether "A", the first autoregression coefficient matrix (of course a matrix, we have multiple timeseries vectors here) is zero (null hypothesis) or not.
        To achieve this an eigenvalue decomposition of "A" is carried out. The rank of the matrix is given by and the Johansen test sequentially tests whether this rank is equal to zero, equal to one, through to r=n-1, where n is the number of time series under test.
        The eigenvalue decomposition results in a set of eigenvectors. 
        The eigenvectors generated from the Johansen test can be used as hedge ratios to form a stationary portfolio out of the input price series, and the one with the largest eigenvalue is the one with the shortest half-life.
        :param ts: dataframe with each column being an instrument ts
        """
        jh = vm.coint_johansen(ts.values, det_order=0, k_ar_diff=1)  # constant term, 1-lag difference

        # assert that the trace statistics are greater than their 90% critical value
        curred_sign = sign(jh.cvt[0, 2], jh.cvt[0, 0])  # note that the 0 index corresponds to the 90% cv and the 2 index is the 99% cv
        # assert (curred_sign(jh.lr1, jh.cvt[:, 0])).all()

        # assert that the maximum eigenvalue statistics are greater than their 90% critical value
        curred_sign = sign(jh.cvm[0, 2], jh.cvm[0, 0])
        # assert (curred_sign(jh.lr2 > jh.cvm[:, 0])).all()

        # E.P.Chan: the eigenvectors (represented as column vectors in r.evec) are ordered in decreasing order of their corresponding eigenvalues. So we should expect the first cointegrating relation to be the “strongest”; that is, have the shortest half-life for mean reversion.
        # assert np.argmax(jh.eig) == 0  # check it to be sure eheh
        self.ws = jh.evec[:, 0]

        # create the mean reverting time series
        yport = np.dot(ts.values, self.ws)  # it's also the (net) market value of portfolio
        self.hl = halflife(yport)

        self._fitted = True
        return self

    def predict(self, ts: pd.DataFrame) -> np.array:
        """
        Use the computed eigenvectors weights and half-life to create a stationary time series. Then calculate its normalized deviation from the moving avg.
        """
        if not self._fitted:
            raise NotFittedError

        # setting lookback to the halflife found above
        lookback = np.round(self.hl).astype(int)
        # calc the mean-reverting time series
        yport = pd.Series(np.dot(ts.values, self.ws))
        # as usual, determine the normalized deviation of the price from its moving average, and maintain the number of currency units (ex. USD) in the portfolio negatively proportional to this normalized deviation.
        nr_units = -(yport - yport.rolling(lookback).mean()) / yport.rolling(lookback).std()
        # since nr_units refers to the total portfolio value, I'll have to calculate the position on the single instruments (represented as percentage of their dollar value)
        return nr_units.values

    def trade(self, ts: pd.DataFrame) -> np.array:
        """
        From the predict() nr_units, which is negatively proportional to our synthetic statioanary ts' Z-score, calculate the single instruments' postion using the weights.
        NOTE: This is valid if we're using time series of PRICES but it needs adjustments if it's log(prices), for exam for example
        Example: the classic case is y = sum(hi*yi) = h1*y1 + h2*y2 + ..., with yi being the price of instrument i; then hi - the hedge ratio - represents the nr.units of the instrument.
        E.P.Chan proposes to consider ln(q) = sum(hi*ln(yi)) instead and proposes to take the delta: d(ln(q))_t= h1 * ( ln(y1[t]) - ln(y1[t-1])) + ... = h1 * ln(y1[t]/y1[t-1]) + ... which for small changes approximates to d(ln(q)) =  h1 * dy1/y1 + h2 * dy2/y2 + ... 
        In this case dyi/yi is the return of instrument i so, assuming hi is the capital allocation to instrument i, we can consider d(ln(q))= dq/q to be the delta in the value of the portfolio. From here we can say that q is the market value of a portfolio with constant capital allocations hi plus a cash component that we need to consider in order to rebalance and keep the capital allocations constant! Thanks Dr. Chan, incredibly insightful. 
        """
        nr_units = self.predict(ts)
        positions = np.dot(np.expand_dims(nr_units, axis=1), np.expand_dims(self.ws, axis=0))  # [t x 1]*[1 x w]
        # NOTE: positions will be like [-1.20727087,  1.66170944, -0.35474169], which means short first instrument by 120%, long the second 166% etc
        return positions

    def fake_backtest(self, ts: pd.DataFrame) -> np.array:
        train_ix = int(0.6 * ts.shape[0])
        train, test = ts.iloc[:train_ix], ts.iloc[train_ix:]
        self.fit(train)
        positions = self.trade(test)
        positions = pd.DataFrame(positions * test)
        pnl = np.sum((positions.shift().values) * (test.pct_change().values), axis=1)  # daily P&L of the strategy
        return pnl / np.sum(np.abs(positions.shift()), axis=1)


class BollingerBandsStrat(CointegratingPortfolioStrat):
    """
    Implements entry and exit rules at given Z-scores
    """

    def __init__(self, entry_z: float = None, exit_z: float = None):
        """
        I assume here symmetric entry and exit for the two bands which means there will actually be 4 bands. When outside the outermost ones we build our position, in between the outermost and the inner ones we do nothing and when we touch the exit we free ourselves. 
        In this scenario, an exit at Z=0 means that the inner bands are lined one upon the other and we only exit when the series reverts to its moving avg
        An exit at -entry_z means that we keep our position until we exit at the opposite boundary of the band, when we'd expect the series to change direction again. It implies that we're very confident about the stationarity of this series...
        """
        super().__init__()
        self.entry = entry_z
        self.exit = exit_z
        if not self.entry:
            self.entry = 1
        if not self.exit:
            self.exit = 0
        if self.entry < self.exit:
            raise ValueError

    @abc.abstractmethod
    def open_positions() -> bool:
        """
        TODO: <ib_insync implementation>
        """
        pass

    def close(self) -> float:
        """
        Close positions
        If the strategy has open trades returns the opposite of the current synthetic stationary ts value, so that it will then close the position,
        else returns zero
        """
        if self.open_positions():
            nr_units = -np.dot(ts.values, self.ws)
        else:
            nr_units = 0
        return nr_units

    def trade(ts: pd.DataFrame) -> np.array:
        """
        Shadows CointegratingPortfolioStrats.trade() with a more reasonable strat
        """
        zscore = -self.predict(ts)
        nr_units = np.where(
            abs(zscore) > self.entry_zscore, np.sign(zscore) * self.entry, np.where(abs(zscore) < self.exit, self.close(), 0)
        )  # NOTE: that I'm not taking positions larger than the entry...this is just a choice
        positions = np.dot(np.expand_dims(nr_units, axis=1), np.expand_dims(self.ws, axis=0))  # [t x 1]*[1 x w]
        return positions


class KalmanFilterStrat:
    """
    Thank you: www.kalmanfilter.net
    The Kalman Filter computations are based on five equations.
    
    Two prediction equations:
    
    1) State extrapolation equation = prediction or estimation of the future state, based on the known present estimation
        
        xhat[t+1|t] = F * xhat[t|t] + G * muhat[t|t] + w[t]
        
        where:
            muhat[t|t] is a control variable, a measurable deterministic input to the system
            w[t] is the process noise, unmeasurable input that affects the state
            F is a state transition matrix
            G is is a control matrix, mapping control to state variables
    
    2) Covariance extrapolation equation = the measure of uncertainty in our prediction
        
        P[t+1|t] = F * P[t|t] * F' + Q
        
        where:
            F is a state trainsition matrix
            Q is a process noise matrix
    
    ---
    
    3) The Kalman Gain equation = required to compute the update equations
        
        K[t] = P[t|t-1]*H'/(H*P[t|t-1]*H' + R[t])
    
    ---
    
    Two update equations:
    
    4) State update equation = estimation of the current state, based on the known past estimation and present measurement
    
        xhat[t|t] = xhat[t|t-1] + K[t] * (z[t] - H*xhat[t|t-1])
        
        where:
            z[t] - H*xhat[t|t-1] is a specification of z[t] - yhat[t|t-1]; check point 6)
    
    5) Covariance update equation = measure of uncertainty in our estimation
        
        P[t|t] = (I - K[t]*H) * P[t|t-1] * (I - K[t]*H)' + K[t]*R[t]*K[t]'
        
        where:
            R[t] is the measurement noise covariance matrix

    ---

    Finally, there are some auxiliary equations.
    
    6) Measurement equation = in many case the measured value is not the desired system state, this equation is a linear transformation of the system state into the measurement
   
        z[t] = Hx[t] + v[t]
        
        where:
            z[t] is a measurement vector
            x[t] is the hidden system state vector
            v[t] is a random noise vector
            H is the observation matrix to apply the linear transformation

    """

    def __init__(self, delta: float = None):

        self.delta = (
            delta
        )  # Used to calculate the State Covariance noise. Delta=1 gives the fastest change in beta, delta~0 allows no change (like traditional linear regression).
        if not self.delta:
            self.delta = 0.0001

    def predict(self, x: np.array, y: np.array):
        """ 
        Measurement equation:
            y[t] = b[t]*x[t] + e[t] 
        where:
            e ~ N(0, Ve)
            b is a [Nx2] array <- intercept and slope
        
        y is the observable variable/state
        x is the observation model
        b is the hidden variable/state
        ---
        
        1)  State extrapolation: 
            
            yhat[t+1] = x[t+1] * bhat[t+1|t] 
            bhat[t+1|t] = bhat[t|t]  <- constant dynamic
        
        2) State Covariance extrapolation: 
            
            Qhat[t+1|t] = x[t+1] * Rhat[t|t] * x[t+1]' + Ve  <-- Qhat is the variance of e[t]
            Rhat[t+1|t] = Rhat[t|t] + Vw  <-- Rhat is the measurement uncertainty

            where: Ve and Vw are gaussian noise
         
        3) Kalman gain:
            K[t] = Rhat[t|t-1]*x[t] / Qhat[t]
        
        4) Hidden State update:
            bhat[t|t] = bhat[t|t-1] + K[t]*( y[t] - x[t]*bhat[t|t-1] )
        
        5) Hidden State covariance update:
            Rhat[t|t] = R[t|t-1] - K[t]*x[t]*Rhat[t|t-1]
        
            """
        x = np.array(ts.add_constant(x))[:, [1, 0]]  # Augment x with ones to  accomodate possible offset in the regression between y vs x.

        # Initialize yhat, bhat, Qhat and Rhat.
        yhat = np.empty(y.shape[0])  # measurement predictions
        Qhat = yhat.copy()  # measurement predictions error variance

        bhat = np.empty((x.shape[1], x.shape[0]))
        bhat[:, 0] = 0  # Initialize beta(:, 1) to zero
        Rhat = np.zeros((x.shape[1], x.shape[1]))

        Vw = self.delta / (1 - self.delta) * np.eye(x.shape[1])
        Ve = 0.001

        for t in range(len(y)):
            if t > 0:
                # Hidden state extrapolation
                bhat[:, t] = bhat[:, t - 1]
                # Hidden state covariance extrapolation
                Rhat = Rhat + Vw

            # Observable State extrapolation
            yhat[t] = np.dot(x[t, :], bhat[:, t])

            # Observable State variance extrapolation
            Qhat[t] = np.dot(np.dot(x[t, :], Rhat), x[t, :].T) + Ve

            # Kalman gain
            K = np.dot(Rhat, x[t, :].T) / Qhat[t]

            # Hidden state update
            bhat[:, t] = bhat[:, t] + np.dot(K, y[t] - yhat[t])

            # Hidden state covariance update
            Rhat = Rhat - np.dot(np.dot(K, x[t, :]), Rhat)

        return yhat, bhat, Qhat

    def trade(self, x: np.array, y: np.array):
        """
        TODO: fix it
        """
        yhat, bhat, Qhat = self.predict(x, y)
        deviation_from_spread_mean = y - yhat

        long_entry = deviation_from_spread_mean < -np.sqrt(Qhat)
        long_exit = deviation_from_spread_mean > 0

        short_entry = deviation_from_spread_mean > np.sqrt(Qhat)
        short_exit = deviation_from_spread_mean < 0

        nr_units = np.empty(x.shape)
        nr_units[long_entry] = 1
        nr_units[short_entry] = -1
        nr_units[short_exit | long_exit] = 0

        positions = np.dot(np.expand_dims(nr_units, axis=1), np.expand_dims(bhat, axis=0))
