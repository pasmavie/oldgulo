def choose_volatility_tgt(sharpe_ratio: float) -> float:
    """
    p.146 You should always adjust your backtested sharpe ratio
    to a lower estimate (*.75)
    Then use the "half-kelly" criterion to set the vol tgt
    """
    max_sharpe_ratio = 1
    max_vol_tgt = max_sharpe_ratio*.75/2
    return min(sharpe_ratio*.75/2, max_vol_tgt)

