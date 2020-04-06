import numpy as np
import functools
import warnings
from typing import Callable


def deprecated(msg: str = ""):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"Call to deprecated function {func.__name__}. {msg}", category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def sign(cv1: float, cv10: float) -> Callable[[float, float], bool]:
    """
    Infer the right function to compare the T-stat to its critical values given two of them, passed in descending order starting from the highest significance level (normally 1%).
    t represents the T-statistic, cv stays for critical value
    """
    if cv1 < cv10:
        return lambda t, cv: t <= cv
    elif cv1 > cv10:
        return lambda t, cv: t >= cv
    else:
        raise ValueError("a and b are equal")
