from __future__ import annotations

from functools import reduce
from operator import add
from typing import Callable, TypeVar

from micrograd.value import Value


def deriv(f: Callable[[float], float], x: Value | float, h: float = 1e-4) -> float:
    """Naive approximation of derivative of function `f` at point `x`"""
    if isinstance(x, Value):
        x = x.val
    y = f(x)
    y_hat_plus = abs(y - f(x + h))
    y_hat_minus = abs(y - f(x - h))
    return (y_hat_plus + y_hat_minus) / (2 * h)


T = TypeVar("T")


def flatten(xs: list[list[T]]) -> list[T]:
    """Flatten a list of lists of type `T` into list of type `T`"""
    return reduce(add, xs, [])
