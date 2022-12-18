import functools
import logging
from collections.abc import Callable
from typing import TypeVar

# TODO: cannot import ParamSpec, Concatenate from typing -> import error
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def log_function_call(function: Callable[P, R]) -> Callable[P, R]:
    """
    decorator function for logging function calls.
    """

    @functools.wraps(function)
    def log_function(*args: P.args, **kwargs: P.kwargs) -> R:
        logging.info(f"{'=' * 8} Call {function.__name__} {'=' * 8} ")
        return function(*args, **kwargs)

    return log_function
