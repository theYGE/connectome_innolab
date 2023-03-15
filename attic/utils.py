""""
contains helper functions and similar
"""
import functools
import logging
import time
from collections.abc import Callable
from time import process_time
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


def time_function_call(function: Callable[P, R]) -> Callable[P, R]:
    """

    Args:
        function:

    Returns:

    """

    @functools.wraps(function)
    def time_function(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.process_time()
        val = function(*args, **kwargs)
        end = time.process_time()
        print(f"clock time: {end - start}")
        return val
