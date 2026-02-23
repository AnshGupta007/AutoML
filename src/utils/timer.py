"""
Performance timing utilities.
Provides decorators and context managers for timing code execution,
with automatic logging and MLflow metric tracking.
"""
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


class Timer:
    """Context manager and decorator for measuring elapsed time.

    Example:
        >>> with Timer("data ingestion"):
        ...     load_data()

        >>> @Timer.timeit
        ... def train_model():
        ...     pass
    """

    def __init__(self, name: str = "operation", log_result: bool = True) -> None:
        self.name = name
        self.log_result = log_result
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.log_result:
            log.info(f"⏱  {self.name} took {self.elapsed:.3f}s")

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    @staticmethod
    def timeit(func: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator to time a function.

        Can be used with or without arguments:
            @Timer.timeit
            def my_func(): ...

            @Timer.timeit(name="custom name")
            def my_func(): ...
        """
        def decorator(fn: Callable) -> Callable:
            label = name or f"{fn.__module__}.{fn.__qualname__}"

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with Timer(label):
                    return fn(*args, **kwargs)

            return wrapper

        if func is not None:
            # Called without arguments: @Timer.timeit
            return decorator(func)
        # Called with arguments: @Timer.timeit(name="...")
        return decorator


@contextmanager
def timed(name: str) -> Generator[Timer, None, None]:
    """Convenience context manager for timing blocks of code.

    Args:
        name: Description of the operation being timed.

    Yields:
        Timer instance with elapsed time after context exits.

    Example:
        >>> with timed("feature engineering") as t:
        ...     engineer_features()
        >>> print(f"Done in {t.elapsed:.2f}s")
    """
    timer = Timer(name, log_result=True)
    with timer:
        yield timer


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string like "2h 30m 15s" or "350ms".
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}m {secs:.0f}s"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"
