"""
Profiling utilities for performance optimization.

This module provides utilities for profiling code performance,
identifying bottlenecks, and optimizing computations.
"""

import cProfile
import io
import logging
import pstats
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str = "Operation", log_level: int = logging.INFO):
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str
        Name of the operation being timed
    log_level : int
        Logging level for the timing message

    Examples
    --------
    >>> with timer("Data loading"):
    ...     data = load_data()
    Data loading completed in 2.34s
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.log(log_level, f"{name} completed in {elapsed:.2f}s")


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function using cProfile.

    Parameters
    ----------
    func : Callable
        Function to profile

    Returns
    -------
    Callable
        Wrapped function that logs profiling stats

    Examples
    --------
    >>> @profile_function
    ... def expensive_operation():
    ...     # Some computation
    ...     pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

            # Print stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions

            logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")

        return result

    return wrapper


class PerformanceMonitor:
    """
    Monitor performance metrics during training or inference.

    This class tracks timing and throughput metrics for various operations.

    Parameters
    ----------
    name : str
        Name of the monitor

    Attributes
    ----------
    name : str
        Monitor name
    metrics : Dict[str, list]
        Dictionary of metric lists
    start_times : Dict[str, float]
        Dictionary of operation start times

    Examples
    --------
    >>> monitor = PerformanceMonitor("Training")
    >>> monitor.start("epoch")
    >>> # ... training code ...
    >>> monitor.stop("epoch")
    >>> monitor.log_summary()
    """

    def __init__(self, name: str = "Performance"):
        """Initialize PerformanceMonitor."""
        self.name = name
        self.metrics: Dict[str, list] = {}
        self.start_times: Dict[str, float] = {}

        logger.info(f"Initialized PerformanceMonitor: {name}")

    def start(self, operation: str) -> None:
        """
        Start timing an operation.

        Parameters
        ----------
        operation : str
            Name of the operation
        """
        self.start_times[operation] = time.time()

    def stop(self, operation: str) -> float:
        """
        Stop timing an operation and record the duration.

        Parameters
        ----------
        operation : str
            Name of the operation

        Returns
        -------
        float
            Duration in seconds

        Raises
        ------
        ValueError
            If operation was not started
        """
        if operation not in self.start_times:
            raise ValueError(f"Operation '{operation}' was not started")

        duration = time.time() - self.start_times[operation]

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)
        del self.start_times[operation]

        return duration

    def record(self, metric_name: str, value: float) -> None:
        """
        Record a metric value.

        Parameters
        ----------
        metric_name : str
            Name of the metric
        value : float
            Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> Optional[float]:
        """
        Get average value of a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric

        Returns
        -------
        Optional[float]
            Average value, or None if metric doesn't exist
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])

    def get_total(self, metric_name: str) -> Optional[float]:
        """
        Get total value of a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric

        Returns
        -------
        Optional[float]
            Total value, or None if metric doesn't exist
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        return sum(self.metrics[metric_name])

    def log_summary(self) -> None:
        """Log summary of all metrics."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Performance Summary: {self.name}")
        logger.info(f"{'=' * 60}")

        for metric_name, values in sorted(self.metrics.items()):
            if not values:
                continue

            avg = sum(values) / len(values)
            total = sum(values)
            count = len(values)

            logger.info(
                f"{metric_name:30s}: " f"avg={avg:8.4f}s, total={total:8.2f}s, count={count:5d}"
            )

        logger.info(f"{'=' * 60}\n")

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()
        logger.debug(f"Reset PerformanceMonitor: {self.name}")


def vectorize_operation(func: Callable) -> Callable:
    """
    Decorator to ensure operations are vectorized where possible.

    This decorator logs a warning if the function is called in a loop
    instead of being vectorized.

    Parameters
    ----------
    func : Callable
        Function to wrap

    Returns
    -------
    Callable
        Wrapped function
    """
    call_count = {"count": 0, "last_reset": time.time()}

    @wraps(func)
    def wrapper(*args, **kwargs):
        call_count["count"] += 1

        # Check if being called frequently (potential loop)
        current_time = time.time()
        if current_time - call_count["last_reset"] < 1.0:  # Within 1 second
            if call_count["count"] > 100:
                logger.warning(
                    f"Function '{func.__name__}' called {call_count['count']} times "
                    f"in 1 second. Consider vectorizing this operation."
                )
                call_count["count"] = 0
                call_count["last_reset"] = current_time
        else:
            call_count["count"] = 0
            call_count["last_reset"] = current_time

        return func(*args, **kwargs)

    return wrapper


def optimize_tensor_operations(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimize tensor for faster operations.

    This function ensures tensors are contiguous and in the optimal format.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Optimized tensor

    Examples
    --------
    >>> x = torch.randn(100, 100).t()  # Non-contiguous
    >>> x_opt = optimize_tensor_operations(x)
    >>> x_opt.is_contiguous()
    True
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        logger.debug("Made tensor contiguous")

    return tensor


def benchmark_operation(
    func: Callable, *args, n_iterations: int = 100, warmup: int = 10, **kwargs
) -> Dict[str, float]:
    """
    Benchmark an operation by running it multiple times.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    *args
        Positional arguments for the function
    n_iterations : int
        Number of iterations to run
    warmup : int
        Number of warmup iterations (not counted)
    **kwargs
        Keyword arguments for the function

    Returns
    -------
    Dict[str, float]
        Dictionary containing timing statistics

    Examples
    --------
    >>> def my_func(x):
    ...     return x ** 2
    >>> stats = benchmark_operation(my_func, 10, n_iterations=1000)
    >>> print(f"Average time: {stats['mean']:.6f}s")
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start)

    # Calculate statistics
    times_array = torch.tensor(times)
    stats = {
        "mean": times_array.mean().item(),
        "std": times_array.std().item(),
        "min": times_array.min().item(),
        "max": times_array.max().item(),
        "median": times_array.median().item(),
        "n_iterations": n_iterations,
    }

    logger.info(
        f"Benchmark results for {func.__name__}: "
        f"mean={stats['mean']:.6f}s, std={stats['std']:.6f}s, "
        f"min={stats['min']:.6f}s, max={stats['max']:.6f}s"
    )

    return stats


class TorchJITOptimizer:
    """
    Utility for optimizing models with TorchScript JIT compilation.

    This class provides methods to compile models or functions using
    torch.jit.script or torch.jit.trace for faster execution.

    Examples
    --------
    >>> optimizer = TorchJITOptimizer()
    >>> optimized_model = optimizer.optimize_model(model, example_input)
    """

    @staticmethod
    def optimize_model(
        model: torch.nn.Module, example_input: Any, use_trace: bool = True
    ) -> torch.jit.ScriptModule:
        """
        Optimize a model using TorchScript.

        Parameters
        ----------
        model : torch.nn.Module
            Model to optimize
        example_input : Any
            Example input for tracing
        use_trace : bool
            Use torch.jit.trace (True) or torch.jit.script (False)

        Returns
        -------
        torch.jit.ScriptModule
            Optimized model

        Examples
        --------
        >>> model = MyModel()
        >>> example = torch.randn(1, 10)
        >>> optimized = TorchJITOptimizer.optimize_model(model, example)
        """
        model.eval()

        try:
            if use_trace:
                logger.info("Optimizing model with torch.jit.trace")
                optimized = torch.jit.trace(model, example_input)
            else:
                logger.info("Optimizing model with torch.jit.script")
                optimized = torch.jit.script(model)

            logger.info("Model optimization successful")
            return optimized

        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            logger.warning("Returning original model")
            return model

    @staticmethod
    def optimize_function(func: Callable) -> Callable:
        """
        Optimize a function using torch.jit.script.

        Parameters
        ----------
        func : Callable
            Function to optimize

        Returns
        -------
        Callable
            Optimized function

        Examples
        --------
        >>> @torch.jit.script
        ... def my_func(x: torch.Tensor) -> torch.Tensor:
        ...     return x * 2
        >>> optimized = TorchJITOptimizer.optimize_function(my_func)
        """
        try:
            logger.info(f"Optimizing function {func.__name__} with torch.jit.script")
            optimized = torch.jit.script(func)
            logger.info("Function optimization successful")
            return optimized

        except Exception as e:
            logger.error(f"Failed to optimize function: {e}")
            logger.warning("Returning original function")
            return func


def enable_cudnn_benchmark() -> None:
    """
    Enable cuDNN autotuner for faster convolutions.

    This should be called at the start of training if input sizes are fixed.

    Examples
    --------
    >>> enable_cudnn_benchmark()
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode for faster convolutions")
    else:
        logger.warning("CUDA not available, cuDNN benchmark not enabled")


def disable_cudnn_benchmark() -> None:
    """
    Disable cuDNN autotuner.

    Examples
    --------
    >>> disable_cudnn_benchmark()
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        logger.info("Disabled cuDNN benchmark mode")
