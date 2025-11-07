"""Utility functions module."""

from .logging import setup_logger, get_logger
from .io import (
    save_results,
    load_results,
    save_checkpoint,
    load_checkpoint,
    save_pickle,
    load_pickle,
    save_json,
    load_json,
)
from .visualization import (
    plot_training_curves,
    plot_predictions,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
)
from .memory import (
    MemoryMappedArray,
    GPUMemoryMonitor,
    AdaptiveBatchSizer,
    cleanup_temp_files,
    cleanup_cache_directory,
)
from .profiling import (
    timer,
    profile_function,
    PerformanceMonitor,
    vectorize_operation,
    optimize_tensor_operations,
    benchmark_operation,
    TorchJITOptimizer,
    enable_cudnn_benchmark,
    disable_cudnn_benchmark,
)

# Lazy import to avoid circular dependency
# from .benchmark import (
#     FeaturizationBenchmark,
#     TrainingBenchmark,
#     InferenceBenchmark,
#     compare_implementations
# )

__all__ = [
    # Logging
    "setup_logger",
    "get_logger",
    # I/O
    "save_results",
    "load_results",
    "save_checkpoint",
    "load_checkpoint",
    "save_pickle",
    "load_pickle",
    "save_json",
    "load_json",
    # Visualization
    "plot_training_curves",
    "plot_predictions",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_feature_importance",
    # Memory
    "MemoryMappedArray",
    "GPUMemoryMonitor",
    "AdaptiveBatchSizer",
    "cleanup_temp_files",
    "cleanup_cache_directory",
    # Profiling
    "timer",
    "profile_function",
    "PerformanceMonitor",
    "vectorize_operation",
    "optimize_tensor_operations",
    "benchmark_operation",
    "TorchJITOptimizer",
    "enable_cudnn_benchmark",
    "disable_cudnn_benchmark",
    # Benchmarking - import directly from benchmark module to avoid circular import
    # 'FeaturizationBenchmark',
    # 'TrainingBenchmark',
    # 'InferenceBenchmark',
    # 'compare_implementations',
]
