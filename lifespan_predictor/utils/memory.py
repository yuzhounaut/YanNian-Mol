"""
Memory optimization utilities for efficient data processing.

This module provides utilities for memory-efficient operations including
memory-mapped arrays, GPU memory monitoring, and automatic batch size reduction.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MemoryMappedArray:
    """
    Memory-mapped array wrapper for large feature matrices.

    This class provides a memory-efficient way to store and access large
    numpy arrays by using memory-mapped files instead of loading everything
    into RAM.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the array
    dtype : np.dtype
        Data type of the array
    filepath : Optional[str]
        Path to memory-mapped file. If None, creates temporary file
    mode : str
        File mode ('r+' for read/write, 'r' for read-only, 'w+' for write)

    Attributes
    ----------
    filepath : Path
        Path to memory-mapped file
    shape : Tuple[int, ...]
        Shape of the array
    dtype : np.dtype
        Data type
    mode : str
        File mode
    array : np.memmap
        Memory-mapped array
    is_temp : bool
        Whether file is temporary

    Examples
    --------
    >>> # Create memory-mapped array
    >>> mmap_array = MemoryMappedArray(shape=(1000, 200, 75), dtype=np.float32)
    >>> mmap_array.array[0] = np.random.randn(200, 75)
    >>>
    >>> # Access data
    >>> data = mmap_array.array[0:10]
    >>>
    >>> # Clean up
    >>> mmap_array.close()
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        filepath: Optional[str] = None,
        mode: str = "w+",
    ):
        """Initialize MemoryMappedArray."""
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.is_temp = filepath is None

        if filepath is None:
            # Create temporary file
            fd, filepath = tempfile.mkstemp(suffix=".mmap", prefix="lifespan_")
            os.close(fd)  # Close file descriptor, numpy will reopen
            logger.debug(f"Created temporary memory-mapped file: {filepath}")

        self.filepath = Path(filepath)

        # Create memory-mapped array
        self.array = np.memmap(
            str(self.filepath), dtype=self.dtype, mode=self.mode, shape=self.shape
        )

        logger.info(
            f"Initialized MemoryMappedArray: shape={shape}, dtype={dtype}, "
            f"size={self.get_size_mb():.2f} MB, filepath={self.filepath}"
        )

    def get_size_mb(self) -> float:
        """
        Get size of array in megabytes.

        Returns
        -------
        float
            Size in MB
        """
        num_elements = np.prod(self.shape)
        bytes_per_element = np.dtype(self.dtype).itemsize
        return (num_elements * bytes_per_element) / (1024 * 1024)

    def flush(self) -> None:
        """Flush changes to disk."""
        if self.array is not None:
            self.array.flush()
            logger.debug(f"Flushed memory-mapped array to {self.filepath}")

    def close(self) -> None:
        """Close and optionally delete the memory-mapped file."""
        if self.array is not None:
            # Flush any pending writes
            self.flush()

            # Delete reference to allow file to be closed
            del self.array
            self.array = None

            # Delete temporary file
            if self.is_temp and self.filepath.exists():
                try:
                    self.filepath.unlink()
                    logger.debug(f"Deleted temporary file: {self.filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {self.filepath}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


class GPUMemoryMonitor:
    """
    Monitor GPU memory usage and provide utilities for memory management.

    This class provides methods to monitor GPU memory usage, clear cache,
    and detect out-of-memory conditions.

    Parameters
    ----------
    device : str
        Device to monitor (e.g., 'cuda:0')
    threshold : float
        Memory usage threshold (0.0 to 1.0) for warnings

    Attributes
    ----------
    device : torch.device
        Device being monitored
    threshold : float
        Warning threshold

    Examples
    --------
    >>> monitor = GPUMemoryMonitor('cuda:0', threshold=0.8)
    >>> monitor.log_memory_usage()
    >>> if monitor.is_memory_critical():
    ...     monitor.clear_cache()
    """

    def __init__(self, device: str = "cuda:0", threshold: float = 0.8):
        """Initialize GPUMemoryMonitor."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device = torch.device(device)
        self.threshold = threshold

        # Get device properties
        device_id = self.device.index if self.device.index is not None else 0
        self.device_properties = torch.cuda.get_device_properties(device_id)
        self.total_memory = self.device_properties.total_memory

        logger.info(
            f"Initialized GPUMemoryMonitor for {device}: "
            f"total_memory={self.total_memory / 1e9:.2f} GB, "
            f"threshold={threshold * 100:.0f}%"
        )

    def get_memory_usage(self) -> Tuple[int, int, float]:
        """
        Get current GPU memory usage.

        Returns
        -------
        Tuple[int, int, float]
            - Allocated memory in bytes
            - Total memory in bytes
            - Usage ratio (0.0 to 1.0)
        """
        device_id = self.device.index if self.device.index is not None else 0
        allocated = torch.cuda.memory_allocated(device_id)
        total = self.total_memory
        ratio = allocated / total if total > 0 else 0.0

        return allocated, total, ratio

    def log_memory_usage(self, prefix: str = "") -> None:
        """
        Log current GPU memory usage.

        Parameters
        ----------
        prefix : str
            Prefix for log message
        """
        allocated, total, ratio = self.get_memory_usage()

        message = (
            f"{prefix}GPU Memory: "
            f"{allocated / 1e9:.2f} GB / {total / 1e9:.2f} GB "
            f"({ratio * 100:.1f}%)"
        )

        if ratio >= self.threshold:
            logger.warning(message)
        else:
            logger.info(message)

    def is_memory_critical(self) -> bool:
        """
        Check if memory usage exceeds threshold.

        Returns
        -------
        bool
            True if memory usage exceeds threshold
        """
        _, _, ratio = self.get_memory_usage()
        return ratio >= self.threshold

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
            self.log_memory_usage("After cache clear - ")

    def get_available_memory(self) -> int:
        """
        Get available GPU memory in bytes.

        Returns
        -------
        int
            Available memory in bytes
        """
        allocated, total, _ = self.get_memory_usage()
        return total - allocated


class AdaptiveBatchSizer:
    """
    Automatically adjust batch size based on GPU memory availability.

    This class monitors GPU memory and reduces batch size when
    out-of-memory errors occur.

    Parameters
    ----------
    initial_batch_size : int
        Initial batch size to try
    min_batch_size : int
        Minimum allowed batch size
    reduction_factor : float
        Factor to reduce batch size by (e.g., 0.5 for halving)
    device : str
        Device to monitor

    Attributes
    ----------
    current_batch_size : int
        Current batch size
    initial_batch_size : int
        Initial batch size
    min_batch_size : int
        Minimum batch size
    reduction_factor : float
        Reduction factor
    monitor : GPUMemoryMonitor
        GPU memory monitor
    oom_count : int
        Number of OOM errors encountered

    Examples
    --------
    >>> sizer = AdaptiveBatchSizer(initial_batch_size=32, min_batch_size=4)
    >>> try:
    ...     # Training code
    ...     pass
    ... except RuntimeError as e:
    ...     if "out of memory" in str(e):
    ...         new_batch_size = sizer.reduce_batch_size()
    ...         print(f"Reduced batch size to {new_batch_size}")
    """

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        reduction_factor: float = 0.5,
        device: str = "cuda:0",
    ):
        """Initialize AdaptiveBatchSizer."""
        if initial_batch_size < min_batch_size:
            raise ValueError(
                f"initial_batch_size ({initial_batch_size}) must be >= "
                f"min_batch_size ({min_batch_size})"
            )

        if not 0 < reduction_factor < 1:
            raise ValueError(f"reduction_factor must be between 0 and 1, got {reduction_factor}")

        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.oom_count = 0

        # Initialize GPU monitor if CUDA available
        self.monitor = None
        if torch.cuda.is_available():
            try:
                self.monitor = GPUMemoryMonitor(device=device)
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitor: {e}")

        logger.info(
            f"Initialized AdaptiveBatchSizer: "
            f"initial={initial_batch_size}, min={min_batch_size}, "
            f"reduction_factor={reduction_factor}"
        )

    def reduce_batch_size(self) -> int:
        """
        Reduce batch size after OOM error.

        Returns
        -------
        int
            New batch size

        Raises
        ------
        RuntimeError
            If batch size is already at minimum
        """
        self.oom_count += 1

        # Calculate new batch size
        new_batch_size = max(
            self.min_batch_size, int(self.current_batch_size * self.reduction_factor)
        )

        if new_batch_size == self.current_batch_size:
            raise RuntimeError(
                f"Cannot reduce batch size below minimum ({self.min_batch_size}). "
                f"OOM errors: {self.oom_count}"
            )

        logger.warning(
            f"Reducing batch size from {self.current_batch_size} to {new_batch_size} "
            f"(OOM count: {self.oom_count})"
        )

        self.current_batch_size = new_batch_size

        # Clear GPU cache
        if self.monitor is not None:
            self.monitor.clear_cache()

        return self.current_batch_size

    def get_current_batch_size(self) -> int:
        """
        Get current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.current_batch_size

    def reset(self) -> None:
        """Reset batch size to initial value."""
        self.current_batch_size = self.initial_batch_size
        self.oom_count = 0
        logger.info(f"Reset batch size to {self.initial_batch_size}")


def cleanup_temp_files(directory: str, pattern: str = "*.mmap") -> int:
    """
    Clean up temporary files in a directory.

    Parameters
    ----------
    directory : str
        Directory to clean
    pattern : str
        File pattern to match (default: "*.mmap")

    Returns
    -------
    int
        Number of files deleted

    Examples
    --------
    >>> count = cleanup_temp_files("/tmp", "*.mmap")
    >>> print(f"Deleted {count} temporary files")
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return 0

    deleted_count = 0
    total_size = 0

    for filepath in directory_path.glob(pattern):
        try:
            file_size = filepath.stat().st_size
            filepath.unlink()
            deleted_count += 1
            total_size += file_size
            logger.debug(f"Deleted temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete {filepath}: {e}")

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} temporary files "
            f"({total_size / 1e6:.2f} MB) from {directory}"
        )

    return deleted_count


def cleanup_cache_directory(cache_dir: str, max_size_gb: float = 10.0) -> None:
    """
    Clean up cache directory if it exceeds size limit.

    Removes oldest files first until directory size is below limit.

    Parameters
    ----------
    cache_dir : str
        Cache directory path
    max_size_gb : float
        Maximum size in gigabytes

    Examples
    --------
    >>> cleanup_cache_directory("cache/features", max_size_gb=5.0)
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return

    # Calculate current size
    total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    total_size_gb = total_size / 1e9

    if total_size_gb <= max_size_gb:
        logger.debug(
            f"Cache directory size ({total_size_gb:.2f} GB) is below limit "
            f"({max_size_gb:.2f} GB)"
        )
        return

    logger.warning(
        f"Cache directory size ({total_size_gb:.2f} GB) exceeds limit "
        f"({max_size_gb:.2f} GB). Cleaning up..."
    )

    # Get all files sorted by modification time (oldest first)
    files = sorted(
        [f for f in cache_path.rglob("*") if f.is_file()], key=lambda f: f.stat().st_mtime
    )

    # Delete files until size is below limit
    deleted_count = 0
    freed_size = 0

    for filepath in files:
        if total_size_gb <= max_size_gb:
            break

        try:
            file_size = filepath.stat().st_size
            filepath.unlink()

            deleted_count += 1
            freed_size += file_size
            total_size_gb -= file_size / 1e9

            logger.debug(f"Deleted cache file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete {filepath}: {e}")

    logger.info(
        f"Cleaned up {deleted_count} cache files "
        f"({freed_size / 1e9:.2f} GB freed). "
        f"New cache size: {total_size_gb:.2f} GB"
    )
