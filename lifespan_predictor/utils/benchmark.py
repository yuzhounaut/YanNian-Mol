"""
Benchmarking utilities for performance measurement.

This module provides comprehensive benchmarking tools to measure
featurization speed, training speed, and inference speed.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from lifespan_predictor.data.featurizers import CachedGraphFeaturizer
from lifespan_predictor.data.fingerprints import FingerprintGenerator
from lifespan_predictor.utils.profiling import PerformanceMonitor

logger = logging.getLogger(__name__)


class FeaturizationBenchmark:
    """
    Benchmark molecular featurization performance.

    This class measures the speed of graph featurization and
    fingerprint generation.

    Parameters
    ----------
    cache_dir : str
        Directory for caching features

    Examples
    --------
    >>> benchmark = FeaturizationBenchmark("cache/benchmark")
    >>> results = benchmark.run(smiles_list, n_runs=3)
    >>> print(f"Featurization speed: {results['molecules_per_second']:.2f} mol/s")
    """

    def __init__(self, cache_dir: str):
        """Initialize FeaturizationBenchmark."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FeaturizationBenchmark with cache_dir={cache_dir}")

    def benchmark_graph_featurization(
        self, smiles_list: List[str], n_runs: int = 3, n_jobs: int = -1
    ) -> Dict[str, float]:
        """
        Benchmark graph featurization speed.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to featurize
        n_runs : int
            Number of benchmark runs
        n_jobs : int
            Number of parallel jobs

        Returns
        -------
        Dict[str, float]
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking graph featurization with {len(smiles_list)} molecules")

        featurizer = CachedGraphFeaturizer(
            cache_dir=str(self.cache_dir / "graph_features"),
            n_jobs=n_jobs,
            use_memory_mapping=False,  # Disable for fair comparison
        )

        times = []

        for run in range(n_runs):
            logger.info(f"Run {run + 1}/{n_runs}")

            start_time = time.time()
            adj, feat, sim, labels = featurizer.featurize(smiles_list, force_recompute=True)
            elapsed = time.time() - start_time

            times.append(elapsed)
            logger.info(f"  Completed in {elapsed:.2f}s")

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        molecules_per_second = len(smiles_list) / mean_time

        results = {
            "n_molecules": len(smiles_list),
            "n_runs": n_runs,
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": np.min(times),
            "max_time": np.max(times),
            "molecules_per_second": molecules_per_second,
            "n_jobs": n_jobs,
        }

        logger.info(
            f"Graph featurization benchmark results:\n"
            f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s\n"
            f"  Throughput: {molecules_per_second:.2f} molecules/s\n"
            f"  Time per molecule: {mean_time / len(smiles_list) * 1000:.2f}ms"
        )

        return results

    def benchmark_fingerprint_generation(
        self, smiles_list: List[str], n_runs: int = 3, n_jobs: int = -1
    ) -> Dict[str, float]:
        """
        Benchmark fingerprint generation speed.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        n_runs : int
            Number of benchmark runs
        n_jobs : int
            Number of parallel jobs

        Returns
        -------
        Dict[str, float]
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking fingerprint generation with {len(smiles_list)} molecules")

        generator = FingerprintGenerator(n_jobs=n_jobs)

        times = []

        for run in range(n_runs):
            logger.info(f"Run {run + 1}/{n_runs}")

            start_time = time.time()
            hashed_fps, maccs_fps = generator.generate_fingerprints(
                smiles_list, cache_dir=None  # Disable caching for fair comparison
            )
            elapsed = time.time() - start_time

            times.append(elapsed)
            logger.info(f"  Completed in {elapsed:.2f}s")

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        molecules_per_second = len(smiles_list) / mean_time

        results = {
            "n_molecules": len(smiles_list),
            "n_runs": n_runs,
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": np.min(times),
            "max_time": np.max(times),
            "molecules_per_second": molecules_per_second,
            "n_jobs": n_jobs,
        }

        logger.info(
            f"Fingerprint generation benchmark results:\n"
            f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s\n"
            f"  Throughput: {molecules_per_second:.2f} molecules/s\n"
            f"  Time per molecule: {mean_time / len(smiles_list) * 1000:.2f}ms"
        )

        return results

    def run_full_benchmark(
        self, smiles_list: List[str], n_runs: int = 3, n_jobs: int = -1
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full featurization benchmark.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        n_runs : int
            Number of benchmark runs
        n_jobs : int
            Number of parallel jobs

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing all benchmark results
        """
        logger.info("Running full featurization benchmark")

        results = {
            "graph_featurization": self.benchmark_graph_featurization(smiles_list, n_runs, n_jobs),
            "fingerprint_generation": self.benchmark_fingerprint_generation(
                smiles_list, n_runs, n_jobs
            ),
        }

        # Calculate total time
        total_time = (
            results["graph_featurization"]["mean_time"]
            + results["fingerprint_generation"]["mean_time"]
        )

        results["total"] = {
            "mean_time": total_time,
            "molecules_per_second": len(smiles_list) / total_time,
        }

        logger.info(
            f"\nFull featurization benchmark summary:\n"
            f"  Total time: {total_time:.2f}s\n"
            f"  Overall throughput: {results['total']['molecules_per_second']:.2f} molecules/s"
        )

        return results


class TrainingBenchmark:
    """
    Benchmark model training performance.

    This class measures training speed per epoch and identifies bottlenecks.

    Examples
    --------
    >>> benchmark = TrainingBenchmark()
    >>> results = benchmark.run(model, train_loader, device='cuda:0')
    >>> print(f"Training speed: {results['samples_per_second']:.2f} samples/s")
    """

    def __init__(self):
        """Initialize TrainingBenchmark."""
        self.monitor = PerformanceMonitor("Training")
        logger.info("Initialized TrainingBenchmark")

    def benchmark_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda:0",
        n_epochs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark training speed per epoch.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train
        dataloader : DataLoader
            Training data loader
        criterion : torch.nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        device : str
            Device to use
        n_epochs : int
            Number of epochs to benchmark

        Returns
        -------
        Dict[str, float]
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking training speed for {n_epochs} epochs")

        model.to(device)
        model.train()

        epoch_times = []
        batch_times = []

        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")

            epoch_start = time.time()

            for batch_idx, batch in enumerate(dataloader):
                batch_start = time.time()

                # Move batch to device
                batch = batch.to(device)

                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch)
                loss = criterion(predictions, batch.y)

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(
                        f"  Batch {batch_idx + 1}/{len(dataloader)}: " f"{batch_time:.4f}s"
                    )

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            logger.info(f"  Epoch completed in {epoch_time:.2f}s")

        # Calculate statistics
        mean_epoch_time = np.mean(epoch_times)
        std_epoch_time = np.std(epoch_times)
        mean_batch_time = np.mean(batch_times)

        # Calculate throughput
        total_samples = len(dataloader.dataset)
        samples_per_second = total_samples / mean_epoch_time

        results = {
            "n_epochs": n_epochs,
            "mean_epoch_time": mean_epoch_time,
            "std_epoch_time": std_epoch_time,
            "min_epoch_time": np.min(epoch_times),
            "max_epoch_time": np.max(epoch_times),
            "mean_batch_time": mean_batch_time,
            "samples_per_second": samples_per_second,
            "batches_per_second": len(dataloader) / mean_epoch_time,
            "total_samples": total_samples,
            "batch_size": dataloader.batch_size,
        }

        logger.info(
            f"\nTraining benchmark results:\n"
            f"  Mean epoch time: {mean_epoch_time:.2f}s ± {std_epoch_time:.2f}s\n"
            f"  Mean batch time: {mean_batch_time:.4f}s\n"
            f"  Throughput: {samples_per_second:.2f} samples/s\n"
            f"  Batches per second: {results['batches_per_second']:.2f}"
        )

        return results


class InferenceBenchmark:
    """
    Benchmark model inference performance.

    This class measures inference speed and latency.

    Examples
    --------
    >>> benchmark = InferenceBenchmark()
    >>> results = benchmark.run(model, test_loader, device='cuda:0')
    >>> print(f"Inference speed: {results['samples_per_second']:.2f} samples/s")
    """

    def __init__(self):
        """Initialize InferenceBenchmark."""
        logger.info("Initialized InferenceBenchmark")

    def benchmark_inference(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda:0",
        n_runs: int = 3,
        use_mixed_precision: bool = False,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Parameters
        ----------
        model : torch.nn.Module
            Model to benchmark
        dataloader : DataLoader
            Data loader
        device : str
            Device to use
        n_runs : int
            Number of benchmark runs
        use_mixed_precision : bool
            Use mixed precision inference

        Returns
        -------
        Dict[str, float]
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking inference speed for {n_runs} runs")

        model.to(device)
        model.eval()

        run_times = []
        batch_times = []

        with torch.no_grad():
            for run in range(n_runs):
                logger.info(f"Run {run + 1}/{n_runs}")

                run_start = time.time()

                for batch in dataloader:
                    batch_start = time.time()

                    # Move batch to device
                    batch = batch.to(device)

                    # Forward pass
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            model(batch)
                    else:
                        model(batch)

                    # Synchronize for accurate timing
                    if "cuda" in device:
                        torch.cuda.synchronize()

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                run_time = time.time() - run_start
                run_times.append(run_time)

                logger.info(f"  Run completed in {run_time:.2f}s")

        # Calculate statistics
        mean_run_time = np.mean(run_times)
        std_run_time = np.std(run_times)
        mean_batch_time = np.mean(batch_times)

        # Calculate throughput
        total_samples = len(dataloader.dataset)
        samples_per_second = total_samples / mean_run_time

        # Calculate latency (time per sample)
        latency_ms = (mean_run_time / total_samples) * 1000

        results = {
            "n_runs": n_runs,
            "mean_run_time": mean_run_time,
            "std_run_time": std_run_time,
            "min_run_time": np.min(run_times),
            "max_run_time": np.max(run_times),
            "mean_batch_time": mean_batch_time,
            "samples_per_second": samples_per_second,
            "latency_ms": latency_ms,
            "total_samples": total_samples,
            "batch_size": dataloader.batch_size,
            "use_mixed_precision": use_mixed_precision,
        }

        logger.info(
            f"\nInference benchmark results:\n"
            f"  Mean run time: {mean_run_time:.2f}s ± {std_run_time:.2f}s\n"
            f"  Mean batch time: {mean_batch_time:.4f}s\n"
            f"  Throughput: {samples_per_second:.2f} samples/s\n"
            f"  Latency: {latency_ms:.2f}ms per sample\n"
            f"  Mixed precision: {use_mixed_precision}"
        )

        return results


def compare_implementations(
    results_new: Dict[str, float],
    results_old: Dict[str, float],
    metric_name: str = "molecules_per_second",
) -> Dict[str, float]:
    """
    Compare performance between two implementations.

    Parameters
    ----------
    results_new : Dict[str, float]
        Results from new implementation
    results_old : Dict[str, float]
        Results from old implementation
    metric_name : str
        Name of the metric to compare

    Returns
    -------
    Dict[str, float]
        Dictionary containing comparison results

    Examples
    --------
    >>> comparison = compare_implementations(new_results, old_results)
    >>> print(f"Speedup: {comparison['speedup']:.2f}x")
    """
    if metric_name not in results_new or metric_name not in results_old:
        raise ValueError(f"Metric '{metric_name}' not found in results")

    new_value = results_new[metric_name]
    old_value = results_old[metric_name]

    speedup = new_value / old_value if old_value > 0 else 0
    improvement_pct = ((new_value - old_value) / old_value * 100) if old_value > 0 else 0

    comparison = {
        "new_value": new_value,
        "old_value": old_value,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "metric_name": metric_name,
    }

    logger.info(
        f"\nPerformance comparison ({metric_name}):\n"
        f"  Old: {old_value:.2f}\n"
        f"  New: {new_value:.2f}\n"
        f"  Speedup: {speedup:.2f}x\n"
        f"  Improvement: {improvement_pct:+.1f}%"
    )

    return comparison
