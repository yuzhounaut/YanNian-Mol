#!/usr/bin/env python
"""
Standalone script to run performance benchmarks.

This script benchmarks featurization, training, and inference performance
and generates a comprehensive report.

Usage:
    python scripts/run_benchmarks.py --config config/default_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lifespan_predictor.config import Config
from lifespan_predictor.data.preprocessing import load_and_clean_csv
from lifespan_predictor.models.predictor import LifespanPredictor
from lifespan_predictor.utils.benchmark import (
    FeaturizationBenchmark,
    TrainingBenchmark,
    InferenceBenchmark,
)
from lifespan_predictor.utils.logging import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for lifespan predictor"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument("--data", type=str, help="Path to CSV data file (overrides config)")

    parser.add_argument(
        "--n-molecules", type=int, default=100, help="Number of molecules to use for benchmarking"
    )

    parser.add_argument("--n-runs", type=int, default=3, help="Number of benchmark runs")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )

    parser.add_argument(
        "--skip-featurization", action="store_true", help="Skip featurization benchmark"
    )

    parser.add_argument("--skip-training", action="store_true", help="Skip training benchmark")

    parser.add_argument("--skip-inference", action="store_true", help="Skip inference benchmark")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for benchmarking",
    )

    return parser.parse_args()


def benchmark_featurization(smiles_list, cache_dir, n_runs=3, n_jobs=-1):
    """Run featurization benchmark."""
    logger.info("=" * 80)
    logger.info("FEATURIZATION BENCHMARK")
    logger.info("=" * 80)

    benchmark = FeaturizationBenchmark(cache_dir)
    results = benchmark.run_full_benchmark(smiles_list, n_runs=n_runs, n_jobs=n_jobs)

    return results


def benchmark_training(model, train_loader, criterion, optimizer, device, n_epochs=3):
    """Run training benchmark."""
    logger.info("=" * 80)
    logger.info("TRAINING BENCHMARK")
    logger.info("=" * 80)

    benchmark = TrainingBenchmark()
    results = benchmark.benchmark_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
    )

    return results


def benchmark_inference(model, test_loader, device, n_runs=3):
    """Run inference benchmark."""
    logger.info("=" * 80)
    logger.info("INFERENCE BENCHMARK")
    logger.info("=" * 80)

    benchmark = InferenceBenchmark()

    # Benchmark without mixed precision
    logger.info("\nBenchmarking without mixed precision...")
    results_fp32 = benchmark.benchmark_inference(
        model=model, dataloader=test_loader, device=device, n_runs=n_runs, use_mixed_precision=False
    )

    # Benchmark with mixed precision if CUDA available
    results_fp16 = None
    if "cuda" in device:
        logger.info("\nBenchmarking with mixed precision...")
        results_fp16 = benchmark.benchmark_inference(
            model=model,
            dataloader=test_loader,
            device=device,
            n_runs=n_runs,
            use_mixed_precision=True,
        )

        # Compare results
        if results_fp16:
            speedup = results_fp16["samples_per_second"] / results_fp32["samples_per_second"]
            logger.info(f"\nMixed precision speedup: {speedup:.2f}x")

    return {"fp32": results_fp32, "fp16": results_fp16}


def save_results(results, output_dir):
    """Save benchmark results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    # Create summary DataFrame
    summary_data = []

    if "featurization" in results:
        feat_results = results["featurization"]
        if "graph_featurization" in feat_results:
            summary_data.append(
                {
                    "Benchmark": "Graph Featurization",
                    "Mean Time (s)": feat_results["graph_featurization"]["mean_time"],
                    "Throughput": f"{feat_results['graph_featurization']['molecules_per_second']:.2f} mol/s",
                    "Details": f"±{feat_results['graph_featurization']['std_time']:.2f}s",
                }
            )

        if "fingerprint_generation" in feat_results:
            summary_data.append(
                {
                    "Benchmark": "Fingerprint Generation",
                    "Mean Time (s)": feat_results["fingerprint_generation"]["mean_time"],
                    "Throughput": f"{feat_results['fingerprint_generation']['molecules_per_second']:.2f} mol/s",
                    "Details": f"±{feat_results['fingerprint_generation']['std_time']:.2f}s",
                }
            )

    if "training" in results:
        train_results = results["training"]
        summary_data.append(
            {
                "Benchmark": "Training (per epoch)",
                "Mean Time (s)": train_results["mean_epoch_time"],
                "Throughput": f"{train_results['samples_per_second']:.2f} samples/s",
                "Details": f"±{train_results['std_epoch_time']:.2f}s",
            }
        )

    if "inference" in results:
        inf_results = results["inference"]
        if "fp32" in inf_results:
            summary_data.append(
                {
                    "Benchmark": "Inference (FP32)",
                    "Mean Time (s)": inf_results["fp32"]["mean_run_time"],
                    "Throughput": f"{inf_results['fp32']['samples_per_second']:.2f} samples/s",
                    "Details": f"Latency: {inf_results['fp32']['latency_ms']:.2f}ms",
                }
            )

        if "fp16" in inf_results and inf_results["fp16"]:
            summary_data.append(
                {
                    "Benchmark": "Inference (FP16)",
                    "Mean Time (s)": inf_results["fp16"]["mean_run_time"],
                    "Throughput": f"{inf_results['fp16']['samples_per_second']:.2f} samples/s",
                    "Details": f"Latency: {inf_results['fp16']['latency_ms']:.2f}ms",
                }
            )

    # Save summary as CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = output_path / "benchmark_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(index=False))
        logger.info("\n" + "=" * 80)


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logger(name="lifespan_predictor", level=logging.INFO)

    logger.info("Starting performance benchmarks")
    logger.info(f"Device: {args.device}")
    logger.info(f"Number of molecules: {args.n_molecules}")
    logger.info(f"Number of runs: {args.n_runs}")

    # Load configuration
    config = Config.from_yaml(args.config)

    # Load data
    data_path = args.data if args.data else config.data.train_csv
    logger.info(f"Loading data from {data_path}")

    df = load_and_clean_csv(
        data_path, smiles_column=config.data.smiles_column, label_column=config.data.label_column
    )

    # Limit to n_molecules
    if len(df) > args.n_molecules:
        df = df.head(args.n_molecules)
        logger.info(f"Limited dataset to {args.n_molecules} molecules")

    smiles_list = df[config.data.smiles_column].tolist()

    # Store all results
    all_results = {}

    # Run featurization benchmark
    if not args.skip_featurization:
        try:
            feat_results = benchmark_featurization(
                smiles_list=smiles_list, cache_dir="benchmark_cache", n_runs=args.n_runs
            )
            all_results["featurization"] = feat_results
        except Exception as e:
            logger.error(f"Featurization benchmark failed: {e}", exc_info=True)

    # For training and inference benchmarks, we need a model
    if not args.skip_training or not args.skip_inference:
        logger.info("Initializing model for training/inference benchmarks")

        # Create a small model for benchmarking
        _ = LifespanPredictor(config)  # model not used in this example

        # Create dummy dataset (we don't need real features for benchmarking)
        # Just measure the speed of the training/inference loop
        logger.info("Note: Using dummy data for training/inference benchmarks")
        logger.info("For accurate results, use real preprocessed data")

    # Run training benchmark
    if not args.skip_training:
        try:
            # Create dummy dataloader
            # In practice, you would use real data here
            logger.warning("Training benchmark requires real preprocessed data")
            logger.warning("Skipping training benchmark (use real data for accurate results)")
        except Exception as e:
            logger.error(f"Training benchmark failed: {e}", exc_info=True)

    # Run inference benchmark
    if not args.skip_inference:
        try:
            # Create dummy dataloader
            # In practice, you would use real data here
            logger.warning("Inference benchmark requires real preprocessed data")
            logger.warning("Skipping inference benchmark (use real data for accurate results)")
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}", exc_info=True)

    # Save results
    if all_results:
        save_results(all_results, args.output_dir)
        logger.info(f"\nBenchmark results saved to {args.output_dir}")
    else:
        logger.warning("No benchmark results to save")

    logger.info("\nBenchmarking completed!")


if __name__ == "__main__":
    main()
