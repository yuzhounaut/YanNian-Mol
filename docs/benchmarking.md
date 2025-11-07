# Performance Benchmarking Guide

This guide explains how to use the benchmarking tools to measure and optimize the performance of the lifespan predictor.

## Overview

The lifespan predictor includes comprehensive benchmarking tools to measure:

1. **Featurization Speed**: How fast molecular features and fingerprints are computed
2. **Training Speed**: How fast the model trains per epoch
3. **Inference Speed**: How fast predictions are made on new data

## Quick Start

### Running All Benchmarks

```bash
python scripts/run_benchmarks.py --config config/default_config.yaml --n-molecules 1000
```

### Running Specific Benchmarks

```bash
# Only featurization
python scripts/run_benchmarks.py --skip-training --skip-inference

# Only training
python scripts/run_benchmarks.py --skip-featurization --skip-inference

# Only inference
python scripts/run_benchmarks.py --skip-featurization --skip-training
```

## Featurization Benchmark

Measures the speed of converting SMILES strings to molecular features.

### Usage in Code

```python
from lifespan_predictor.utils.benchmark import FeaturizationBenchmark

# Initialize benchmark
benchmark = FeaturizationBenchmark(cache_dir="benchmark_cache")

# Run benchmark
results = benchmark.run_full_benchmark(
    smiles_list=smiles_list,
    n_runs=3,
    n_jobs=-1  # Use all CPU cores
)

# Print results
print(f"Graph featurization: {results['graph_featurization']['molecules_per_second']:.2f} mol/s")
print(f"Fingerprint generation: {results['fingerprint_generation']['molecules_per_second']:.2f} mol/s")
```

### Metrics

- **molecules_per_second**: Throughput in molecules processed per second
- **mean_time**: Average time to process all molecules
- **std_time**: Standard deviation of processing time
- **time_per_molecule**: Average time per molecule in milliseconds

### Optimization Tips

1. **Use parallel processing**: Set `n_jobs=-1` to use all CPU cores
2. **Enable caching**: Reuse computed features when possible
3. **Batch processing**: Process molecules in batches rather than one at a time
4. **Memory mapping**: Use memory-mapped arrays for large datasets

## Training Benchmark

Measures the speed of model training.

### Usage in Code

```python
from lifespan_predictor.utils.benchmark import TrainingBenchmark

# Initialize benchmark
benchmark = TrainingBenchmark()

# Run benchmark
results = benchmark.benchmark_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda:0',
    n_epochs=3
)

# Print results
print(f"Training speed: {results['samples_per_second']:.2f} samples/s")
print(f"Time per epoch: {results['mean_epoch_time']:.2f}s")
```

### Metrics

- **samples_per_second**: Training throughput in samples per second
- **mean_epoch_time**: Average time per epoch
- **mean_batch_time**: Average time per batch
- **batches_per_second**: Number of batches processed per second

### Optimization Tips

1. **Use mixed precision**: Enable `use_mixed_precision=True` for faster training on GPUs
2. **Optimize batch size**: Find the largest batch size that fits in GPU memory
3. **Enable cuDNN benchmark**: Call `enable_cudnn_benchmark()` for faster convolutions
4. **Use gradient checkpointing**: Trade compute for memory with `use_gradient_checkpointing=True`
5. **Monitor GPU memory**: Use `GPUMemoryMonitor` to track memory usage

## Inference Benchmark

Measures the speed of making predictions.

### Usage in Code

```python
from lifespan_predictor.utils.benchmark import InferenceBenchmark

# Initialize benchmark
benchmark = InferenceBenchmark()

# Run benchmark
results = benchmark.benchmark_inference(
    model=model,
    dataloader=test_loader,
    device='cuda:0',
    n_runs=3,
    use_mixed_precision=True
)

# Print results
print(f"Inference speed: {results['samples_per_second']:.2f} samples/s")
print(f"Latency: {results['latency_ms']:.2f}ms per sample")
```

### Metrics

- **samples_per_second**: Inference throughput in samples per second
- **latency_ms**: Average time per sample in milliseconds
- **mean_run_time**: Average time to process all samples
- **mean_batch_time**: Average time per batch

### Optimization Tips

1. **Use mixed precision**: Enable `use_mixed_precision=True` for faster inference
2. **Batch predictions**: Process multiple samples at once
3. **Use TorchScript**: Compile model with `torch.jit.trace` or `torch.jit.script`
4. **Optimize batch size**: Find optimal batch size for throughput vs latency tradeoff

## Comparing Implementations

Compare performance between different implementations:

```python
from lifespan_predictor.utils.benchmark import compare_implementations

# Run benchmarks for both implementations
results_new = benchmark.run(...)
results_old = benchmark.run(...)

# Compare
comparison = compare_implementations(
    results_new=results_new,
    results_old=results_old,
    metric_name='molecules_per_second'
)

print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Improvement: {comparison['improvement_pct']:+.1f}%")
```

## Memory Optimization

### GPU Memory Monitoring

Monitor GPU memory usage during training:

```python
from lifespan_predictor.utils.memory import GPUMemoryMonitor

# Initialize monitor
monitor = GPUMemoryMonitor(device='cuda:0', threshold=0.8)

# Log memory usage
monitor.log_memory_usage("Before training")

# Check if memory is critical
if monitor.is_memory_critical():
    monitor.clear_cache()
```

### Adaptive Batch Sizing

Automatically reduce batch size on out-of-memory errors:

```python
from lifespan_predictor.utils.memory import AdaptiveBatchSizer

# Initialize sizer
sizer = AdaptiveBatchSizer(
    initial_batch_size=32,
    min_batch_size=4,
    reduction_factor=0.5
)

# Training loop with OOM handling
try:
    # Training code
    pass
except RuntimeError as e:
    if "out of memory" in str(e):
        new_batch_size = sizer.reduce_batch_size()
        # Recreate dataloader with new batch size
```

### Memory-Mapped Arrays

Use memory-mapped arrays for large feature matrices:

```python
from lifespan_predictor.utils.memory import MemoryMappedArray

# Create memory-mapped array
with MemoryMappedArray(shape=(10000, 200, 75), dtype=np.float32) as mmap_array:
    # Write data
    mmap_array.array[0] = features
    
    # Read data
    data = mmap_array.array[0:100]
    
    # Automatically cleaned up on exit
```

## Profiling

### Function Profiling

Profile individual functions to identify bottlenecks:

```python
from lifespan_predictor.utils.profiling import profile_function

@profile_function
def expensive_operation():
    # Some computation
    pass

# Function will log profiling stats when called
expensive_operation()
```

### Performance Monitoring

Track performance metrics during execution:

```python
from lifespan_predictor.utils.profiling import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor("Training")

# Time operations
monitor.start("data_loading")
# ... load data ...
monitor.stop("data_loading")

monitor.start("forward_pass")
# ... forward pass ...
monitor.stop("forward_pass")

# Log summary
monitor.log_summary()
```

### Timing Context Manager

Time code blocks easily:

```python
from lifespan_predictor.utils.profiling import timer

with timer("Data preprocessing"):
    # Preprocessing code
    pass
# Logs: "Data preprocessing completed in X.XXs"
```

## Best Practices

1. **Warm up**: Run a few iterations before benchmarking to warm up caches
2. **Multiple runs**: Run benchmarks multiple times and report mean ± std
3. **Consistent environment**: Use the same hardware and software versions
4. **Isolate benchmarks**: Close other applications to reduce noise
5. **Monitor resources**: Track CPU, GPU, and memory usage during benchmarks
6. **Document results**: Save benchmark results with timestamps and configuration

## Example Benchmark Report

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Benchmark                    Mean Time (s)    Throughput              Details
Graph Featurization          12.34           81.03 mol/s             ±0.45s
Fingerprint Generation       3.21            311.53 mol/s            ±0.12s
Training (per epoch)         45.67           701.23 samples/s        ±1.23s
Inference (FP32)             8.91            1123.45 samples/s       Latency: 0.89ms
Inference (FP16)             5.43            1842.71 samples/s       Latency: 0.54ms

================================================================================
```

## Troubleshooting

### Benchmark Results Vary Widely

- Ensure consistent environment (close other applications)
- Increase number of runs (`n_runs`)
- Check for thermal throttling on GPU
- Verify data is cached/loaded before timing

### Out of Memory During Benchmarking

- Reduce batch size
- Use smaller dataset for benchmarking
- Enable gradient checkpointing
- Use memory-mapped arrays

### Slow Featurization

- Enable parallel processing (`n_jobs=-1`)
- Check if RDKit is using all CPU cores
- Verify molecules are valid (invalid molecules slow down processing)
- Use caching to avoid recomputation

### Slow Training

- Enable mixed precision training
- Optimize batch size
- Check data loading is not the bottleneck
- Profile to identify slow operations
- Enable cuDNN benchmark mode

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Python Profiling](https://docs.python.org/3/library/profile.html)
