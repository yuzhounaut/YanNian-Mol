# Performance Optimization Quick Reference

## Memory Optimizations

### GPU Memory Monitoring
```python
from lifespan_predictor.utils.memory import GPUMemoryMonitor

monitor = GPUMemoryMonitor('cuda:0', threshold=0.8)
monitor.log_memory_usage()
if monitor.is_memory_critical():
    monitor.clear_cache()
```

### Memory-Mapped Arrays
```python
from lifespan_predictor.utils.memory import MemoryMappedArray

with MemoryMappedArray(shape=(10000, 200, 75), dtype=np.float32) as mmap:
    mmap.array[0] = features
    data = mmap.array[0:100]
```

### Adaptive Batch Sizing
```python
from lifespan_predictor.utils.memory import AdaptiveBatchSizer

sizer = AdaptiveBatchSizer(initial_batch_size=32, min_batch_size=4)
try:
    # Training code
    pass
except RuntimeError as e:
    if "out of memory" in str(e):
        new_batch_size = sizer.reduce_batch_size()
```

### Cleanup Utilities
```python
from lifespan_predictor.utils.memory import cleanup_temp_files, cleanup_cache_directory

cleanup_temp_files("/tmp", "*.mmap")
cleanup_cache_directory("cache/features", max_size_gb=10.0)
```

## Computation Optimizations

### Timing Operations
```python
from lifespan_predictor.utils.profiling import timer

with timer("Data loading"):
    data = load_data()
```

### Function Profiling
```python
from lifespan_predictor.utils.profiling import profile_function

@profile_function
def expensive_operation():
    # Your code
    pass
```

### Performance Monitoring
```python
from lifespan_predictor.utils.profiling import PerformanceMonitor

monitor = PerformanceMonitor("Training")
monitor.start("epoch")
# ... code ...
monitor.stop("epoch")
monitor.log_summary()
```

### TorchScript Optimization
```python
from lifespan_predictor.utils.profiling import TorchJITOptimizer

optimized_model = TorchJITOptimizer.optimize_model(model, example_input)
```

### cuDNN Benchmark
```python
from lifespan_predictor.utils.profiling import enable_cudnn_benchmark

enable_cudnn_benchmark()  # Call at start of training
```

## Benchmarking

### Featurization Benchmark
```python
from lifespan_predictor.utils.benchmark import FeaturizationBenchmark

benchmark = FeaturizationBenchmark("cache")
results = benchmark.run_full_benchmark(smiles_list, n_runs=3)
print(f"Speed: {results['total']['molecules_per_second']:.2f} mol/s")
```

### Training Benchmark
```python
from lifespan_predictor.utils.benchmark import TrainingBenchmark

benchmark = TrainingBenchmark()
results = benchmark.benchmark_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda:0',
    n_epochs=3
)
print(f"Speed: {results['samples_per_second']:.2f} samples/s")
```

### Inference Benchmark
```python
from lifespan_predictor.utils.benchmark import InferenceBenchmark

benchmark = InferenceBenchmark()
results = benchmark.benchmark_inference(
    model=model,
    dataloader=test_loader,
    device='cuda:0',
    n_runs=3,
    use_mixed_precision=True
)
print(f"Latency: {results['latency_ms']:.2f}ms")
```

### Compare Implementations
```python
from lifespan_predictor.utils.benchmark import compare_implementations

comparison = compare_implementations(
    results_new=new_results,
    results_old=old_results,
    metric_name='molecules_per_second'
)
print(f"Speedup: {comparison['speedup']:.2f}x")
```

## Model Optimizations

### Gradient Checkpointing
```python
from lifespan_predictor.models.predictor import LifespanPredictor

model = LifespanPredictor(config, use_gradient_checkpointing=True)
```

### Memory-Mapped Featurization
```python
from lifespan_predictor.data.featurizers import CachedGraphFeaturizer

featurizer = CachedGraphFeaturizer(
    cache_dir="cache",
    use_memory_mapping=True
)
```

## Command-Line Tools

### Run All Benchmarks
```bash
python scripts/run_benchmarks.py --config config/default_config.yaml --n-molecules 1000
```

### Run Specific Benchmarks
```bash
# Only featurization
python scripts/run_benchmarks.py --skip-training --skip-inference

# Only training
python scripts/run_benchmarks.py --skip-featurization --skip-inference

# Only inference
python scripts/run_benchmarks.py --skip-featurization --skip-training
```

### Custom Settings
```bash
python scripts/run_benchmarks.py \
    --config config/default_config.yaml \
    --n-molecules 1000 \
    --n-runs 5 \
    --output-dir benchmark_results \
    --device cuda:0
```

## Best Practices

1. **Always warm up** before benchmarking (run a few iterations first)
2. **Use multiple runs** and report mean ± std
3. **Monitor GPU memory** during training to prevent OOM
4. **Enable mixed precision** for faster training on modern GPUs
5. **Use memory mapping** for large datasets
6. **Profile first** before optimizing
7. **Benchmark regularly** to track performance improvements
8. **Clean up temp files** after processing

## Performance Targets

- **Featurization**: 16.7+ molecules/second (1000 in < 60s)
- **Training**: < 5 minutes per epoch on GPU
- **Memory**: < 8GB GPU memory for batch size 32
- **Inference**: 33.3+ samples/second (1000 in < 30s)

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use memory-mapped arrays
- Clear GPU cache between epochs

### Slow Performance
- Enable mixed precision training
- Use cuDNN benchmark mode
- Profile to find bottlenecks
- Increase batch size (if memory allows)
- Use parallel processing for featurization

### High Memory Usage
- Monitor with GPUMemoryMonitor
- Use adaptive batch sizing
- Enable gradient checkpointing
- Clean up temporary files
- Limit cache directory size
