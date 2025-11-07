# Notebooks

This directory contains Jupyter notebooks demonstrating the usage of the refactored `lifespan_predictor` package. These notebooks replace the original monolithic notebooks with a clean, modular approach.

## Notebooks

### 1. Data Preprocessing (`01_data_preprocessing.ipynb`)

This notebook demonstrates the complete data preprocessing pipeline:
- Loading and cleaning CSV data with SMILES strings
- Validating and canonicalizing molecular structures
- Generating graph features using the `CachedGraphFeaturizer`
- Computing molecular fingerprints (Morgan, RDKit, MACCS)
- Saving processed data for training

**Key Features:**
- Automatic caching to avoid recomputation
- Parallel processing for faster featurization
- Progress bars and detailed logging
- Robust error handling for invalid molecules

### 2. Model Training (`02_model_training.ipynb`)

This notebook shows how to train the lifespan prediction model:
- Loading configuration from YAML files
- Creating PyTorch Geometric datasets and dataloaders
- Initializing the multi-modal model architecture
- Training with callbacks (early stopping, checkpointing)
- Visualizing training curves and metrics
- Evaluating on validation data

**Key Features:**
- Configurable model architecture (GNN, CNN, DNN branches)
- Support for both classification and regression tasks
- Automatic mixed precision training
- Comprehensive metrics tracking
- Model checkpointing with best model selection

### 3. Inference (`03_inference.ipynb`)

This notebook demonstrates how to use a trained model for predictions:
- Loading trained models and configurations
- Processing new molecules through the pipeline
- Generating predictions with confidence scores
- Visualizing prediction distributions
- Saving results in multiple formats

**Key Features:**
- Easy loading of trained models
- Batch processing for efficiency
- Confidence estimation for predictions
- Multiple output formats (CSV, NumPy)
- Visualization of prediction distributions

## Getting Started

### Prerequisites

Make sure you have installed the `lifespan_predictor` package:

```bash
# From the project root directory
pip install -e .
```

### Required Data

Before running the notebooks, ensure you have:
1. Training data CSV with SMILES and labels
2. Test data CSV with SMILES (optional)
3. Configured paths in `lifespan_predictor/config/default_config.yaml`

### Running the Notebooks

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory

3. Run the notebooks in order:
   - Start with `01_data_preprocessing.ipynb` to process your data
   - Then run `02_model_training.ipynb` to train a model
   - Finally use `03_inference.ipynb` to make predictions

## Configuration

All notebooks use the configuration system from `lifespan_predictor.config`. You can:
- Modify `default_config.yaml` for global settings
- Create custom YAML files for different experiments
- Override settings programmatically in notebooks

## Advantages Over Original Notebooks

These refactored notebooks provide several improvements:

1. **Modularity**: Clean separation of concerns with reusable components
2. **Maintainability**: Easy to update and extend functionality
3. **Performance**: Built-in caching and parallel processing
4. **Robustness**: Comprehensive error handling and validation
5. **Reproducibility**: Configuration management and logging
6. **Documentation**: Clear structure with detailed explanations

## Troubleshooting

### Common Issues

**Import errors**: Make sure the package is installed with `pip install -e .`

**CUDA errors**: Set `device.use_cuda: false` in config if GPU is unavailable

**Memory errors**: Reduce `batch_size` or `n_jobs` in configuration

**Cache issues**: Delete cache directories and set `force_recompute: true`

## Additional Resources

- See `lifespan_predictor/config/default_config.yaml` for all configuration options
- Check `tests/` directory for usage examples
- Refer to the main README.md for package documentation
