# Lifespan Predictor

A modular, efficient machine learning pipeline for predicting compound effects on *C. elegans* lifespan using multi-modal deep learning with graph neural networks, molecular fingerprints, and various featurization techniques.

## Features

- **Multi-Modal Architecture**: Combines Graph Neural Networks (AttentiveFP), CNNs, and DNNs for comprehensive molecular analysis
- **Efficient Caching**: Disk-based caching for molecular features and fingerprints to avoid redundant computations
- **Parallel Processing**: Multi-core support for featurization and fingerprint generation
- **Flexible Configuration**: YAML-based configuration with validation using Pydantic
- **Comprehensive Metrics**: Support for both classification and regression tasks with multiple evaluation metrics
- **Training Callbacks**: Early stopping, model checkpointing, and learning rate scheduling
- **Mixed Precision Training**: Automatic mixed precision (AMP) support for faster training on modern GPUs
- **Visualization**: Built-in plotting utilities for training curves, predictions, and ROC curves

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-capable GPU (optional, but recommended for training)

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd lifespan_predictor

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Dependencies

Core dependencies:
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- DGL >= 1.0
- DGL-LifeSci >= 0.3
- RDKit >= 2022.09
- DeepChem >= 2.7
- Pydantic >= 2.0
- NumPy, Pandas, scikit-learn

See `requirements.txt` for the complete list.

## Quick Start

### 1. Prepare Your Data

Prepare a CSV file with SMILES strings and labels:

```csv
SMILES,Life_extended
CCO,0
c1ccccc1,1
CC(C)C,0
```

### 2. Configure the Pipeline

Create or modify `lifespan_predictor/config/default_config.yaml`:

```yaml
data:
  train_csv: "train.csv"
  test_csv: "test.csv"
  smiles_column: "SMILES"
  label_column: "Life_extended"

training:
  task: "classification"  # or "regression"
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.0001

model:
  enable_gnn: true
  enable_fp_dnn: true
  enable_fp_cnn: true
```

### 3. Run the Pipeline

Use the provided notebooks for a complete workflow:

```python
# See notebooks/01_data_preprocessing.ipynb for data preparation
# See notebooks/02_model_training.ipynb for training
# See notebooks/03_inference.ipynb for predictions
```

Or use the Python API directly:

```python
from lifespan_predictor.config import Config
from lifespan_predictor.data import (
    load_and_clean_csv,
    CachedGraphFeaturizer,
    FingerprintGenerator,
    LifespanDataset,
    create_dataloader
)
from lifespan_predictor.models import LifespanPredictor
from lifespan_predictor.training import Trainer, EarlyStopping, ModelCheckpoint
from lifespan_predictor.training.metrics import MetricCollection, AUC, Accuracy

# Load configuration
config = Config.from_yaml("config/default_config.yaml")

# Load and clean data
df = load_and_clean_csv(
    config.data.train_csv,
    smiles_column=config.data.smiles_column,
    label_column=config.data.label_column
)

# Featurize molecules
featurizer = CachedGraphFeaturizer(
    cache_dir=config.data.graph_features_dir,
    max_atoms=config.featurization.max_atoms
)
adj, feat, sim, labels = featurizer.featurize(
    df[config.data.smiles_column].tolist(),
    labels=df[config.data.label_column].values
)

# Generate fingerprints
fp_gen = FingerprintGenerator(
    morgan_radius=config.featurization.morgan_radius,
    morgan_nbits=config.featurization.morgan_nbits
)
hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(
    df[config.data.smiles_column].tolist(),
    cache_dir=config.data.fingerprints_dir
)

# Create dataset
dataset = LifespanDataset(
    root="data/processed",
    smiles_list=df[config.data.smiles_column].tolist(),
    graph_features=(adj, feat, sim),
    fingerprints=(hashed_fps, non_hashed_fps),
    labels=labels
)

# Create data loaders
train_loader = create_dataloader(dataset, batch_size=config.training.batch_size)

# Initialize model
model = LifespanPredictor(config)

# Setup training
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)
criterion = torch.nn.BCEWithLogitsLoss()
metrics = MetricCollection([AUC(), Accuracy()])

# Train model
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    metrics=metrics,
    callbacks=[
        EarlyStopping(patience=config.training.patience),
        ModelCheckpoint(save_dir="checkpoints", monitor="val_AUC")
    ]
)

history = trainer.train()
```

## Usage Examples

### Data Preprocessing

```python
from lifespan_predictor.data import load_and_clean_csv, clean_smiles

# Load and clean CSV
df = load_and_clean_csv(
    "data.csv",
    smiles_column="SMILES",
    label_column="Activity",
    clean=True,
    drop_invalid=True
)

# Clean individual SMILES
canonical_smiles = clean_smiles("CCO.[Na+]")  # Returns "CCO"
```

### Molecular Featurization

```python
from lifespan_predictor.data import CachedGraphFeaturizer

# Initialize featurizer with caching
featurizer = CachedGraphFeaturizer(
    cache_dir="cache/features",
    max_atoms=200,
    n_jobs=-1  # Use all CPU cores
)

# Featurize molecules
smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]
adj, feat, sim, labels = featurizer.featurize(
    smiles_list,
    labels=np.array([0, 1, 0]),
    force_recompute=False  # Use cache if available
)
```

### Fingerprint Generation

```python
from lifespan_predictor.data import FingerprintGenerator

# Initialize generator
fp_gen = FingerprintGenerator(
    morgan_radius=2,
    morgan_nbits=2048,
    rdkit_fp_nbits=2048,
    n_jobs=-1
)

# Generate fingerprints
hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(
    smiles_list,
    cache_dir="cache/fingerprints"
)
```

### Model Configuration

```python
from lifespan_predictor.config import Config

# Load from YAML
config = Config.from_yaml("config.yaml")

# Create from dictionary
config = Config.from_dict({
    "model": {
        "enable_gnn": True,
        "enable_fp_dnn": True,
        "enable_fp_cnn": False  # Disable CNN branch
    }
})

# Save configuration
config.save("my_config.yaml")

# Get device
device = config.get_device()  # Returns "cuda:0" or "cpu"
```

### Custom Training Callbacks

```python
from lifespan_predictor.training.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] < 0.1:
            print(f"Target loss reached at epoch {epoch}!")
            return True  # Stop training
        return False

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    metrics=metrics,
    callbacks=[CustomCallback()]
)
```

## Project Structure

```
lifespan_predictor/
├── config/
│   ├── config.py              # Configuration management with Pydantic
│   └── default_config.yaml    # Default configuration
├── data/
│   ├── preprocessing.py       # SMILES cleaning and validation
│   ├── featurizers.py         # Molecular graph featurization
│   ├── fingerprints.py        # Fingerprint generation
│   └── dataset.py             # PyTorch Geometric datasets
├── models/
│   ├── attentive_fp.py        # AttentiveFP GNN module
│   ├── predictor.py           # Main predictor model
│   └── layers.py              # Custom layers and initialization
├── training/
│   ├── trainer.py             # Training orchestrator
│   ├── metrics.py             # Evaluation metrics
│   └── callbacks.py           # Training callbacks
├── utils/
│   ├── logging.py             # Logging utilities
│   ├── io.py                  # File I/O operations
│   └── visualization.py       # Plotting utilities
└── notebooks/
    ├── 01_data_preprocessing.ipynb
    ├── 02_model_training.ipynb
    └── 03_inference.ipynb
```

## Configuration Guide

See [docs/configuration.md](docs/configuration.md) for detailed configuration options.

Key configuration sections:

- **data**: Paths to data files and output directories
- **featurization**: Parameters for molecular featurization and fingerprints
- **model**: Model architecture hyperparameters
- **training**: Training parameters (batch size, learning rate, etc.)
- **device**: CUDA/CPU settings
- **logging**: Logging configuration

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in configuration
- Enable gradient checkpointing (if implemented)
- Use mixed precision training (enabled by default)
- Reduce model size (fewer layers, smaller dimensions)

```yaml
training:
  batch_size: 16  # Reduce from 32
  use_mixed_precision: true
```

#### 2. Slow Featurization

**Problem**: Molecular featurization takes too long

**Solutions**:
- Enable caching (enabled by default)
- Increase number of parallel jobs
- Use pre-computed features if available

```yaml
featurization:
  use_cache: true
  n_jobs: -1  # Use all CPU cores
```

#### 3. Invalid SMILES Strings

**Problem**: Many molecules fail to parse

**Solutions**:
- Enable SMILES cleaning and canonicalization
- Check input data quality
- Review preprocessing logs for specific errors

```python
df = load_and_clean_csv(
    "data.csv",
    clean=True,  # Enable cleaning
    drop_invalid=True  # Drop invalid molecules
)
```

#### 4. Model Not Learning

**Problem**: Training loss not decreasing

**Solutions**:
- Check learning rate (try 1e-4 to 1e-3)
- Verify data labels are correct
- Check for data leakage or imbalance
- Try different model configurations

```yaml
training:
  learning_rate: 0.001  # Increase if too slow
  gradient_clip: 1.0    # Prevent exploding gradients
```

#### 5. Import Errors

**Problem**: `ModuleNotFoundError` for dependencies

**Solutions**:
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version (>= 3.9 required)
- Verify virtual environment is activated

### Getting Help

- Check the [documentation](docs/)
- Review example notebooks in `notebooks/`
- Open an issue on GitHub with:
  - Error message and full traceback
  - Configuration file
  - Python and package versions
  - Steps to reproduce

## Performance Tips

1. **Use Caching**: Always enable caching for featurization and fingerprints
2. **Parallel Processing**: Set `n_jobs=-1` to use all CPU cores
3. **Mixed Precision**: Keep `use_mixed_precision=true` for GPU training
4. **Batch Size**: Use largest batch size that fits in GPU memory
5. **Data Loading**: Use `num_workers > 0` in DataLoader for faster data loading

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lifespan_predictor,
  title = {Lifespan Predictor: Multi-Modal Deep Learning for Molecular Property Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/lifespan_predictor}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

This project uses:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [DGL](https://www.dgl.ai/) and [DGL-LifeSci](https://lifesci.dgl.ai/)
- [RDKit](https://www.rdkit.org/)
- [DeepChem](https://deepchem.io/)
