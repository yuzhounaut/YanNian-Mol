# Design Document

## Overview

This document describes the architectural design for optimizing the lifespan prediction codebase. The optimization will transform the existing notebook-based implementation into a modular, efficient, and maintainable Python package while preserving all functionality and improving performance.

## Architecture

### High-Level Architecture

```
lifespan_predictor/
├── config/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── default_config.yaml    # Default configuration
├── data/
│   ├── __init__.py
│   ├── dataset.py             # PyG Dataset classes
│   ├── featurizers.py         # Molecular featurization
│   ├── fingerprints.py        # Fingerprint generation
│   └── preprocessing.py       # Data cleaning and validation
├── models/
│   ├── __init__.py
│   ├── attentive_fp.py        # AttentiveFP GNN module
│   ├── predictor.py           # Main predictor model
│   └── layers.py              # Custom neural network layers
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training loop and logic
│   ├── metrics.py             # Evaluation metrics
│   └── callbacks.py           # Training callbacks (early stopping, etc.)
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Logging utilities
│   ├── io.py                  # File I/O operations
│   └── visualization.py       # Plotting and visualization
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference.ipynb
└── tests/
    ├── __init__.py
    ├── test_featurizers.py
    ├── test_models.py
    └── test_training.py
```

### Design Principles

1. **Separation of Concerns**: Each module handles a specific aspect of the pipeline
2. **Dependency Injection**: Configuration and dependencies passed explicitly
3. **Lazy Evaluation**: Compute features only when needed
4. **Caching**: Store intermediate results to avoid recomputation
5. **Fail-Fast**: Validate inputs early and provide clear error messages

## Components and Interfaces

### 1. Configuration Management (`config/config.py`)

**Purpose**: Centralize all configuration parameters and provide validation

**Key Classes**:

```python
class Config:
    """Main configuration class with validation"""
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config'
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config'
    
    def validate(self) -> None
    
    def to_dict(self) -> dict
    
    def save(self, path: str) -> None
```

**Configuration Structure**:
- `data`: Paths to train/test CSV, processed features, output directories
- `featurization`: Parameters for graph features and fingerprints
- `model`: Architecture hyperparameters (GNN layers, dimensions, dropout)
- `training`: Batch size, learning rate, epochs, early stopping
- `device`: CUDA/CPU settings
- `logging`: Verbosity and log file paths

**Design Decisions**:
- Use Pydantic for validation and type checking
- Support environment variable expansion in paths
- Compute derived values (e.g., total fingerprint dimension) automatically
- Provide sensible defaults for all optional parameters

### 2. Data Preprocessing (`data/preprocessing.py`)

**Purpose**: Clean and validate SMILES strings before featurization

**Key Functions**:

```python
def clean_smiles(smiles: str) -> Optional[str]:
    """Clean and canonicalize SMILES string"""
    
def validate_smiles_list(smiles_list: List[str]) -> Tuple[List[str], List[int]]:
    """Validate list of SMILES, return valid ones and failed indices"""
    
def load_and_clean_csv(
    csv_path: str,
    smiles_column: str = 'SMILES',
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """Load CSV and clean SMILES strings"""
```

**Design Decisions**:
- Use RDKit's SaltRemover for standardization
- Canonicalize SMILES to ensure consistency
- Return both valid data and indices of failures for tracking
- Log warnings for invalid molecules but continue processing

### 3. Molecular Featurization (`data/featurizers.py`)

**Purpose**: Convert SMILES to graph features with caching

**Key Classes**:

```python
class CachedGraphFeaturizer:
    """Graph featurizer with disk caching"""
    
    def __init__(
        self,
        cache_dir: str,
        max_atoms: int = 200,
        atom_feature_dim: int = 75
    )
    
    def featurize(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray] = None,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Featurize molecules, using cache when available"""
    
    def _compute_features(self, mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        """Compute adjacency matrix and node features for a molecule"""
```

**Design Decisions**:
- Cache features using hash of SMILES list as key
- Store metadata (SMILES, timestamps) alongside features
- Support parallel processing using multiprocessing
- Validate cache integrity before loading
- Use memory-mapped arrays for large feature matrices

### 4. Fingerprint Generation (`data/fingerprints.py`)

**Purpose**: Generate molecular fingerprints efficiently

**Key Classes**:

```python
class FingerprintGenerator:
    """Unified fingerprint generator with batching"""
    
    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
        rdkit_fp_nbits: int = 2048,
        n_jobs: int = -1
    )
    
    def generate_fingerprints(
        self,
        smiles_list: List[str],
        cache_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate hashed and non-hashed fingerprints"""
    
    def _batch_compute_morgan(
        self,
        mols: List[Chem.Mol]
    ) -> np.ndarray:
        """Compute Morgan fingerprints in batch"""
```

**Design Decisions**:
- Batch process molecules to reduce overhead
- Use joblib for parallel processing
- Cache fingerprints separately from graph features
- Support incremental computation for large datasets
- Validate fingerprint dimensions match configuration

### 5. Dataset Classes (`data/dataset.py`)

**Purpose**: PyTorch Geometric dataset with efficient loading

**Key Classes**:

```python
class LifespanDataset(InMemoryDataset):
    """Optimized dataset for lifespan prediction"""
    
    def __init__(
        self,
        smiles_list: List[str],
        graph_features: Tuple[np.ndarray, np.ndarray],
        fingerprints: Tuple[np.ndarray, np.ndarray],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None
    )
    
    def _create_pyg_data(
        self,
        idx: int
    ) -> Data:
        """Create PyG Data object for a single molecule"""

class GraphDataBuilder:
    """Builder for DGL graph objects"""
    
    def __init__(self, use_edge_features: bool = True)
    
    def build_dgl_graph(
        self,
        smiles: str,
        node_features: np.ndarray
    ) -> dgl.DGLGraph:
        """Build DGL graph from SMILES and features"""
```

**Design Decisions**:
- Separate graph construction from dataset class
- Reuse featurizers across molecules
- Store DGL graphs as attributes of PyG Data objects
- Implement efficient collate function for batching
- Support on-the-fly graph construction for large datasets

### 6. Model Architecture (`models/`)

**Purpose**: Modular neural network components

**Key Classes**:

```python
# models/attentive_fp.py
class AttentiveFPModule(nn.Module):
    """AttentiveFP GNN using DGL-LifeSci"""
    
    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        num_layers: int,
        num_timesteps: int,
        graph_feat_size: int,
        dropout: float
    )
    
    def forward(
        self,
        dgl_graphs: List[dgl.DGLGraph]
    ) -> torch.Tensor:
        """Forward pass returning graph embeddings"""

# models/predictor.py
class LifespanPredictor(nn.Module):
    """Main predictor combining GNN, CNN, and DNN"""
    
    def __init__(self, config: Config)
    
    def forward(
        self,
        batch: Data
    ) -> torch.Tensor:
        """Forward pass returning predictions"""
    
    def _forward_gnn(self, batch: Data) -> torch.Tensor:
        """GNN branch for graph features"""
    
    def _forward_fp_cnn(self, batch: Data) -> torch.Tensor:
        """CNN branch for MACCS fingerprints"""
    
    def _forward_fp_dnn(self, batch: Data) -> torch.Tensor:
        """DNN branch for Morgan/RDKit fingerprints"""
```

**Design Decisions**:
- Separate each model component into its own method
- Support disabling individual branches via configuration
- Use nn.ModuleDict for dynamic branch selection
- Implement proper weight initialization
- Add gradient checkpointing for memory efficiency

### 7. Training Infrastructure (`training/`)

**Purpose**: Robust training loop with monitoring

**Key Classes**:

```python
# training/trainer.py
class Trainer:
    """Training orchestrator with callbacks"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None
    )
    
    def train(self) -> Dict[str, Any]:
        """Run training loop"""
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
    
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint with metadata"""

# training/callbacks.py
class Callback:
    """Base callback class"""
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        pass

class EarlyStopping(Callback):
    """Early stopping based on validation metric"""
    
class ModelCheckpoint(Callback):
    """Save best model based on metric"""
    
class LearningRateScheduler(Callback):
    """Adjust learning rate during training"""
```

**Design Decisions**:
- Use callback pattern for extensibility
- Separate training logic from model definition
- Support mixed precision training with torch.cuda.amp
- Implement gradient accumulation for large batches
- Log metrics to TensorBoard and CSV
- Save full training state for resuming

### 8. Metrics (`training/metrics.py`)

**Purpose**: Evaluation metrics for classification and regression

**Key Classes**:

```python
class Metric(ABC):
    """Base metric class"""
    
    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    def higher_is_better(self) -> bool:
        return True

class MetricCollection:
    """Collection of metrics for evaluation"""
    
    def __init__(self, metrics: List[Metric])
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute all metrics"""
```

**Specific Metrics**:
- Classification: AUC, Accuracy, F1, Precision, Recall
- Regression: RMSE, MAE, R2, Pearson Correlation

**Design Decisions**:
- Use abstract base class for consistency
- Support both numpy and torch tensors
- Handle edge cases (e.g., single class in batch)
- Provide clear error messages for invalid inputs

### 9. Utilities (`utils/`)

**Purpose**: Cross-cutting concerns

**Key Modules**:

```python
# utils/logging.py
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with file and console handlers"""

# utils/io.py
def save_results(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = "results"
) -> None:
    """Save results with metadata"""

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load model checkpoint"""

# utils/visualization.py
def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str
) -> None:
    """Plot training and validation metrics"""

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    task: str = "classification"
) -> None:
    """Plot predictions vs ground truth"""
```

**Design Decisions**:
- Use Python logging module instead of print
- Include timestamps and metadata in all outputs
- Support both JSON and pickle for serialization
- Generate publication-quality plots with seaborn
- Provide utility functions for common operations

## Data Models

### Configuration Schema

```yaml
data:
  train_csv: "train.csv"
  test_csv: "test.csv"
  smiles_column: "SMILES"
  label_column: "Life_extended"
  graph_features_dir: "processed_graph_features"
  fingerprints_dir: "processed_fingerprints"
  output_dir: "results"

featurization:
  max_atoms: 200
  atom_feature_dim: 75
  morgan_radius: 2
  morgan_nbits: 2048
  rdkit_fp_nbits: 2048
  maccs_nbits: 166
  use_cache: true
  n_jobs: -1

model:
  enable_gnn: true
  enable_fp_dnn: true
  enable_fp_cnn: true
  gnn_node_input_dim: 78
  gnn_edge_input_dim: 11
  gnn_graph_embed_dim: 128
  gnn_num_layers: 2
  gnn_num_timesteps: 2
  gnn_dropout: 0.5
  fp_cnn_output_dim: 64
  fp_dnn_layers: [256, 128]
  fp_dnn_output_dim: 64
  fp_dropout: 0.5
  n_output_tasks: 1

training:
  task: "classification"  # or "regression"
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  patience: 15
  gradient_clip: 1.0
  use_mixed_precision: true
  val_split: 0.3
  stratify: true
  main_metric: "AUC"  # or "RMSE" for regression

device:
  use_cuda: true
  device_id: 0

logging:
  level: "INFO"
  log_file: "training.log"
  tensorboard_dir: "runs"
  print_every_n_epochs: 5

random_seed: 42
```

### Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,  # if using scheduler
    'train_metrics': Dict[str, List[float]],
    'val_metrics': Dict[str, List[float]],
    'config': dict,
    'timestamp': str,
    'git_commit': str,  # if available
    'package_versions': dict
}
```

## Error Handling

### Error Hierarchy

```python
class LifespanPredictorError(Exception):
    """Base exception for lifespan predictor"""

class ConfigurationError(LifespanPredictorError):
    """Configuration validation errors"""

class FeaturizationError(LifespanPredictorError):
    """Errors during molecular featurization"""

class DataError(LifespanPredictorError):
    """Data loading and validation errors"""

class ModelError(LifespanPredictorError):
    """Model architecture and forward pass errors"""

class TrainingError(LifespanPredictorError):
    """Training loop errors"""
```

### Error Handling Strategy

1. **Validation Errors**: Fail fast with clear messages
2. **Data Errors**: Log and skip invalid samples, continue processing
3. **Training Errors**: Save checkpoint and raise with context
4. **Resource Errors**: Provide suggestions (e.g., reduce batch size)

## Testing Strategy

### Unit Tests

- Test each featurizer independently with known molecules
- Test fingerprint generation with reference implementations
- Test model forward pass with dummy data
- Test metric calculations with known inputs/outputs
- Test configuration validation with valid/invalid configs

### Integration Tests

- Test full preprocessing pipeline end-to-end
- Test training loop with small dataset
- Test checkpoint saving and loading
- Test inference on new data

### Performance Tests

- Benchmark featurization speed
- Measure memory usage during training
- Profile bottlenecks with cProfile

### Test Data

- Use small subset of real data (10-20 molecules)
- Include edge cases (single atom, large molecules)
- Test with invalid SMILES strings

## Migration Strategy

### Phase 1: Extract Core Functionality

1. Create package structure
2. Extract configuration management
3. Extract data preprocessing functions
4. Extract featurization classes
5. Add unit tests for extracted code

### Phase 2: Refactor Models

1. Extract model classes to separate files
2. Refactor forward pass into modular methods
3. Add model unit tests
4. Verify outputs match original implementation

### Phase 3: Refactor Training

1. Extract training loop to Trainer class
2. Implement callback system
3. Add metrics module
4. Test training with small dataset

### Phase 4: Create New Notebooks

1. Create simplified notebooks using new package
2. Add documentation and examples
3. Verify results match original notebooks
4. Archive original notebooks

### Phase 5: Optimization

1. Add caching to featurizers
2. Implement parallel processing
3. Add mixed precision training
4. Profile and optimize bottlenecks

## Performance Targets

- **Featurization**: Process 1000 molecules in < 60 seconds
- **Training**: Train one epoch in < 5 minutes on GPU
- **Memory**: Peak GPU memory < 8GB for batch size 32
- **Inference**: Predict 1000 molecules in < 30 seconds

## Dependencies

### Core Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- DGL >= 1.0
- DGL-LifeSci >= 0.3
- RDKit >= 2022.09
- DeepChem >= 2.7
- NumPy >= 1.23
- Pandas >= 1.5
- scikit-learn >= 1.2

### Additional Dependencies

- Pydantic >= 2.0 (configuration validation)
- PyYAML >= 6.0 (configuration files)
- tqdm >= 4.65 (progress bars)
- tensorboard >= 2.12 (logging)
- matplotlib >= 3.7 (visualization)
- seaborn >= 0.12 (visualization)
- joblib >= 1.2 (parallel processing)

### Development Dependencies

- pytest >= 7.3
- pytest-cov >= 4.0
- black >= 23.0 (code formatting)
- flake8 >= 6.0 (linting)
- mypy >= 1.0 (type checking)
- sphinx >= 6.0 (documentation)

## Documentation

### Code Documentation

- Docstrings for all public functions and classes
- Type hints for all function signatures
- Inline comments for complex logic
- Examples in docstrings

### User Documentation

- README with installation and quick start
- Tutorial notebooks for common workflows
- API reference generated from docstrings
- Configuration guide with all options explained
- Troubleshooting guide for common issues

### Developer Documentation

- Architecture overview (this document)
- Contributing guidelines
- Testing guidelines
- Release process
