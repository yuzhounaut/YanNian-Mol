# Configuration Guide

This guide provides detailed documentation for all configuration parameters in the Lifespan Predictor system.

## Table of Contents

- [Overview](#overview)
- [Configuration File Format](#configuration-file-format)
- [Data Configuration](#data-configuration)
- [Featurization Configuration](#featurization-configuration)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Device Configuration](#device-configuration)
- [Logging Configuration](#logging-configuration)
- [Common Scenarios](#common-scenarios)
- [Derived Parameters](#derived-parameters)

## Overview

The Lifespan Predictor uses YAML configuration files with Pydantic validation to ensure type safety and parameter correctness. Configuration files support:

- Environment variable expansion (e.g., `$HOME/data`)
- User home directory expansion (e.g., `~/data`)
- Automatic validation of parameter types and ranges
- Default values for all optional parameters

## Configuration File Format

Configuration files are written in YAML format:

```yaml
# config.yaml
data:
  train_csv: "train.csv"
  test_csv: "test.csv"

featurization:
  max_atoms: 200
  morgan_nbits: 2048

model:
  enable_gnn: true
  gnn_num_layers: 2

training:
  task: "classification"
  batch_size: 32
  max_epochs: 100

device:
  use_cuda: true

logging:
  level: "INFO"

random_seed: 42
```

Load configuration in Python:

```python
from lifespan_predictor.config import Config

config = Config.from_yaml("config.yaml")
```

## Data Configuration

Controls data file paths and directories.

### Parameters

#### `train_csv`
- **Type**: `str`
- **Default**: `"train.csv"`
- **Description**: Path to training CSV file containing SMILES and labels
- **Example**: `"data/train.csv"` or `"$DATA_DIR/train.csv"`

#### `test_csv`
- **Type**: `str`
- **Default**: `"test.csv"`
- **Description**: Path to test CSV file
- **Example**: `"data/test.csv"`

#### `smiles_column`
- **Type**: `str`
- **Default**: `"SMILES"`
- **Description**: Name of the column containing SMILES strings in CSV files
- **Example**: `"canonical_smiles"` or `"molecule"`

#### `label_column`
- **Type**: `str`
- **Default**: `"Life_extended"`
- **Description**: Name of the column containing labels/target values
- **Example**: `"activity"`, `"lifespan_change"`, `"IC50"`

#### `graph_features_dir`
- **Type**: `str`
- **Default**: `"processed_graph_features"`
- **Description**: Directory for caching computed graph features (adjacency matrices, node features)
- **Example**: `"cache/features"` or `"~/cache/graph_features"`

#### `fingerprints_dir`
- **Type**: `str`
- **Default**: `"processed_fingerprints"`
- **Description**: Directory for caching computed molecular fingerprints
- **Example**: `"cache/fingerprints"`

#### `output_dir`
- **Type**: `str`
- **Default**: `"results"`
- **Description**: Directory for saving training outputs, checkpoints, and results
- **Example**: `"outputs/experiment_1"`

### Example

```yaml
data:
  train_csv: "$HOME/data/lifespan/train.csv"
  test_csv: "$HOME/data/lifespan/test.csv"
  smiles_column: "canonical_smiles"
  label_column: "lifespan_extension"
  graph_features_dir: "cache/graph_features"
  fingerprints_dir: "cache/fingerprints"
  output_dir: "results/experiment_001"
```

## Featurization Configuration

Controls molecular featurization and fingerprint generation.

### Parameters

#### `max_atoms`
- **Type**: `int`
- **Default**: `200`
- **Range**: `>= 1`
- **Description**: Maximum number of atoms allowed in a molecule. Molecules exceeding this limit are skipped.
- **Recommendation**: Set based on your dataset. Most drug-like molecules have < 100 atoms.

#### `atom_feature_dim`
- **Type**: `int`
- **Default**: `75`
- **Range**: `>= 1`
- **Description**: Dimension of atom feature vectors from DeepChem's ConvMolFeaturizer
- **Note**: This is determined by the featurizer and should typically not be changed

#### `morgan_radius`
- **Type**: `int`
- **Default**: `2`
- **Range**: `>= 0`
- **Description**: Radius for Morgan (circular) fingerprints. Larger radius captures more distant atom relationships.
- **Common values**: 2 (ECFP4), 3 (ECFP6)

#### `morgan_nbits`
- **Type**: `int`
- **Default**: `2048`
- **Range**: `>= 1`
- **Description**: Number of bits in Morgan fingerprint bit vector
- **Common values**: 1024, 2048, 4096
- **Trade-off**: Larger = more information but higher memory usage

#### `rdkit_fp_nbits`
- **Type**: `int`
- **Default**: `2048`
- **Range**: `>= 1`
- **Description**: Number of bits in RDKit topological fingerprint
- **Common values**: 1024, 2048, 4096

#### `maccs_nbits`
- **Type**: `int`
- **Default**: `166`
- **Fixed**: Must be exactly 166
- **Description**: Number of MACCS structural keys (fixed by MACCS definition)
- **Note**: Do not change this value

#### `use_cache`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to cache computed features to disk
- **Recommendation**: Always keep `true` for efficiency

#### `n_jobs`
- **Type**: `int`
- **Default**: `-1`
- **Description**: Number of parallel jobs for featurization. `-1` uses all CPU cores.
- **Example**: `4` for 4 cores, `-1` for all cores

### Example

```yaml
featurization:
  max_atoms: 150
  atom_feature_dim: 75
  morgan_radius: 2
  morgan_nbits: 2048
  rdkit_fp_nbits: 2048
  maccs_nbits: 166
  use_cache: true
  n_jobs: -1
```

## Model Configuration

Controls model architecture and hyperparameters.

### Branch Control

#### `enable_gnn`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Enable Graph Neural Network (AttentiveFP) branch
- **Note**: At least one branch must be enabled

#### `enable_fp_dnn`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Enable fingerprint DNN branch (processes Morgan + RDKit fingerprints)

#### `enable_fp_cnn`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Enable fingerprint CNN branch (processes MACCS keys)

### GNN Parameters

#### `gnn_node_input_dim`
- **Type**: `int`
- **Default**: `78`
- **Range**: `>= 1`
- **Description**: Input dimension for GNN node features
- **Note**: Must match the output of your node featurizer (typically 78 for ConvMolFeaturizer with chirality)

#### `gnn_edge_input_dim`
- **Type**: `int`
- **Default**: `11`
- **Range**: `>= 1`
- **Description**: Input dimension for GNN edge features
- **Note**: Must match the output of your edge featurizer

#### `gnn_graph_embed_dim`
- **Type**: `int`
- **Default**: `128`
- **Range**: `>= 1`
- **Description**: Dimension of graph-level embeddings from GNN
- **Common values**: 64, 128, 256

#### `gnn_num_layers`
- **Type**: `int`
- **Default**: `2`
- **Range**: `>= 1`
- **Description**: Number of GNN layers
- **Trade-off**: More layers = larger receptive field but slower training

#### `gnn_num_timesteps`
- **Type**: `int`
- **Default**: `2`
- **Range**: `>= 1`
- **Description**: Number of timesteps for AttentiveFP attention mechanism
- **Common values**: 2, 3

#### `gnn_dropout`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `[0.0, 1.0]`
- **Description**: Dropout rate for GNN layers
- **Recommendation**: 0.3-0.5 for regularization

### Fingerprint CNN Parameters

#### `fp_cnn_output_dim`
- **Type**: `int`
- **Default**: `64`
- **Range**: `>= 1`
- **Description**: Output dimension of fingerprint CNN branch
- **Common values**: 32, 64, 128

### Fingerprint DNN Parameters

#### `fp_dnn_layers`
- **Type**: `List[int]`
- **Default**: `[256, 128]`
- **Description**: List of hidden layer sizes for fingerprint DNN
- **Example**: `[512, 256, 128]` for 3 hidden layers

#### `fp_dnn_output_dim`
- **Type**: `int`
- **Default**: `64`
- **Range**: `>= 1`
- **Description**: Output dimension of fingerprint DNN branch

#### `fp_dropout`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `[0.0, 1.0]`
- **Description**: Dropout rate for fingerprint branches

### Output Parameters

#### `n_output_tasks`
- **Type**: `int`
- **Default**: `1`
- **Range**: `>= 1`
- **Description**: Number of output tasks (typically 1 for single-task prediction)

### Example

```yaml
model:
  # Branch control
  enable_gnn: true
  enable_fp_dnn: true
  enable_fp_cnn: true
  
  # GNN parameters
  gnn_node_input_dim: 78
  gnn_edge_input_dim: 11
  gnn_graph_embed_dim: 128
  gnn_num_layers: 2
  gnn_num_timesteps: 2
  gnn_dropout: 0.5
  
  # Fingerprint CNN parameters
  fp_cnn_output_dim: 64
  
  # Fingerprint DNN parameters
  fp_dnn_layers: [256, 128]
  fp_dnn_output_dim: 64
  fp_dropout: 0.5
  
  # Output
  n_output_tasks: 1
```

## Training Configuration

Controls training process and hyperparameters.

### Parameters

#### `task`
- **Type**: `str`
- **Default**: `"classification"`
- **Options**: `"classification"` or `"regression"`
- **Description**: Type of prediction task
- **Note**: Determines loss function and metrics used

#### `batch_size`
- **Type**: `int`
- **Default**: `32`
- **Range**: `>= 1`
- **Description**: Number of samples per training batch
- **Recommendation**: Use largest size that fits in GPU memory (16, 32, 64, 128)

#### `max_epochs`
- **Type**: `int`
- **Default**: `100`
- **Range**: `>= 1`
- **Description**: Maximum number of training epochs
- **Note**: Training may stop earlier due to early stopping

#### `learning_rate`
- **Type**: `float`
- **Default**: `0.0001`
- **Range**: `> 0.0`
- **Description**: Learning rate for optimizer
- **Common values**: 1e-4, 1e-3, 5e-4
- **Recommendation**: Start with 1e-4 and adjust based on training curves

#### `weight_decay`
- **Type**: `float`
- **Default**: `0.0001`
- **Range**: `>= 0.0`
- **Description**: L2 regularization weight decay
- **Common values**: 0, 1e-5, 1e-4

#### `patience`
- **Type**: `int`
- **Default**: `15`
- **Range**: `>= 1`
- **Description**: Number of epochs to wait for improvement before early stopping
- **Recommendation**: 10-20 for most tasks

#### `gradient_clip`
- **Type**: `float`
- **Default**: `1.0`
- **Range**: `> 0.0`
- **Description**: Maximum gradient norm for gradient clipping
- **Purpose**: Prevents exploding gradients

#### `use_mixed_precision`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Use automatic mixed precision (AMP) training
- **Benefit**: Faster training and lower memory usage on modern GPUs

#### `val_split`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `(0.0, 1.0)`
- **Description**: Fraction of training data to use for validation
- **Common values**: 0.2, 0.3

#### `stratify`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to stratify train/validation split (for classification)
- **Recommendation**: Keep `true` for imbalanced datasets

#### `main_metric`
- **Type**: `str`
- **Default**: `"AUC"`
- **Description**: Main metric for model selection and early stopping
- **Classification options**: `"AUC"`, `"Accuracy"`, `"F1"`, `"Precision"`, `"Recall"`
- **Regression options**: `"RMSE"`, `"MAE"`, `"R2"`, `"PearsonCorrelation"`

### Example

```yaml
training:
  task: "classification"
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  patience: 15
  gradient_clip: 1.0
  use_mixed_precision: true
  val_split: 0.3
  stratify: true
  main_metric: "AUC"
```

## Device Configuration

Controls hardware device selection.

### Parameters

#### `use_cuda`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Use CUDA (GPU) if available
- **Note**: Automatically falls back to CPU if CUDA is not available

#### `device_id`
- **Type**: `int`
- **Default**: `0`
- **Range**: `>= 0`
- **Description**: CUDA device ID to use (for multi-GPU systems)
- **Example**: `0` for first GPU, `1` for second GPU

### Example

```yaml
device:
  use_cuda: true
  device_id: 0
```

Get device in Python:

```python
device = config.get_device()  # Returns "cuda:0" or "cpu"
```

## Logging Configuration

Controls logging behavior.

### Parameters

#### `level`
- **Type**: `str`
- **Default**: `"INFO"`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Description**: Logging level
- **Recommendation**: Use `"INFO"` for normal operation, `"DEBUG"` for troubleshooting

#### `log_file`
- **Type**: `str` or `null`
- **Default**: `"training.log"`
- **Description**: Path to log file. Set to `null` to disable file logging
- **Example**: `"logs/experiment_001.log"`

#### `tensorboard_dir`
- **Type**: `str`
- **Default**: `"runs"`
- **Description**: Directory for TensorBoard logs
- **Example**: `"tensorboard/experiment_001"`

#### `print_every_n_epochs`
- **Type**: `int`
- **Default**: `5`
- **Range**: `>= 1`
- **Description**: Print detailed metrics every N epochs
- **Purpose**: Reduces console output clutter

### Example

```yaml
logging:
  level: "INFO"
  log_file: "logs/training.log"
  tensorboard_dir: "runs/experiment_001"
  print_every_n_epochs: 5
```

## Common Scenarios

### Scenario 1: Small Dataset (< 1000 molecules)

```yaml
featurization:
  max_atoms: 150
  use_cache: true
  n_jobs: -1

model:
  enable_gnn: true
  enable_fp_dnn: true
  enable_fp_cnn: false  # Disable to reduce overfitting
  gnn_num_layers: 2
  gnn_dropout: 0.5
  fp_dnn_layers: [128, 64]  # Smaller network

training:
  batch_size: 16  # Smaller batches
  max_epochs: 200  # More epochs
  learning_rate: 0.0001
  patience: 30  # More patience
  val_split: 0.2  # Smaller validation set
```

### Scenario 2: Large Dataset (> 10,000 molecules)

```yaml
featurization:
  max_atoms: 200
  use_cache: true
  n_jobs: -1

model:
  enable_gnn: true
  enable_fp_dnn: true
  enable_fp_cnn: true
  gnn_num_layers: 3  # Deeper network
  gnn_graph_embed_dim: 256
  fp_dnn_layers: [512, 256, 128]

training:
  batch_size: 64  # Larger batches
  max_epochs: 100
  learning_rate: 0.001  # Higher learning rate
  patience: 15
  val_split: 0.3
```

### Scenario 3: Regression Task

```yaml
training:
  task: "regression"
  main_metric: "RMSE"  # Use regression metric
  batch_size: 32
  learning_rate: 0.0001

# Use MSE loss in Python:
# criterion = torch.nn.MSELoss()
```

### Scenario 4: Fast Prototyping

```yaml
featurization:
  max_atoms: 100  # Smaller molecules only
  morgan_nbits: 1024  # Smaller fingerprints
  rdkit_fp_nbits: 1024

model:
  enable_gnn: false  # Disable slow GNN
  enable_fp_dnn: true
  enable_fp_cnn: true
  fp_dnn_layers: [128]  # Single hidden layer

training:
  batch_size: 64
  max_epochs: 50
  patience: 10
```

### Scenario 5: GPU Memory Constrained

```yaml
training:
  batch_size: 8  # Reduce batch size
  use_mixed_precision: true  # Enable AMP

model:
  gnn_graph_embed_dim: 64  # Smaller embeddings
  fp_dnn_layers: [128, 64]  # Smaller network
```

## Derived Parameters

Some parameters are automatically computed from the configuration:

### Total Fingerprint Dimension

```python
total_fp_dim = morgan_nbits + rdkit_fp_nbits + maccs_nbits
# Default: 2048 + 2048 + 166 = 4262
```

Access in Python:

```python
total_dim = config.get_total_fp_dim()
```

### Total Feature Dimension

The total feature dimension for the prediction head is computed as:

```python
total_feat_dim = 0
if enable_gnn:
    total_feat_dim += gnn_graph_embed_dim
if enable_fp_dnn:
    total_feat_dim += fp_dnn_output_dim
if enable_fp_cnn:
    total_feat_dim += fp_cnn_output_dim
# Default: 128 + 64 + 64 = 256
```

### Device String

```python
device = "cuda:{device_id}" if use_cuda and torch.cuda.is_available() else "cpu"
```

Access in Python:

```python
device = config.get_device()
```

## Validation Rules

The configuration system enforces several validation rules:

1. **At least one branch must be enabled**: `enable_gnn`, `enable_fp_dnn`, or `enable_fp_cnn` must be `true`
2. **MACCS keys must be 166 bits**: `maccs_nbits` must equal 166
3. **Task must be valid**: `task` must be `"classification"` or `"regression"`
4. **Metric must match task**: Classification tasks require classification metrics, regression tasks require regression metrics
5. **Ranges must be valid**: All numeric parameters must be within their specified ranges
6. **Paths are expanded**: Environment variables and `~` are automatically expanded

## Environment Variables

You can use environment variables in configuration files:

```yaml
data:
  train_csv: "$DATA_DIR/train.csv"
  output_dir: "$HOME/experiments/lifespan"

logging:
  log_file: "${LOG_DIR}/training.log"
```

Set environment variables before running:

```bash
export DATA_DIR=/path/to/data
export LOG_DIR=/path/to/logs
python train.py
```

## Best Practices

1. **Start with defaults**: The default configuration works well for most tasks
2. **Use caching**: Always keep `use_cache: true` for efficiency
3. **Monitor training**: Use TensorBoard to visualize training progress
4. **Tune learning rate**: If training is unstable, reduce learning rate
5. **Adjust batch size**: Use largest batch size that fits in memory
6. **Enable all branches**: Start with all branches enabled, disable if needed
7. **Save configurations**: Save configuration with each experiment for reproducibility
8. **Version control**: Keep configuration files in version control

## Troubleshooting

### Configuration not loading

```python
# Check for YAML syntax errors
import yaml
with open("config.yaml") as f:
    yaml.safe_load(f)  # Will raise error if invalid
```

### Validation errors

```python
# Pydantic will provide detailed error messages
try:
    config = Config.from_yaml("config.yaml")
except Exception as e:
    print(e)  # Shows which parameter is invalid and why
```

### Path not found

```python
# Check path expansion
import os
path = os.path.expandvars(os.path.expanduser("$HOME/data"))
print(path)  # Shows expanded path
```
