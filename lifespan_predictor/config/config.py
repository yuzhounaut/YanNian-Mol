"""Configuration management with Pydantic validation."""

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data-related configuration."""

    train_csv: str = Field(default="train.csv", description="Path to training CSV file")
    test_csv: str = Field(default="test.csv", description="Path to test CSV file")
    smiles_column: str = Field(default="SMILES", description="Name of SMILES column in CSV")
    label_column: str = Field(default="Life_extended", description="Name of label column in CSV")
    graph_features_dir: str = Field(
        default="processed_graph_features", description="Directory for cached graph features"
    )
    fingerprints_dir: str = Field(
        default="processed_fingerprints", description="Directory for cached fingerprints"
    )
    output_dir: str = Field(default="results", description="Directory for output files")

    @field_validator(
        "train_csv", "test_csv", "graph_features_dir", "fingerprints_dir", "output_dir"
    )
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand environment variables and user home directory in paths."""
        return os.path.expandvars(os.path.expanduser(v))


class FeaturizationConfig(BaseModel):
    """Featurization-related configuration."""

    max_atoms: int = Field(default=200, ge=1, description="Maximum number of atoms in molecule")
    atom_feature_dim: int = Field(default=75, ge=1, description="Dimension of atom features")
    morgan_radius: int = Field(default=2, ge=0, description="Radius for Morgan fingerprints")
    morgan_nbits: int = Field(
        default=2048, ge=1, description="Number of bits for Morgan fingerprints"
    )
    rdkit_fp_nbits: int = Field(
        default=2048, ge=1, description="Number of bits for RDKit fingerprints"
    )
    maccs_nbits: int = Field(default=166, description="Number of bits for MACCS keys (fixed)")
    use_cache: bool = Field(default=True, description="Whether to use cached features")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")

    @field_validator("maccs_nbits")
    @classmethod
    def validate_maccs_nbits(cls, v: int) -> int:
        """MACCS keys always have 166 bits."""
        if v != 166:
            raise ValueError("MACCS keys must have exactly 166 bits")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    enable_gnn: bool = Field(default=True, description="Enable GNN branch")
    enable_fp_dnn: bool = Field(default=True, description="Enable fingerprint DNN branch")
    enable_fp_cnn: bool = Field(default=True, description="Enable fingerprint CNN branch")

    gnn_node_input_dim: int = Field(default=78, ge=1, description="GNN node input dimension")
    gnn_edge_input_dim: int = Field(default=11, ge=1, description="GNN edge input dimension")
    gnn_graph_embed_dim: int = Field(default=128, ge=1, description="GNN graph embedding dimension")
    gnn_num_layers: int = Field(default=2, ge=1, description="Number of GNN layers")
    gnn_num_timesteps: int = Field(default=2, ge=1, description="Number of GNN timesteps")
    gnn_dropout: float = Field(default=0.5, ge=0.0, le=1.0, description="GNN dropout rate")

    fp_cnn_output_dim: int = Field(default=64, ge=1, description="Fingerprint CNN output dimension")
    fp_dnn_layers: List[int] = Field(
        default=[256, 128], description="Fingerprint DNN hidden layer sizes"
    )
    fp_dnn_output_dim: int = Field(default=64, ge=1, description="Fingerprint DNN output dimension")
    fp_dropout: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Fingerprint branch dropout rate"
    )

    n_output_tasks: int = Field(default=1, ge=1, description="Number of output tasks")

    @model_validator(mode="after")
    def validate_at_least_one_branch(self) -> "ModelConfig":
        """Ensure at least one branch is enabled."""
        if not (self.enable_gnn or self.enable_fp_dnn or self.enable_fp_cnn):
            raise ValueError("At least one model branch must be enabled")
        return self


class TrainingConfig(BaseModel):
    """Training-related configuration."""

    task: str = Field(
        default="classification", description="Task type: 'classification' or 'regression'"
    )
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    max_epochs: int = Field(default=100, ge=1, description="Maximum number of training epochs")
    learning_rate: float = Field(default=0.0001, gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=0.0001, ge=0.0, description="Weight decay for optimizer")
    patience: int = Field(default=15, ge=1, description="Early stopping patience")
    gradient_clip: float = Field(default=1.0, gt=0.0, description="Gradient clipping threshold")
    use_mixed_precision: bool = Field(default=True, description="Use mixed precision training")
    val_split: float = Field(default=0.3, gt=0.0, lt=1.0, description="Validation split ratio")
    stratify: bool = Field(default=True, description="Stratify train/val split")
    main_metric: str = Field(default="AUC", description="Main metric for model selection")

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        """Validate task type."""
        if v not in ["classification", "regression"]:
            raise ValueError("Task must be 'classification' or 'regression'")
        return v

    @model_validator(mode="after")
    def validate_metric_for_task(self) -> "TrainingConfig":
        """Validate that main_metric is appropriate for task."""
        classification_metrics = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
        regression_metrics = ["RMSE", "MAE", "R2", "PearsonCorrelation"]

        if self.task == "classification" and self.main_metric not in classification_metrics:
            raise ValueError(
                f"For classification, main_metric must be one of {classification_metrics}"
            )
        elif self.task == "regression" and self.main_metric not in regression_metrics:
            raise ValueError(f"For regression, main_metric must be one of {regression_metrics}")

        return self


class DeviceConfig(BaseModel):
    """Device configuration."""

    use_cuda: bool = Field(default=True, description="Use CUDA if available")
    device_id: int = Field(default=0, ge=0, description="CUDA device ID")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="training.log", description="Log file path")
    tensorboard_dir: str = Field(default="runs", description="TensorBoard log directory")
    print_every_n_epochs: int = Field(default=5, ge=1, description="Print metrics every N epochs")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}")
        return v_upper

    @field_validator("log_file", "tensorboard_dir")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """Expand environment variables and user home directory in paths."""
        if v is None:
            return None
        return os.path.expandvars(os.path.expanduser(v))


class Config(BaseModel):
    """Main configuration class with validation."""

    data: DataConfig = Field(default_factory=DataConfig)
    featurization: FeaturizationConfig = Field(default_factory=FeaturizationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = os.path.expandvars(os.path.expanduser(path))

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        return cls(**config_dict)

    def validate(self) -> None:
        """
        Validate configuration.

        This method is called automatically by Pydantic during initialization,
        but can be called explicitly if needed.
        """
        # Pydantic handles validation automatically

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save configuration file
        """
        path = os.path.expandvars(os.path.expanduser(path))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_total_fp_dim(self) -> int:
        """
        Calculate total fingerprint dimension.

        Returns:
            Total dimension of concatenated fingerprints
        """
        return (
            self.featurization.morgan_nbits
            + self.featurization.rdkit_fp_nbits
            + self.featurization.maccs_nbits
        )

    def get_device(self) -> str:
        """
        Get device string for PyTorch.

        Returns:
            Device string (e.g., 'cuda:0' or 'cpu')
        """
        import torch

        if self.device.use_cuda and torch.cuda.is_available():
            return f"cuda:{self.device.device_id}"
        return "cpu"
