"""
Training orchestrator for lifespan prediction models.

This module provides the Trainer class that handles the training loop,
validation, and coordination of callbacks.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from lifespan_predictor.config import Config
from lifespan_predictor.training.callbacks import Callback
from lifespan_predictor.training.metrics import MetricCollection
from lifespan_predictor.utils.memory import GPUMemoryMonitor

logger = logging.getLogger(__name__)

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install tensorboard to enable logging.")


class Trainer:
    """
    Training orchestrator with callbacks and monitoring.

    This class handles the training loop, validation, gradient clipping,
    mixed precision training, and coordination of callbacks.

    Parameters
    ----------
    model : nn.Module
        Model to train
    config : Config
        Configuration object
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    optimizer : Optimizer
        Optimizer for training
    criterion : nn.Module
        Loss function
    metrics : MetricCollection
        Collection of metrics to compute
    callbacks : Optional[List[Callback]]
        List of callbacks to execute during training
    device : Optional[str]
        Device to use for training (e.g., 'cuda:0' or 'cpu')

    Attributes
    ----------
    model : nn.Module
        Model being trained
    config : Config
        Configuration object
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    optimizer : Optimizer
        Optimizer
    criterion : nn.Module
        Loss function
    metrics : MetricCollection
        Metrics collection
    callbacks : List[Callback]
        List of callbacks
    device : str
        Device for training
    scaler : Optional[torch.cuda.amp.GradScaler]
        Gradient scaler for mixed precision training
    history : Dict[str, List[float]]
        Training history
    current_epoch : int
        Current epoch number

    Examples
    --------
    >>> from lifespan_predictor.training import Trainer, EarlyStopping
    >>> trainer = Trainer(
    ...     model=model,
    ...     config=config,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     optimizer=optimizer,
    ...     criterion=criterion,
    ...     metrics=metrics,
    ...     callbacks=[EarlyStopping(patience=10)]
    ... )
    >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        metrics: MetricCollection,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[str] = None,
    ):
        """Initialize Trainer."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.callbacks = callbacks or []

        # Set device
        if device is None:
            device = config.get_device()
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Setup mixed precision training
        self.use_mixed_precision = config.training.use_mixed_precision and "cuda" in self.device
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

        # Setup GPU memory monitoring
        self.gpu_monitor = None
        if "cuda" in self.device and torch.cuda.is_available():
            try:
                self.gpu_monitor = GPUMemoryMonitor(device=self.device, threshold=0.8)
                logger.info("GPU memory monitoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory monitor: {e}")

        # Training state
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.current_epoch = 0
        self.best_val_metric = None

        # Setup TensorBoard logging
        self.writer = None
        if TENSORBOARD_AVAILABLE and config.logging.tensorboard_dir:
            tensorboard_dir = Path(config.logging.tensorboard_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        # Setup CSV logging
        self.csv_log_file = None
        if config.data.output_dir:
            output_dir = Path(config.data.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.csv_log_file = output_dir / "training_metrics.csv"
            logger.info(f"CSV logging enabled: {self.csv_log_file}")

        logger.info(
            f"Initialized Trainer: device={self.device}, "
            f"mixed_precision={self.use_mixed_precision}, "
            f"gradient_clip={config.training.gradient_clip}"
        )

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training history and final metrics

        Examples
        --------
        >>> trainer = Trainer(...)
        >>> history = trainer.train()
        >>> print(f"Best validation loss: {min(history['val_loss'])}")
        """
        logger.info("Starting training")
        start_time = time.time()

        # Call on_train_begin for all callbacks
        for callback in self.callbacks:
            callback.on_train_begin()

        try:
            for epoch in range(self.config.training.max_epochs):
                self.current_epoch = epoch

                # Log GPU memory at start of epoch
                if self.gpu_monitor is not None:
                    self.gpu_monitor.log_memory_usage(f"Epoch {epoch + 1} start - ")

                # Call on_epoch_begin for all callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)

                # Train for one epoch
                train_metrics = self._train_epoch(epoch)

                # Clear GPU cache if memory is critical
                if self.gpu_monitor is not None and self.gpu_monitor.is_memory_critical():
                    logger.warning("GPU memory usage critical, clearing cache")
                    self.gpu_monitor.clear_cache()

                # Validate
                val_metrics = self._validate(epoch)

                # Combine metrics
                epoch_logs = {**train_metrics, **val_metrics}

                # Update history
                for key, value in epoch_logs.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

                # Log metrics periodically
                if (epoch + 1) % self.config.logging.print_every_n_epochs == 0:
                    self._log_metrics(epoch, epoch_logs)

                # Log to TensorBoard
                self._log_to_tensorboard(epoch, epoch_logs)

                # Log to CSV
                self._log_to_csv(epoch, epoch_logs)

                # Call on_epoch_end for all callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, epoch_logs)

                # Check for early stopping
                if self._should_stop():
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Call on_train_end for all callbacks
            for callback in self.callbacks:
                callback.on_train_end({"history": self.history})

            # Close TensorBoard writer
            if self.writer is not None:
                self.writer.close()
                logger.info("Closed TensorBoard writer")

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.2f}s "
            f"({total_time / (self.current_epoch + 1):.2f}s per epoch)"
        )

        return {
            "history": self.history,
            "final_epoch": self.current_epoch,
            "total_time": total_time,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number

        Returns
        -------
        Dict[str, float]
            Dictionary containing training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.max_epochs} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # Call on_batch_begin for all callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx)

            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch)
                    loss = self.criterion(predictions, batch.y)
            else:
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch.y)

            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )

                self.optimizer.step()

            # Accumulate loss and predictions
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(batch.y.detach().cpu())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Call on_batch_end for all callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, {"loss": loss.item()})

        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Compute metrics
        train_metrics = self.metrics.compute_all(all_labels, all_predictions)
        train_metrics["train_loss"] = avg_loss

        return train_metrics

    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Parameters
        ----------
        epoch : int
            Current epoch number

        Returns
        -------
        Dict[str, float]
            Dictionary containing validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.max_epochs} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch)
                        loss = self.criterion(predictions, batch.y)
                else:
                    predictions = self.model(batch)
                    loss = self.criterion(predictions, batch.y)

                # Accumulate loss and predictions
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_labels.append(batch.y.cpu())

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})

        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Compute metrics
        val_metrics = self.metrics.compute_all(all_labels, all_predictions)
        val_metrics["val_loss"] = avg_loss

        return val_metrics

    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics to console.

        Parameters
        ----------
        epoch : int
            Current epoch number
        metrics : Dict[str, float]
            Dictionary containing metrics
        """
        # Separate train and val metrics
        train_metrics = {k: v for k, v in metrics.items() if k.startswith("train_")}
        val_metrics = {k: v for k, v in metrics.items() if k.startswith("val_")}

        # Format metrics
        train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

        logger.info(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
        logger.info(f"  Train - {train_str}")
        logger.info(f"  Val   - {val_str}")

    def _should_stop(self) -> bool:
        """
        Check if training should stop early.

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        for callback in self.callbacks:
            if hasattr(callback, "early_stop") and callback.early_stop:
                return True
        return False

    def save_checkpoint(
        self, filepath: str, epoch: int, metrics: Optional[Dict[str, float]] = None, **kwargs
    ) -> None:
        """
        Save model checkpoint with metadata.

        Parameters
        ----------
        filepath : str
            Path to save checkpoint
        epoch : int
            Current epoch number
        metrics : Optional[Dict[str, float]]
            Current metrics
        **kwargs
            Additional metadata to save

        Examples
        --------
        >>> trainer.save_checkpoint(
        ...     'checkpoints/model.pt',
        ...     epoch=10,
        ...     metrics={'val_loss': 0.5}
        ... )
        """
        import subprocess
        from datetime import datetime

        # Get git commit hash if available
        git_commit = None
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            )
        except Exception:
            pass

        # Get package versions
        package_versions = {"torch": torch.__version__, "numpy": np.__version__}

        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "package_versions": package_versions,
            **kwargs,
        }

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint for resuming training.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizer : bool
            Whether to load optimizer state

        Returns
        -------
        Dict[str, Any]
            Checkpoint dictionary containing metadata

        Examples
        --------
        >>> trainer = Trainer(...)
        >>> checkpoint = trainer.load_checkpoint('checkpoints/model.pt')
        >>> print(f"Resuming from epoch {checkpoint['epoch']}")
        """
        logger.info(f"Loading checkpoint from {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore history
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        # Restore epoch
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]

        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def _log_to_tensorboard(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics to TensorBoard.

        Parameters
        ----------
        epoch : int
            Current epoch number
        metrics : Dict[str, float]
            Dictionary containing metrics
        """
        if self.writer is None:
            return

        # Log all metrics
        for key, value in metrics.items():
            # Skip internal keys
            if key.startswith("_"):
                continue

            # Determine tag prefix
            if key.startswith("train_"):
                tag = f"train/{key.replace('train_', '')}"
            elif key.startswith("val_"):
                tag = f"val/{key.replace('val_', '')}"
            else:
                tag = f"other/{key}"

            self.writer.add_scalar(tag, value, epoch)

        # Log learning rate if available
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", current_lr, epoch)

        # Flush writer
        self.writer.flush()

    def _log_to_csv(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics to CSV file.

        Parameters
        ----------
        epoch : int
            Current epoch number
        metrics : Dict[str, float]
            Dictionary containing metrics
        """
        if self.csv_log_file is None:
            return

        # Filter out internal keys
        metrics_to_log = {k: v for k, v in metrics.items() if not k.startswith("_")}

        # Add epoch and learning rate
        metrics_to_log["epoch"] = epoch
        metrics_to_log["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Check if file exists
        file_exists = self.csv_log_file.exists()

        # Write to CSV
        with open(self.csv_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics_to_log.keys()))

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics_to_log)
