"""
Training callbacks for monitoring and controlling the training process.

This module provides callback classes for early stopping, model checkpointing,
and learning rate scheduling during training.
"""

import logging
import os
from abc import ABC
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base callback class for training hooks.

    Callbacks allow custom code to be executed at specific points during training,
    such as at the end of each epoch or batch. Subclasses should override the
    relevant hook methods.

    Examples
    --------
    >>> class CustomCallback(Callback):
    ...     def on_epoch_end(self, epoch, logs):
    ...         print(f"Epoch {epoch} completed with loss: {logs['train_loss']}")
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training.

        Parameters
        ----------
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Dict[str, float]
            Dictionary containing metrics for the epoch
        """

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each batch.

        Parameters
        ----------
        batch : int
            Current batch number
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each batch.

        Parameters
        ----------
        batch : int
            Current batch number
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Name of the metric to monitor (e.g., 'val_loss', 'val_AUC')
    patience : int
        Number of epochs with no improvement after which training will be stopped
    mode : str
        One of {'min', 'max'}. In 'min' mode, training stops when the monitored
        metric stops decreasing; in 'max' mode, it stops when the metric stops increasing
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement
    verbose : bool
        If True, print messages when early stopping is triggered

    Attributes
    ----------
    best_score : Optional[float]
        Best score observed so far
    counter : int
        Number of epochs since last improvement
    early_stop : bool
        Whether early stopping has been triggered

    Examples
    --------
    >>> early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    >>> # In training loop:
    >>> early_stop.on_epoch_end(epoch, {'val_loss': 0.5})
    >>> if early_stop.early_stop:
    ...     print("Early stopping triggered")
    ...     break
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        """Initialize EarlyStopping callback."""
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

        # Set comparison operator based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

        logger.info(
            f"Initialized EarlyStopping: monitor={monitor}, patience={patience}, "
            f"mode={mode}, min_delta={min_delta}"
        )

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Check if early stopping should be triggered.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Dict[str, float]
            Dictionary containing metrics for the epoch
        """
        if self.monitor not in logs:
            logger.warning(
                f"Early stopping metric '{self.monitor}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )
            return

        current_score = logs[self.monitor]

        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            if self.verbose:
                logger.info(f"Epoch {epoch}: {self.monitor} = {current_score:.6f} (initial)")
        elif self.is_better(current_score, self.best_score):
            # Improvement
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} improved from "
                    f"{self.best_score:.6f} to {current_score:.6f}"
                )
            self.best_score = current_score
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} = {current_score:.6f} "
                    f"(no improvement for {self.counter}/{self.patience} epochs)"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs. "
                        f"Best {self.monitor}: {self.best_score:.6f}"
                    )


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.

    Parameters
    ----------
    filepath : str
        Path to save the model checkpoint. Can include formatting options:
        - {epoch}: Current epoch number
        - {monitor}: Value of the monitored metric
    monitor : str
        Name of the metric to monitor for saving best model
    mode : str
        One of {'min', 'max'}. In 'min' mode, saves when monitored metric decreases;
        in 'max' mode, saves when it increases
    save_best_only : bool
        If True, only save when the monitored metric improves
    save_last : bool
        If True, always save the last checkpoint
    verbose : bool
        If True, print messages when saving checkpoints

    Attributes
    ----------
    best_score : Optional[float]
        Best score observed so far

    Examples
    --------
    >>> checkpoint = ModelCheckpoint(
    ...     filepath='checkpoints/model_epoch_{epoch}.pt',
    ...     monitor='val_AUC',
    ...     mode='max',
    ...     save_best_only=True
    ... )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """Initialize ModelCheckpoint callback."""
        super().__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.best_score: Optional[float] = None

        # Set comparison operator based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        logger.info(
            f"Initialized ModelCheckpoint: filepath={filepath}, monitor={monitor}, "
            f"mode={mode}, save_best_only={save_best_only}"
        )

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Save checkpoint if conditions are met.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Dict[str, float]
            Dictionary containing metrics for the epoch
        """
        if self.monitor not in logs:
            logger.warning(
                f"ModelCheckpoint metric '{self.monitor}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )
            return

        current_score = logs[self.monitor]
        should_save = False

        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            should_save = True
        elif self.is_better(current_score, self.best_score):
            # Improvement
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} improved from "
                    f"{self.best_score:.6f} to {current_score:.6f}"
                )
            self.best_score = current_score
            should_save = True
        elif not self.save_best_only:
            # Save even without improvement
            should_save = True

        if should_save:
            # Format filepath with epoch and metric value
            filepath = self.filepath.format(epoch=epoch, monitor=f"{current_score:.6f}")

            # Store filepath for trainer to use
            logs["_checkpoint_path"] = filepath
            logs["_should_save_checkpoint"] = True

            if self.verbose:
                logger.info(f"Saving checkpoint to {filepath}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Save final checkpoint if save_last is True.

        Parameters
        ----------
        logs : Optional[Dict[str, Any]]
            Dictionary containing training information
        """
        if self.save_last and logs is not None:
            filepath = self.filepath.replace("{epoch}", "final")
            logs["_checkpoint_path"] = filepath
            logs["_should_save_checkpoint"] = True

            if self.verbose:
                logger.info(f"Saving final checkpoint to {filepath}")


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.

    This callback wraps PyTorch learning rate schedulers and calls their step()
    method at the end of each epoch.

    Parameters
    ----------
    scheduler : _LRScheduler
        PyTorch learning rate scheduler instance
    verbose : bool
        If True, print messages when learning rate changes

    Examples
    --------
    >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    >>> lr_scheduler = LearningRateScheduler(scheduler, verbose=True)
    """

    def __init__(self, scheduler: _LRScheduler, verbose: bool = True):
        """Initialize LearningRateScheduler callback."""
        super().__init__()

        self.scheduler = scheduler
        self.verbose = verbose

        logger.info(f"Initialized LearningRateScheduler: {scheduler.__class__.__name__}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Step the learning rate scheduler.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Dict[str, float]
            Dictionary containing metrics for the epoch
        """
        # Get current learning rate before stepping
        current_lr = self.scheduler.optimizer.param_groups[0]["lr"]

        # Step the scheduler
        # ReduceLROnPlateau requires a metric value
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Try to find the metric the scheduler is monitoring
            # Default to val_loss if not specified
            metric_name = getattr(self.scheduler, "monitor", "val_loss")
            if metric_name in logs:
                self.scheduler.step(logs[metric_name])
            else:
                logger.warning(
                    f"Metric '{metric_name}' not found in logs for ReduceLROnPlateau. "
                    f"Available metrics: {list(logs.keys())}"
                )
        else:
            self.scheduler.step()

        # Get new learning rate after stepping
        new_lr = self.scheduler.optimizer.param_groups[0]["lr"]

        # Log if learning rate changed
        if self.verbose and new_lr != current_lr:
            logger.info(
                f"Epoch {epoch}: Learning rate changed from {current_lr:.6e} to {new_lr:.6e}"
            )

        # Add learning rate to logs
        logs["learning_rate"] = new_lr
