"""Training loop for the CNN Autoencoder."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig, PathConfig
from ..model.cnn_ae import CNNAutoencoder


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Trainer for the CNN Autoencoder."""

    def __init__(
        self,
        model: CNNAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        paths: PathConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            paths: Path configuration
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.paths = paths

        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.t_0,
            T_mult=config.t_mult,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(batch)

            # Compute loss
            loss = self.criterion(reconstructed, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        for batch in pbar:
            batch = batch.to(self.device)

            reconstructed, _ = self.model(batch)
            loss = self.criterion(reconstructed, batch)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """
        Save a checkpoint.

        Args:
            path: Path to save the checkpoint
            is_best: Whether this is the best model so far
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": {
                "seq_length": self.model.seq_length,
                "channels": self.model.channels,
                "embedding_dim": self.model.embedding_dim,
            },
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = self.paths.get_checkpoint_path(epoch=None)
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load a checkpoint.

        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, resume_from: Optional[Path] = None) -> None:
        """
        Run the full training loop.

        Args:
            resume_from: Optional path to resume training from
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        print("=" * 60)
        print("Starting training")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Max epochs: {self.config.max_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Log
            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {lr:.2e}"
            )

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = self.paths.get_checkpoint_path(epoch + 1)
                self.save_checkpoint(checkpoint_path, is_best=is_best)
            elif is_best:
                self.save_checkpoint(
                    self.paths.get_checkpoint_path(epoch=None),
                    is_best=True,
                )

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print("=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def load_model_for_inference(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> CNNAutoencoder:
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to the checkpoint
        device: Device to load the model on

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = CNNAutoencoder(
        seq_length=config["seq_length"],
        channels=tuple(config["channels"]),
        embedding_dim=config["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model
