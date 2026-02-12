"""Configuration for model and training parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """CNN Autoencoder model configuration."""

    # Input dimensions
    seq_length: int = 2160  # 90 days * 24 hours

    # CNN channels at each encoder stage
    channels: tuple[int, ...] = (64, 128, 256, 256, 512, 512, 512)

    # Regularization
    dropout: float = 0.1

    # Embedding dimension (bottleneck)
    embedding_dim: int = 512

    # Hidden dimension in bottleneck MLP
    bottleneck_hidden_dim: int = 1024


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    batch_size: int = 128
    train_split: float = 0.9
    num_workers: int = 4

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 500

    # Scheduler (ReduceLROnPlateau)
    scheduler_factor: float = 0.5  # Multiply LR by this on plateau
    scheduler_patience: int = 5  # Epochs to wait before reducing LR
    min_lr: float = 1e-6  # LR floor

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4

    # Loss
    spectral_loss_weight: float = 0.1  # Weight for FFT spectral loss

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 5


@dataclass
class PathConfig:
    """Path configuration."""

    base_dir: Path = field(default_factory=lambda: Path("/Users/zechen/dev/wpv"))

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def model_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def embedding_dir(self) -> Path:
        return self.base_dir / "embeddings"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "dataset_20k.pkl"

    @property
    def preprocessed_path(self) -> Path:
        return self.data_dir / "preprocessed_data.pkl"

    def get_checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        if epoch is None:
            return self.model_dir / "best_model.pt"
        return self.model_dir / f"checkpoint_epoch_{epoch}.pt"

    @property
    def embedding_index_path(self) -> Path:
        return self.embedding_dir / "faiss_index.bin"

    @property
    def embedding_metadata_path(self) -> Path:
        return self.embedding_dir / "metadata.pkl"


@dataclass
class Config:
    """Combined configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
