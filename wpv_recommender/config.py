"""Configuration for model and training parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer Autoencoder model configuration."""

    # Input dimensions
    seq_length: int = 2160  # 90 days * 24 hours
    input_dim: int = 1  # Single pageview value per timestep

    # Transformer dimensions
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512

    # Architecture
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4

    # Regularization
    dropout: float = 0.1

    # Embedding dimension (bottleneck)
    embedding_dim: int = 128


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    batch_size: int = 64
    train_split: float = 0.9
    num_workers: int = 4

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100

    # Scheduler
    t_0: int = 10  # CosineAnnealingWarmRestarts initial period
    t_mult: int = 2  # Period multiplier

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

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
