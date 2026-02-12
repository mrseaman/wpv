#!/usr/bin/env python3
"""Script to train the CNN Autoencoder model."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from wpv_recommender.config import get_config
from wpv_recommender.data.preprocessing import PageviewDataLoader
from wpv_recommender.data.dataset import create_data_loaders
from wpv_recommender.model.cnn_ae import CNNAutoencoder
from wpv_recommender.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the CNN Autoencoder for Wikipedia pageview embeddings."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to preprocessed data (default: from config)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to train on",
    )
    args = parser.parse_args()

    config = get_config()

    # Override config with CLI args
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    data_path = args.data or config.paths.preprocessed_path

    print("=" * 60)
    print("Wikipedia Pageview Recommender - Model Training")
    print("=" * 60)

    # Load preprocessed data
    print(f"Loading data from {data_path}")
    data_loader = PageviewDataLoader.load_preprocessed(data_path)

    # Create data loaders
    train_loader, val_loader, _, _ = create_data_loaders(
        data_loader,
        batch_size=config.training.batch_size,
        train_split=config.training.train_split,
        num_workers=config.training.num_workers,
    )

    # Create model
    print("\nCreating model...")
    model = CNNAutoencoder(
        seq_length=config.model.seq_length,
        channels=config.model.channels,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        bottleneck_hidden_dim=config.model.bottleneck_hidden_dim,
    )
    print(f"Model: {model}")

    # Set device
    device = None
    if args.device is not None:
        device = torch.device(args.device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        paths=config.paths,
        device=device,
    )

    # Train
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
