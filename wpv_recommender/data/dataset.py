"""PyTorch Dataset for Wikipedia pageview data."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .preprocessing import PageviewDataLoader


class PageviewDataset(Dataset):
    """PyTorch Dataset for Wikipedia pageview time series."""

    def __init__(
        self,
        data: np.ndarray,
        page_titles: Optional[list[str]] = None,
        return_titles: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data: Normalized pageview data [num_pages, seq_length]
            page_titles: Optional list of page titles
            return_titles: Whether to return page titles with data
        """
        self.data = torch.from_numpy(data).float()
        self.page_titles = page_titles
        self.return_titles = return_titles

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        Returns:
            If return_titles: (data [seq_length], page_title)
            Otherwise: data [seq_length]
        """
        sample = self.data[idx]
        if self.return_titles and self.page_titles is not None:
            return sample, self.page_titles[idx]
        return sample


def create_data_loaders(
    data_loader: PageviewDataLoader,
    batch_size: int = 64,
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, list[str], list[str]]:
    """
    Create train and validation data loaders.

    Args:
        data_loader: PageviewDataLoader with normalized data
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, train_titles, val_titles)
    """
    if data_loader.normalized_data is None:
        raise ValueError("Data loader has no normalized data")

    # Create full dataset
    full_dataset = PageviewDataset(
        data_loader.normalized_data,
        page_titles=data_loader.page_titles,
        return_titles=False,
    )

    # Split into train and validation
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Get the indices for title mapping
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    train_titles = [data_loader.page_titles[i] for i in train_indices]
    val_titles = [data_loader.page_titles[i] for i in val_indices]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"Created data loaders:")
    print(f"  Train: {n_train} samples, {len(train_loader)} batches")
    print(f"  Val:   {n_val} samples, {len(val_loader)} batches")

    return train_loader, val_loader, train_titles, val_titles


def create_inference_loader(
    data_loader: PageviewDataLoader,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a data loader for inference (embedding extraction).

    Args:
        data_loader: PageviewDataLoader with normalized data
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader for inference
    """
    if data_loader.normalized_data is None:
        raise ValueError("Data loader has no normalized data")

    dataset = PageviewDataset(
        data_loader.normalized_data,
        page_titles=data_loader.page_titles,
        return_titles=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
