"""Data loading and preprocessing utilities."""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from wpv_recommender.data.hourly_views import get_all_hourly_views


class PageviewDataLoader:
    """Loads and preprocesses Wikipedia pageview data."""

    def __init__(self, dataset_path: Path, seq_length: int = 2160):
        """
        Initialize the data loader.

        Args:
            dataset_path: Path to the dataset_20k.pkl file
            seq_length: Expected sequence length (90 days * 24 hours = 2160)
        """
        self.dataset_path = Path(dataset_path)
        self.seq_length = seq_length
        self.page_titles: list[str] = []
        self.raw_data: Optional[np.ndarray] = None
        self.normalized_data: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def load_raw_data(self) -> Tuple[np.ndarray, list[str]]:
        """
        Load raw pageview data from the pickle file.

        Returns:
            Tuple of (data array [num_pages, seq_length], page_titles list)
        """
        print(f"Loading dataset from {self.dataset_path}")
        dataset = pd.read_pickle(self.dataset_path)
        dataset = dataset.set_index("page_title", drop=True)

        data_list = []
        self.page_titles = []
        errors = 0

        print("Extracting hourly views...")
        for idx, (page_title, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            try:
                hourly_views, _ = get_all_hourly_views(row)
                values = hourly_views.values.astype(np.float32)

                # Ensure correct length
                if len(values) == self.seq_length:
                    data_list.append(values)
                    self.page_titles.append(page_title)
                elif len(values) > self.seq_length:
                    # Truncate if longer
                    data_list.append(values[: self.seq_length])
                    self.page_titles.append(page_title)
                else:
                    # Pad with zeros if shorter
                    padded = np.zeros(self.seq_length, dtype=np.float32)
                    padded[: len(values)] = values
                    data_list.append(padded)
                    self.page_titles.append(page_title)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error processing {page_title}: {e}")

        if errors > 5:
            print(f"... and {errors - 5} more errors")

        self.raw_data = np.stack(data_list)
        print(f"Loaded {len(self.page_titles)} pages with shape {self.raw_data.shape}")
        return self.raw_data, self.page_titles

    def normalize(
        self, data: Optional[np.ndarray] = None, eps: float = 1e-8
    ) -> np.ndarray:
        """
        Apply log1p transform and per-series z-score normalization.

        Args:
            data: Data to normalize (uses self.raw_data if None)
            eps: Small constant to prevent division by zero

        Returns:
            Normalized data array
        """
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_raw_data first.")
            data = self.raw_data

        print("Normalizing data...")
        # Log1p transform to handle heavy-tailed distribution
        log_data = np.log1p(data)

        # Per-series z-score normalization
        self.means = log_data.mean(axis=1, keepdims=True)
        self.stds = log_data.std(axis=1, keepdims=True)
        self.stds = np.maximum(self.stds, eps)  # Prevent division by zero

        self.normalized_data = (log_data - self.means) / self.stds
        print(f"Normalized data shape: {self.normalized_data.shape}")
        print(
            f"Value range: [{self.normalized_data.min():.2f}, {self.normalized_data.max():.2f}]"
        )

        return self.normalized_data

    def denormalize(
        self,
        normalized_data: np.ndarray,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reverse the normalization process.

        Args:
            normalized_data: Normalized data to denormalize
            means: Means used for normalization (uses stored if None)
            stds: Stds used for normalization (uses stored if None)

        Returns:
            Denormalized data in original scale
        """
        if means is None:
            means = self.means
        if stds is None:
            stds = self.stds

        if means is None or stds is None:
            raise ValueError("No normalization parameters. Call normalize first.")

        # Reverse z-score
        log_data = normalized_data * stds + means
        # Reverse log1p
        return np.expm1(log_data)

    def save_preprocessed(self, output_path: Path) -> None:
        """
        Save preprocessed data to a pickle file.

        Args:
            output_path: Path to save the preprocessed data
        """
        if self.normalized_data is None:
            raise ValueError("No normalized data. Call normalize first.")

        data = {
            "normalized_data": self.normalized_data,
            "page_titles": self.page_titles,
            "means": self.means,
            "stds": self.stds,
            "raw_data": self.raw_data,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved preprocessed data to {output_path}")

    @classmethod
    def load_preprocessed(cls, preprocessed_path: Path) -> "PageviewDataLoader":
        """
        Load preprocessed data from a pickle file.

        Args:
            preprocessed_path: Path to the preprocessed data file

        Returns:
            PageviewDataLoader instance with loaded data
        """
        with open(preprocessed_path, "rb") as f:
            data = pickle.load(f)

        loader = cls(dataset_path=preprocessed_path)
        loader.normalized_data = data["normalized_data"]
        loader.page_titles = data["page_titles"]
        loader.means = data["means"]
        loader.stds = data["stds"]
        loader.raw_data = data.get("raw_data")

        print(f"Loaded preprocessed data: {loader.normalized_data.shape}")
        return loader

    def get_page_index(self, page_title: str) -> int:
        """Get the index of a page by title."""
        try:
            return self.page_titles.index(page_title)
        except ValueError:
            raise ValueError(f"Page '{page_title}' not found in dataset")

    def search_pages(self, query: str, limit: int = 10) -> list[str]:
        """
        Search for pages by title substring.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching page titles
        """
        query_lower = query.lower()
        matches = [
            title for title in self.page_titles if query_lower in title.lower()
        ]
        return matches[:limit]
