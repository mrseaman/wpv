#!/usr/bin/env python3
"""Script to preprocess Wikipedia pageview data."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wpv_recommender.config import get_config
from wpv_recommender.data.preprocessing import PageviewDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia pageview data for the recommender model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input dataset (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output preprocessed data (default: from config)",
    )
    args = parser.parse_args()

    config = get_config()

    input_path = args.input or config.paths.dataset_path
    output_path = args.output or config.paths.preprocessed_path

    print("=" * 60)
    print("Wikipedia Pageview Data Preprocessing")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load raw data
    loader = PageviewDataLoader(
        dataset_path=input_path,
        seq_length=config.model.seq_length,
    )
    raw_data, page_titles = loader.load_raw_data()

    print()
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Number of pages: {len(page_titles)}")
    print(f"Sample page titles: {page_titles[:5]}")

    # Normalize
    normalized_data = loader.normalize()

    print()
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Mean: {normalized_data.mean():.4f}")
    print(f"Std:  {normalized_data.std():.4f}")

    # Save
    loader.save_preprocessed(output_path)

    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
