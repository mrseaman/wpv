#!/usr/bin/env python3
"""Script to extract embeddings from the trained model and build FAISS index."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from wpv_recommender.config import get_config
from wpv_recommender.data.preprocessing import PageviewDataLoader
from wpv_recommender.data.dataset import create_inference_loader
from wpv_recommender.model.embedding_store import EmbeddingStore
from wpv_recommender.training.trainer import load_model_for_inference


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from trained model and build FAISS index."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to preprocessed data (default: from config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: best model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference",
    )
    args = parser.parse_args()

    config = get_config()

    data_path = args.data or config.paths.preprocessed_path
    checkpoint_path = args.checkpoint or config.paths.get_checkpoint_path(epoch=None)

    print("=" * 60)
    print("Wikipedia Pageview Recommender - Embedding Extraction")
    print("=" * 60)

    # Set device
    device = None
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, device=device)
    print(f"Model loaded: {model}")

    # Load data
    print(f"\nLoading data from {data_path}")
    data_loader = PageviewDataLoader.load_preprocessed(data_path)

    # Create inference loader
    inference_loader = create_inference_loader(
        data_loader,
        batch_size=args.batch_size,
        num_workers=0,  # Use main process for stability
    )

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Extracting"):
            batch = batch.to(device)
            embedding = model.get_embedding(batch)
            embeddings_list.append(embedding.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    store = EmbeddingStore(
        embedding_dim=config.model.embedding_dim,
        use_cosine=True,
    )
    store.add_embeddings(embeddings, data_loader.page_titles)

    # Save
    print("\nSaving embedding store...")
    store.save(
        index_path=config.paths.embedding_index_path,
        metadata_path=config.paths.embedding_metadata_path,
    )

    # Test search
    print("\n" + "=" * 60)
    print("Testing similarity search...")
    print("=" * 60)

    test_pages = data_loader.page_titles[:3]
    for page in test_pages:
        print(f"\nSimilar to '{page}':")
        results = store.search_by_title(page, k=5)
        for title, score in results:
            print(f"  {score:.4f} - {title}")

    print("\n" + "=" * 60)
    print("Embedding extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
