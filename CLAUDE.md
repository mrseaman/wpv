# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wikipedia Pageview Recommender - A CNN Autoencoder-based recommendation system that learns page embeddings from Wikipedia pageview patterns to find similar pages. Includes original course analysis code for event detection using FFT and Poisson modeling.

## Project Structure

```
wpv/
├── wpv_recommender/     # Main recommender package
│   ├── api/             # FastAPI recommendation service
│   ├── data/            # Data preprocessing and dataset
│   ├── model/           # CNN Autoencoder and embedding store
│   └── training/        # Training loop with early stopping
├── scripts/             # CLI scripts for the pipeline
├── data/                # Data files (not tracked in git)
├── models/              # Trained model checkpoints (not tracked)
├── embeddings/          # FAISS index and metadata (not tracked)
├── course/              # Original course materials
│   ├── wiki_pageviews.py
│   ├── wiki-pageviews.ipynb
│   └── event_fit.py
└── logs/                # Training logs
```

## Running the Code

```bash
# Recommender system pipeline
python scripts/preprocess_data.py    # Preprocess pageview data
python scripts/train_model.py        # Train CNN Autoencoder
python scripts/extract_embeddings.py # Extract page embeddings to FAISS index
python scripts/run_api.py            # Start FastAPI recommendation server

# Original course analysis (in course/ directory)
jupyter notebook course/wiki-pageviews.ipynb
```

Dependencies: numpy, pandas, matplotlib, torch, faiss-cpu, fastapi, uvicorn, tqdm

## Recommender System (wpv_recommender/)

### Architecture

- **wpv_recommender/config.py** - Model, training, and path configuration
- **wpv_recommender/data/** - Data preprocessing and PyTorch dataset
- **wpv_recommender/model/** - CNN Autoencoder and embedding store
- **wpv_recommender/training/** - Training loop with early stopping
- **wpv_recommender/api/** - FastAPI recommendation service

### Model Details

The `CNNAutoencoder` learns embeddings from 90-day (2160 hourly) pageview sequences:
- **Encoder**: 7 strided 1D convolutions (channels: 64→128→256→256→512→512→512) with BatchNorm, GELU, reducing spatial dim to 17, then two-layer MLP bottleneck (8704→1024→512) to embedding
- **Decoder**: Two-layer MLP expansion (512→1024→8704), then transposed 1D convolutions mirroring the encoder back to original sequence length
- **Training**: MSE reconstruction loss, ReduceLROnPlateau scheduler (factor=0.5, patience=5, min_lr=1e-6), AdamW (lr=3e-4), early stopping (patience=20)

### API Endpoints

- `GET /recommendations/{page_title}` - Get similar pages (k results)
- `GET /search?q=query` - Search pages by title substring
- `GET /page/{page_title}` - Get embedding vector for a page
- `GET /pages` - List all indexed pages (paginated)
- `GET /health` - Service health check

### Data Files (in data/, not tracked in git)

- `wiki-en-2019-0[1-3].pkl` - Raw monthly pageview data
- `dataset_raw.pkl` / `dataset_20k.pkl` - Merged/filtered datasets
- `preprocessed_data.pkl` - Normalized pageview sequences

### Generated Files (not tracked in git)

- `models/best_model.pt` - Trained model checkpoint
- `embeddings/faiss_index.bin` - FAISS similarity index
- `embeddings/metadata.pkl` - Page titles and embeddings

## Original Course Analysis (course/)

Event detection using signal processing and statistical modeling on Q1 2019 English Wikipedia data.

### Pipeline

1. **Data Loading** (`load_data_from_archive`) - Loads/merges monthly pickle files
2. **Hourly Parsing** (`parse_hourly_views`) - Decodes Wikimedia's compressed hourly format
3. **Event Extraction** (`extract_event`) - FFT low-pass filter removes daily/weekly patterns
4. **Event Detection** (`event_detection`) - Identifies pages with coefficient of variation > 0.3
5. **Statistical Fitting** (`event_poisson_MLE`) - Fits peaks to Poisson distribution

### Key Technical Details

- **Hourly encoding**: Hours A-X (0-23), Days A-Z+ (1-31)
- **FFT cutoff**: 0.005 Hz removes periodicities (daily=0.042, weekly=0.006)
- **Event threshold**: Coefficient of variation > 0.3 indicates significant event
