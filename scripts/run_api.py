#!/usr/bin/env python3
"""Script to run the FastAPI server."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Run the Wikipedia Pageview Recommender API server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Wikipedia Pageview Recommender API")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "wpv_recommender.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
