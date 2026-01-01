"""Uvicorn entrypoint for the rerank service."""

import uvicorn


def main() -> None:
    """Launch the FastAPI rerank service."""

    uvicorn.run(
        "services.rerank.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    main()
