"""Uvicorn entrypoint for the router service."""

import uvicorn


def main() -> None:
    """Launch the FastAPI router."""

    uvicorn.run(
        "services.router.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    main()
