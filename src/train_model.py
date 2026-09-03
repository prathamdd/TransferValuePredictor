"""Train a linear regression model and evaluate performance."""

from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def train() -> None:
    """Fit linear regression on the processed dataset and report metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    train()
