"""Predict a player's transfer value from input features."""

from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def predict_value(**features) -> float:
    """Return estimated market value for the given player features."""
    raise NotImplementedError


if __name__ == "__main__":
    predict_value()
