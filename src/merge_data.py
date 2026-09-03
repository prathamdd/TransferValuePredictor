"""Join Transfermarkt and FBref sources with fuzzy name matching."""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def merge_sources() -> None:
    """Merge raw sources into a cleaned dataset under data/processed/."""
    raise NotImplementedError


if __name__ == "__main__":
    merge_sources()
