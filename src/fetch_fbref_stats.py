"""Fetch FBref stats: goals, assists, minutes, position, age."""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_player_stats(season: str) -> None:
    """Pull season stats and save under data/raw/."""
    raise NotImplementedError


if __name__ == "__main__":
    fetch_player_stats(season="2023")
