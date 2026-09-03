"""
Scrapes Premier League player standard stats (goals, assists, minutes,
position, age) from FBref for a given range of seasons and saves one CSV
per season to data/raw/.

Usage:
    python src/fetch_fbref_stats.py
"""

from __future__ import annotations

import time
import re
from io import StringIO
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# FBref season format is "YYYY-YYYY" (the season spanning two calendar years).
SEASONS = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

REQUEST_DELAY_SECONDS = 4  # FBref rate-limits aggressively; don't lower this

STANDARD_STATS_TABLE_ID = "stats_standard"


def get_table_from_comments(soup: BeautifulSoup, table_id: str) -> pd.DataFrame | None:
    """
    FBref hides many stats tables inside HTML comments so they don't show
    up in a naive parse. This pulls each comment out, re-parses it as HTML,
    and looks for the table we want inside it.
    """
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", id=table_id)
            if table is not None:
                return pd.read_html(StringIO(str(table)))[0]

    # Fallback: sometimes the table isn't commented out after all.
    table = soup.find("table", id=table_id)
    if table is not None:
        return pd.read_html(StringIO(str(table)))[0]

    return None


def clean_standard_stats(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    FBref's standard stats table has multi-level headers (two header rows),
    which pandas flattens awkwardly. This picks out and renames just the
    columns we actually need.
    """
    # Flatten multi-index columns like ('Unnamed: 0_level_0', 'Player') -> 'Player'
    df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]

    # The table repeats its header row periodically as a visual divider —
    # drop those repeated rows (they show up with 'Player' as the value).
    df = df[df["Player"] != "Player"]

    keep_and_rename = {
        "Player": "name",
        "Pos": "position_raw",
        "Age": "age",
        "Gls": "goals",
        "Ast": "assists",
        "Min": "minutes",
    }
    missing = [col for col in keep_and_rename if col not in df.columns]
    if missing:
        print(f"  WARNING: expected columns missing from FBref table: {missing}")

    df = df[[col for col in keep_and_rename if col in df.columns]]
    df = df.rename(columns=keep_and_rename)
    df["season"] = season

    # Numeric cleanup
    for col in ["age", "goals", "assists", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["name"]).reset_index(drop=True)


def scrape_season(season: str) -> pd.DataFrame:
    url = f"https://fbref.com/en/comps/9/{season}/stats/{season}-Premier-League-Stats"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} for {season}, skipping.")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    raw_table = get_table_from_comments(soup, STANDARD_STATS_TABLE_ID)

    if raw_table is None:
        print(f"  Could not find stats table for {season}.")
        return pd.DataFrame()

    return clean_standard_stats(raw_table, season)


def main():
    for season in SEASONS:
        print(f"Scraping FBref season {season}...")
        df = scrape_season(season)
        out_path = f"data/raw/fbref_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} players to {out_path}\n")
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()