"""
Scrapes Premier League player season stats (goals, assists, minutes,
position) from Understat for a given range of seasons and saves one CSV
per season to data/raw/.

History:
- v1 used FBref -- blocked by Cloudflare (HTTP 403 on every request).
- v2 used Understat's old inline `playersData = JSON.parse(...)` pattern
  embedded in the league page HTML -- that pattern is gone; Understat now
  loads player data via an internal AJAX endpoint instead.
- v3 (this version) hits that AJAX endpoint directly. No HTML parsing or
  hex-escape decoding needed anymore.

Note: Understat does not provide player age -- we get that from the
Transfermarkt data instead during the merge step.

Usage:
    python src/fetch_understat_stats.py
"""

from __future__ import annotations

import time
import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    # Some internal AJAX endpoints reject requests without a Referer that
    # matches the site, since they're meant to be called from the page's
    # own JS, not hit directly. Setting this defensively.
    "Referer": "https://understat.com/",
    "X-Requested-With": "XMLHttpRequest",
}

SEASONS = [2019, 2020, 2021, 2022, 2023]
REQUEST_DELAY_SECONDS = 2
LEAGUE = "EPL"


def scrape_season(season: int) -> pd.DataFrame:
    url = f"https://understat.com/getLeagueData/{LEAGUE}/{season}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} for {season}, skipping.")
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError:
        print(f"  Response for {season} was not valid JSON, skipping.")
        return pd.DataFrame()

    players = data.get("players")
    if not players:
        print(f"  No 'players' key in response for {season}.")
        return pd.DataFrame()

    df = pd.DataFrame(players)

    keep_and_rename = {
        "player_name": "name",
        "team_title": "club",
        "position": "position_raw",
        "games": "appearances",
        "time": "minutes",
        "goals": "goals",
        "assists": "assists",
    }
    missing = [col for col in keep_and_rename if col not in df.columns]
    if missing:
        print(f"  WARNING: expected fields missing from Understat data: {missing}")

    df = df[[col for col in keep_and_rename if col in df.columns]]
    df = df.rename(columns=keep_and_rename)
    df["season"] = season

    for col in ["appearances", "minutes", "goals", "assists"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    for season in SEASONS:
        print(f"Scraping Understat season {season}/{season + 1}...")
        df = scrape_season(season)
        out_path = f"data/raw/understat_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} players to {out_path}\n")
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()