"""
Scrapes Premier League player market values from Transfermarkt for a given
range of seasons and saves one CSV per season to data/raw/.

Usage:
    python src/scrape_transfermarkt.py
"""

from __future__ import annotations

import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Transfermarkt blocks requests that don't look like a real browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1"

# Transfermarkt's saison_id is the year the season STARTS in.
# e.g. saison_id=2022 -> 2022/23 season.
SEASONS = [2019, 2020, 2021, 2022, 2023]

REQUEST_DELAY_SECONDS = 2  # be polite, avoid getting rate-limited/blocked


def parse_market_value(value_str: str) -> float | None:
    """Convert strings like '€45.00m' or '€800k' into a plain float (in euros)."""
    if not value_str:
        return None
    value_str = value_str.replace("€", "").strip()
    match = re.match(r"([\d.]+)([mk]?)", value_str)
    if not match:
        return None
    number, suffix = match.groups()
    number = float(number)
    if suffix == "m":
        return number * 1_000_000
    elif suffix == "k":
        return number * 1_000
    return number


def scrape_season(season_id: int, max_pages: int = 20) -> pd.DataFrame:
    """Scrape all pages of the market value list for one season."""
    all_players = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/plus/?saison_id={season_id}&page={page}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"  Page {page}: HTTP {response.status_code}, stopping.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="items")

        if table is None:
            print(f"  Page {page}: no table found, stopping.")
            break

        rows = table.find("tbody").find_all("tr", recursive=False)
        if not rows:
            print(f"  Page {page}: no rows, assuming last page.")
            break

        for row in rows:
            player = extract_player_from_row(row)
            if player:
                player["season"] = season_id
                all_players.append(player)

        print(f"  Page {page}: {len(rows)} rows scraped.")
        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(all_players)


def extract_player_from_row(row) -> dict | None:
    """
    NOTE: This selector logic is my best guess at Transfermarkt's current
    layout, but their HTML structure changes periodically. Treat this
    function as the first thing to fix if scraping comes back empty --
    we'll inspect real page HTML together and adjust the selectors.
    """
    try:
        name_tag = row.select_one("td.hauptlink a")
        name = name_tag.text.strip() if name_tag else None

        # Market value is typically in the last table cell.
        value_tag = row.select_one("td.rechts.hauptlink")
        market_value = parse_market_value(value_tag.text.strip()) if value_tag else None

        # Position is usually the second row of the inline table in that cell.
        position_tag = row.select_one("td:nth-of-type(2)")
        position = position_tag.text.strip() if position_tag else None

        # Age
        age_tag = row.select_one("td:nth-of-type(3)")
        age = age_tag.text.strip() if age_tag else None

        if not name:
            return None

        return {
            "name": name,
            "position_raw": position,
            "age": age,
            "market_value_eur": market_value,
        }
    except Exception as e:
        print(f"  Row parse error: {e}")
        return None


def main():
    for season in SEASONS:
        print(f"Scraping season {season}/{season + 1}...")
        df = scrape_season(season)
        out_path = f"data/raw/transfermarkt_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} players to {out_path}\n")


if __name__ == "__main__":
    main()