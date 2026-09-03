"""
Scrapes Premier League squad market values from Transfermarkt for a given
range of seasons and saves one CSV per season to data/raw/.

v2 -- rewritten after v1's approach (the global "marktwerte" top-values
ranking page) turned out to ignore saison_id entirely and return current
values with heavy duplication and thin coverage (~100 unique names per
"season", ~2-10% overlap with Understat).

This version scrapes per-club squad ("Kader") pages instead, which do
respect saison_id and give full squad coverage with no duplicates:
  1. get_season_clubs()  -- get the 20 club IDs/slugs for a season from
     the league overview page.
  2. scrape_club_squad() -- get every player's name, position, age, and
     market value from one club's squad page for that season.

Usage:
    python src/scrape_transfermarkt.py           # full run, all seasons
    python src/scrape_transfermarkt.py --test     # ONE club, ONE season only
"""

from __future__ import annotations

import re
import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

SEASONS = [2019, 2020, 2021, 2022, 2023]
REQUEST_DELAY_SECONDS = 2
LEAGUE_URL = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
SQUAD_URL_TEMPLATE = "https://www.transfermarkt.com/{slug}/kader/verein/{club_id}/saison_id/{season}"


def parse_market_value(value_str: str) -> float | None:
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


def get_season_clubs(season: int) -> list[dict]:
    """Get the club id/slug/name for every club in the league that season."""
    url = f"{LEAGUE_URL}/plus/?saison_id={season}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} fetching club list for {season}.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    club_links = soup.select("a[href*='/startseite/verein/']")

    clubs = {}
    for a in club_links:
        href = a.get("href", "")
        match = re.search(r"/([\w-]+)/startseite/verein/(\d+)", href)
        if not match:
            continue
        slug, club_id = match.groups()
        name = a.get("title") or a.text.strip()
        if name and club_id not in clubs:
            clubs[club_id] = {"slug": slug, "id": club_id, "name": name}

    return list(clubs.values())


def extract_age(row) -> int | None:
    """
    On squad ('Kader') pages, age is a plain integer (not a birth-date
    string with parens like the old marktwerte page used). The first
    'zentriert' cell is the shirt number, the second is age -- both are
    bare digits, so we take the second digit-only zentriert cell as age,
    with a sanity range check to avoid grabbing an unrelated numeric cell.
    """
    zentriert_cells = [c.get_text(strip=True) for c in row.select("td.zentriert")]
    numeric_cells = [t for t in zentriert_cells if t.isdigit()]

    if len(numeric_cells) >= 2:
        candidate = int(numeric_cells[1])
    elif len(numeric_cells) == 1:
        candidate = int(numeric_cells[0])
    else:
        return None

    return candidate if 14 <= candidate <= 50 else None


def extract_player_from_row(row) -> dict | None:
    """
    NOTE: same caveat as before -- these selectors are my best guess at
    the squad page's current layout, based on Transfermarkt's general
    conventions (posrela cell for name+position, rechts hauptlink for
    value). Verify with --test before trusting a full run.
    """
    try:
        name_tag = row.select_one("td.posrela .hauptlink a") or row.select_one("td.hauptlink a")
        name = name_tag.text.strip() if name_tag else None
        if not name:
            return None

        # Position is usually the second row of a small inline table
        # inside the same cell as the player's name/photo.
        position_tag = row.select_one("td.posrela table.inline-table tr:nth-of-type(2) td")
        position = position_tag.text.strip() if position_tag else None

        age = extract_age(row)

        value_tag = row.select_one("td.rechts.hauptlink")
        market_value = parse_market_value(value_tag.get_text(strip=True)) if value_tag else None

        return {
            "name": name,
            "position_raw": position,
            "age": age,
            "market_value_eur": market_value,
        }
    except Exception as e:
        print(f"  Row parse error: {e}")
        return None


def scrape_club_squad(club: dict, season: int) -> list[dict]:
    url = SQUAD_URL_TEMPLATE.format(slug=club["slug"], club_id=club["id"], season=season)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"    HTTP {response.status_code} for {club['name']} {season}.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        print(f"    No squad table found for {club['name']} {season}.")
        return []

    rows = table.find("tbody").find_all("tr", recursive=False)
    players = []
    for row in rows:
        player = extract_player_from_row(row)
        if player:
            player["club"] = club["name"]
            player["season"] = season
            players.append(player)

    return players


def scrape_season(season: int, club_limit: int | None = None) -> pd.DataFrame:
    print(f"  Getting club list for {season}...")
    clubs = get_season_clubs(season)
    print(f"  Found {len(clubs)} clubs.")
    time.sleep(REQUEST_DELAY_SECONDS)

    if club_limit:
        clubs = clubs[:club_limit]

    all_players = []
    for club in clubs:
        print(f"  Scraping {club['name']}...")
        players = scrape_club_squad(club, season)
        print(f"    {len(players)} players.")
        all_players.extend(players)
        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(all_players)


def main():
    test_mode = "--test" in sys.argv
    seasons = SEASONS[:1] if test_mode else SEASONS
    club_limit = 1 if test_mode else None

    if test_mode:
        print("TEST MODE: one club, one season only.\n")

    for season in seasons:
        print(f"Scraping season {season}/{season + 1}...")
        df = scrape_season(season, club_limit=club_limit)
        suffix = "_TEST" if test_mode else ""
        out_path = f"data/raw/transfermarkt_{season}{suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} players to {out_path}\n")


if __name__ == "__main__":
    main()