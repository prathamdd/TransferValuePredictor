"""
Fetches each Premier League club's league points per season, as the
club-strength feature for the model (v2).

Source: football-data.co.uk, which publishes every result as a CSV per
season (no login, no Cloudflare). Replaces the Understat-based plan, since
Understat has been unreachable. Points are computed from the scores
(3 for a win, 1 for a draw), so the source's result column isn't needed.

Club names are short here ("Man City", "Wolves") and long in Transfermarkt
("Manchester City", "Wolverhampton Wanderers"). This script maps each
season's Transfermarkt clubs to football-data's names and saves the points
keyed by the Transfermarkt name, so train_model.py can join on club_tm
directly. If any club can't be mapped, nothing is saved for that season.

Output: data/raw/team_points_{season}.csv with columns
        season, club_tm, fd_name, points, rank

Run scrape_transfermarkt.py first (its club names are needed).

Usage:
    python src/fetch_team_points.py
"""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz

from merge_data import normalize_clubs

HEADERS = {"User-Agent": "transfer-value-predictor (student project)"}
SEASONS = [2019, 2020, 2021, 2022, 2023]
RAW_DIR = Path("data/raw")
REQUEST_DELAY_SECONDS = 2
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Short names that fuzzy matching can't link to the full name on its own.
FD_ALIASES = {
    "man city": "manchester city",
    "man united": "manchester united",
    "nott m forest": "nottingham forest",
    "wolves": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "sheffield weds": "sheffield wednesday",
}
MIN_MATCH_SCORE = 85


def season_code(season: int) -> str:
    """2023 -> '2324', the format in football-data's URLs."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def compute_table(results: pd.DataFrame) -> pd.DataFrame:
    results = results.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    home = pd.DataFrame({
        "team": results["HomeTeam"],
        "points": (results["FTHG"] > results["FTAG"]) * 3 + (results["FTHG"] == results["FTAG"]),
    })
    away = pd.DataFrame({
        "team": results["AwayTeam"],
        "points": (results["FTAG"] > results["FTHG"]) * 3 + (results["FTHG"] == results["FTAG"]),
    })
    table = pd.concat([home, away]).groupby("team").agg(
        points=("points", "sum"), matches=("points", "size")).reset_index()
    table = table.sort_values("points", ascending=False).reset_index(drop=True)
    table["rank"] = range(1, len(table) + 1)
    return table


def fd_key(name: str) -> str:
    clubs = normalize_clubs(name)
    key = clubs[0] if clubs else ""
    return FD_ALIASES.get(key, key)


def map_clubs(tm_clubs: list[str], fd_teams: list[str]) -> dict[str, str] | None:
    """Map each Transfermarkt club to one football-data team, best score
    first and one-to-one. Returns None (and prints why) if any club fails."""
    tm_keys = {c: (normalize_clubs(c) or [""])[0] for c in tm_clubs}
    fd_keys = {t: fd_key(t) for t in fd_teams}

    pairs = sorted(
        ((fuzz.partial_ratio(tm_keys[c], fd_keys[t]), c, t) for c in tm_clubs for t in fd_teams),
        reverse=True,
    )
    mapping, used = {}, set()
    for score, club, team in pairs:
        if score < MIN_MATCH_SCORE:
            break
        if club in mapping or team in used:
            continue
        mapping[club] = team
        used.add(team)

    unmapped = [c for c in tm_clubs if c not in mapping]
    if unmapped:
        print(f"  Could not map: {unmapped}")
        print(f"  Unused football-data names: {sorted(set(fd_teams) - used)}")
        print("  Add the short name to FD_ALIASES and re-run.")
        return None
    return mapping


def tm_clubs_for(season: int) -> list[str]:
    path = RAW_DIR / f"transfermarkt_{season}.csv"
    if not path.exists():
        print(f"  Missing {path}. Run scrape_transfermarkt.py first.")
        return []
    return sorted(pd.read_csv(path)["club"].dropna().unique())


def main():
    for season in SEASONS:
        print(f"Fetching results for {season}/{season + 1}...")
        tm_clubs = tm_clubs_for(season)
        if not tm_clubs:
            continue

        url = URL_TEMPLATE.format(code=season_code(season))
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} for {url}, skipping.")
            continue

        # The CSVs occasionally have stray encoding bytes; latin-1 never fails.
        results = pd.read_csv(StringIO(response.content.decode("latin-1")))
        table = compute_table(results)

        if len(table) != 20 or (table["matches"] != 38).any():
            print(f"  WARNING: expected 20 clubs with 38 matches each, got "
                  f"{len(table)} clubs, matches {sorted(table['matches'].unique())}")

        mapping = map_clubs(tm_clubs, table["team"].tolist())
        if mapping is None:
            continue
        # Show pairings where the names differ beyond "FC", so a wrong
        # fuzzy match is visible rather than silent.
        differing = {tm: fd for tm, fd in mapping.items()
                     if (normalize_clubs(tm) or [""])[0] != (normalize_clubs(fd) or [""])[0]}
        print("  Mapped: " + ", ".join(f"{tm} = {fd}" for tm, fd in sorted(differing.items())))

        out = table.rename(columns={"team": "fd_name"})
        reverse = {fd: tm for tm, fd in mapping.items()}
        out["club_tm"] = out["fd_name"].map(reverse)
        out["season"] = season
        out = out[["season", "club_tm", "fd_name", "points", "rank"]]

        out_path = RAW_DIR / f"team_points_{season}.csv"
        out.to_csv(out_path, index=False)
        top, bottom = out.iloc[0], out.iloc[-1]
        print(f"  Top: {top['club_tm']} ({top['points']} pts). "
              f"Bottom: {bottom['club_tm']} ({bottom['points']} pts).")
        print(f"  Saved to {out_path}\n")
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()