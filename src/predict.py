"""
"What are you worth?" -- uses the trained model to value a player.

Two modes:
  1. Look up a real player and compare the model's value with Transfermarkt's:
       python src/predict.py --player "Bukayo Saka" --season 2023
     (--season is optional; without it, the player's latest season is used.)

  2. Enter any stats, e.g. your own:
       python src/predict.py --goals 12 --assists 6 --minutes 2400 --age 23 --position F
     Positions: G (goalkeeper), D (defender), M (midfielder), F (forward).

Run src/train_model.py first; it saves the model this script loads.

Note on lookups: the saved model was refit on all five seasons, so it has
already seen any player you look up. Treat a lookup as "how the model
reads this player", not as a test score. For out-of-sample numbers, see
outputs/predictions_2023.csv, written by the train-only fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from merge_data import normalize_name
from train_model import POSITION_NAMES, add_features

MODEL_PATH = Path("models/transfer_model.joblib")
MERGED_PATH = Path("data/processed/merged.csv")


def fmt_eur(value: float) -> str:
    return f"€{value / 1e6:.1f}m" if value >= 1e6 else f"€{value / 1e3:.0f}k"


def predict_value(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    X = add_features(rows)[bundle["features"]]
    return np.exp(bundle["pipeline"].predict(X))


def list_ambiguous(df: pd.DataFrame, query: str, tm_names: list[str]) -> None:
    print(f"Several players match '{query}':")
    for n in sorted(tm_names):
        seasons = ", ".join(str(s) for s in sorted(df.loc[df["tm_name"] == n, "season"]))
        print(f"  {n}  (seasons: {seasons})")
    print("Re-run with the full name.")
    sys.exit(1)


def find_player(df: pd.DataFrame, query: str) -> str | None:
    """
    Find a player and return his Transfermarkt name, which is the identity
    key: Understat lists Emerson Palmieri and Emerson Royal both as
    'Emerson', while Transfermarkt keeps them apart.

    Tried in order, so the first tier with a unique hit wins:
      1. exact Transfermarkt name ('Emerson' -> Emerson Palmieri)
      2. exact Understat name
      3. every query word is a whole word of the name ('Saka' -> Bukayo
         Saka, not Aaron Wan-Bissaka)
      4. fuzzy match ('odegard' -> Martin Ødegaard)
    If several players tie within a tier, list them rather than guess.
    """
    q = normalize_name(query)
    q_words = set(q.split())
    pairs = df[["name", "tm_name"]].drop_duplicates()
    tm_norm = pairs["tm_name"].map(normalize_name)
    us_norm = pairs["name"].map(normalize_name)

    tiers = [
        tm_norm == q,
        us_norm == q,
        tm_norm.map(lambda n: q_words <= set(n.split()))
        | us_norm.map(lambda n: q_words <= set(n.split())),
    ]
    for mask in tiers:
        hits = pairs.loc[mask, "tm_name"].unique().tolist()
        if len(hits) == 1:
            others = pairs.loc[(us_norm == q) & (pairs["tm_name"] != hits[0]), "tm_name"].unique()
            if len(others):
                print(f"Note: Understat also lists {', '.join(others)} as '{query}'. "
                      "Use the full name to see them.")
            return hits[0]
        if len(hits) > 1:
            list_ambiguous(df, query, hits)

    def score(norm_name: str) -> float:
        # A misspelled surname scores poorly against the full name
        # ('odegard' vs 'martin odegaard' is 77), so a one-word query is
        # also compared word by word ('odegard' vs 'odegaard' is 93).
        best = fuzz.WRatio(q, norm_name)
        if len(q_words) == 1:
            best = max([best] + [fuzz.ratio(q, w) for w in norm_name.split()])
        return best

    scores = {n: score(normalize_name(n)) for n in pairs["tm_name"].unique()}
    top = max(scores.values(), default=0)
    if top < 80:
        return None
    tied = [n for n, s in scores.items() if s == top]
    if len(tied) > 1:
        list_ambiguous(df, query, tied)
    return tied[0]


def lookup_mode(bundle: dict, query: str, season: int | None) -> None:
    df = pd.read_csv(MERGED_PATH)
    name = find_player(df, query)
    if name is None:
        print(f"No player found matching '{query}'.")
        sys.exit(1)

    rows = df[df["tm_name"] == name].sort_values("season")
    if season is not None:
        rows = rows[rows["season"] == season]
        if rows.empty:
            seasons = ", ".join(str(s) for s in df.loc[df["tm_name"] == name, "season"])
            print(f"{name} has no {season} season in the data. Available: {seasons}")
            sys.exit(1)

    rows = rows.copy()
    rows["predicted_eur"] = predict_value(bundle, rows)

    print(f"\n{name} ({POSITION_NAMES.get(rows['position_group'].iloc[-1], '?')})\n")
    print(f"{'season':<9}{'club':<22}{'age':>4}{'min':>6}{'G':>4}{'A':>4}"
          f"{'actual':>10}{'model':>10}   verdict")
    for _, r in rows.iterrows():
        ratio = r["predicted_eur"] / r["market_value_eur"]
        if ratio < 0.67:
            verdict = "market values him higher"
        elif ratio > 1.5:
            verdict = "model values him higher"
        else:
            verdict = "roughly agree"
        low_minutes = "*" if r["minutes"] < bundle["min_minutes"] else " "
        club = str(r.get("club_tm", ""))[:21]
        print(f"{r['season']:<9}{club:<22}{r['age']:>4.0f}{r['minutes']:>5.0f}{low_minutes}"
              f"{r['goals']:>4.0f}{r['assists']:>4.0f}{fmt_eur(r['market_value_eur']):>10}"
              f"{fmt_eur(r['predicted_eur']):>10}   {verdict}")

    if (rows["minutes"] < bundle["min_minutes"]).any():
        print(f"\n* under {bundle['min_minutes']} minutes: the model wasn't trained on "
              "seasons this short, so treat that prediction loosely.")
    print("\nThe model only sees goals, assists, minutes, age and position. "
          "It can't see club strength, so big-club players tend to come out low.")
    print("The saved model was trained on all five seasons, including these rows, so this "
          "shows how the model reads him, not a test score. Out-of-sample results are in "
          "outputs/predictions_2023.csv.")


def custom_mode(bundle: dict, args: argparse.Namespace) -> None:
    row = pd.DataFrame([{
        "goals": args.goals, "assists": args.assists, "minutes": args.minutes,
        "age": args.age, "position_group": args.position,
    }])
    value = predict_value(bundle, row)[0]
    print(f"\nA {args.age}-year-old {POSITION_NAMES[args.position].lower()} with "
          f"{args.goals} goals and {args.assists} assists in {args.minutes} minutes:")
    print(f"  Model's market value: {fmt_eur(value)}")
    if args.minutes < bundle["min_minutes"]:
        print(f"  (Under {bundle['min_minutes']} minutes is outside what the model "
              "was trained on, so this is a rough guess.)")


def main():
    parser = argparse.ArgumentParser(description="What is a player worth, according to the model?")
    parser.add_argument("--player", help="look up a real player by name")
    parser.add_argument("--season", type=int, help="season start year, e.g. 2023 for 2023/24")
    parser.add_argument("--goals", type=int)
    parser.add_argument("--assists", type=int)
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--age", type=int)
    parser.add_argument("--position", choices=list(POSITION_NAMES), type=str.upper)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"No model at {MODEL_PATH}. Run src/train_model.py first.")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)

    custom_fields = [args.goals, args.assists, args.minutes, args.age, args.position]
    if args.player:
        lookup_mode(bundle, args.player, args.season)
    elif all(v is not None for v in custom_fields):
        custom_mode(bundle, args)
    else:
        parser.error("use --player NAME, or give all of --goals --assists --minutes --age --position")


if __name__ == "__main__":
    main()