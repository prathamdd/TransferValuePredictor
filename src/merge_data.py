"""
Merges Understat player stats with Transfermarkt market values, season by
season, and writes the result to data/processed/.

Matching strategy:
  1. Normalize names on both sides (lowercase, strip accents, punctuation).
  2. Score every Understat player against every Transfermarkt player in the
     same season with rapidfuzz. Identical normalized names score 100.
  3. If both sources have a club column, add a small bonus when the clubs
     match. Club is a soft signal, never a hard join key.
  4. Assign matches one-to-one, highest score first, so one Transfermarkt
     row can't be claimed by two Understat players.
  5. High scores are accepted automatically. Middling scores go to a review
     file for a human to check. Low scores count as unmatched.

Manual fixes: create data/processed/manual_overrides.csv with columns
season,understat_name,tm_name and re-run. Those pairs are forced before any
fuzzy matching happens.

Outputs:
  data/processed/merged.csv       accepted matches (training data)
  data/processed/match_review.csv review-tier matches + unmatched regulars (audit log)

Usage:
    python src/merge_data.py
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SEASONS = [2019, 2020, 2021, 2022, 2023]

AUTO_ACCEPT_SCORE = 92   # at or above: accepted without review
REVIEW_FLOOR_SCORE = 78  # between this and AUTO_ACCEPT: review tier
                         # (raised from 75 -- true matches sat at 80-90,
                         # the only match at 75 was wrong)
CLUB_BONUS = 5           # added when clubs match in both sources
GK_MISMATCH_PENALTY = 20       # goalkeeper on one side, outfielder on the other
OUTFIELD_MISMATCH_PENALTY = 5  # e.g. defender vs forward; small, since
                               # Understat's position labels are coarse
REGULAR_MINUTES = 900    # unmatched players above this are worth reviewing

# Manual review of the first run found ~97% of review-tier matches correct,
# and the position check removes the known errors, so they go into the
# training data. match_review.csv is still written as an audit log.
INCLUDE_REVIEW_MATCHES = True

# Letters that NFKD normalization doesn't decompose into base + accent.
SPECIAL_LETTERS = str.maketrans({
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss",
    "æ": "ae", "Æ": "AE", "đ": "d", "Đ": "D", "ı": "i",
})
CLUB_STOPWORDS = {"fc", "afc", "the", "and"}


def normalize_name(name) -> str:
    """'Martin Ødegaard' -> 'martin odegaard', 'Heung-Min Son' -> 'heung min son'."""
    if not isinstance(name, str):
        return ""
    name = name.translate(SPECIAL_LETTERS)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^a-z\s]", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def normalize_clubs(clubs) -> list[str]:
    """'Brighton & Hove Albion,Arsenal FC' -> ['brighton hove albion', 'arsenal']."""
    if not isinstance(clubs, str):
        return []
    out = []
    for club in clubs.split(","):
        tokens = [t for t in normalize_name(club).split() if t not in CLUB_STOPWORDS]
        if tokens:
            out.append(" ".join(tokens))
    return out


def clubs_match(a: str, b: str) -> bool:
    """True if any club on side A plausibly equals any club on side B.
    partial_ratio lets 'brighton' match 'brighton hove albion'."""
    return any(
        fuzz.partial_ratio(x, y) >= 90
        for x in normalize_clubs(a)
        for y in normalize_clubs(b)
    )


def tm_position_groups(position) -> set[str]:
    """
    Map Transfermarkt's detailed position to coarse groups G/D/M/F.
    Hybrid roles get two groups so they agree with either Understat label
    (a winger is often 'M' on Understat, a defensive mid sometimes 'D').
    """
    if not isinstance(position, str):
        return set()
    p = position.lower()
    if "goalkeeper" in p:
        return {"G"}
    if "back" in p or "defender" in p or "sweeper" in p:
        return {"D"}
    if "defensive midfield" in p:
        return {"M", "D"}
    if "attacking midfield" in p:
        return {"M", "F"}
    if "midfield" in p:
        return {"M"}
    if "winger" in p:
        return {"F", "M"}
    if "forward" in p or "striker" in p:
        return {"F"}
    return set()


def understat_position_groups(position) -> set[str]:
    """Understat uses space-separated codes like 'F M S' or 'GK'. 'S' means
    'substitute' and isn't a position, so it's ignored."""
    if not isinstance(position, str):
        return set()
    return {tok[0] for tok in position.upper().split() if tok and tok[0] in "GDMF"}


def tm_primary_group(position) -> str | None:
    """One group per player for the training data, used as the model's
    position feature. Unlike tm_position_groups (which is deliberately loose
    for matching), this picks the single conventional group: defensive mids
    are M, wingers are F."""
    if not isinstance(position, str):
        return None
    p = position.lower()
    if "goalkeeper" in p:
        return "G"
    if "back" in p or "defender" in p or "sweeper" in p:
        return "D"
    if "midfield" in p:
        return "M"
    if "winger" in p or "forward" in p or "striker" in p:
        return "F"
    return None


def position_penalty(us_pos, tm_pos) -> float:
    us_g, tm_g = understat_position_groups(us_pos), tm_position_groups(tm_pos)
    if not us_g or not tm_g:
        return 0.0  # unknown on one side: no evidence either way
    if ("G" in us_g) != ("G" in tm_g):
        return GK_MISMATCH_PENALTY
    if not (us_g & tm_g):
        return OUTFIELD_MISMATCH_PENALTY
    return 0.0


def position_penalty_matrix(us_pos: pd.Series, tm_pos: pd.Series) -> np.ndarray:
    us_unique = list(us_pos.fillna("").unique())
    tm_unique = list(tm_pos.fillna("").unique())
    small = np.array([[position_penalty(a, b) for b in tm_unique] for a in us_unique])
    us_idx = us_pos.fillna("").map({c: i for i, c in enumerate(us_unique)}).to_numpy()
    tm_idx = tm_pos.fillna("").map({c: i for i, c in enumerate(tm_unique)}).to_numpy()
    return small[np.ix_(us_idx, tm_idx)]


def load_transfermarkt() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = RAW_DIR / f"transfermarkt_{season}.csv"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        frames.append(pd.read_csv(path))
    tm = pd.concat(frames, ignore_index=True)

    # No market value means nothing to train on.
    before = len(tm)
    tm = tm.dropna(subset=["market_value_eur"])
    print(f"Transfermarkt: dropped {before - len(tm)} rows with no market value.")

    # Mid-season transfers appear once per club with the same age and value.
    # Collapse them into one row and keep both clubs.
    before = len(tm)
    tm = (
        tm.groupby(["season", "name", "age", "market_value_eur"], as_index=False)
        .agg(
            position_raw=("position_raw", "first"),
            club=("club", lambda c: ",".join(sorted(set(c.dropna())))),
        )
    )
    print(f"Transfermarkt: collapsed {before - len(tm)} mid-season duplicate rows.")
    return tm


def load_understat() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = RAW_DIR / f"understat_{season}.csv"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def load_overrides() -> pd.DataFrame:
    path = PROCESSED_DIR / "manual_overrides.csv"
    if path.exists():
        print(f"Loaded manual overrides from {path}")
        return pd.read_csv(path)
    return pd.DataFrame(columns=["season", "understat_name", "tm_name"])


def name_score_matrix(us_names: list[str], tm_names: list[str]) -> np.ndarray:
    """
    token_sort_ratio handles reordered names ('Son Heung-Min' vs
    'Heung-Min Son'). token_set_ratio catches one name being a subset of the
    other ('Emerson' vs 'Emerson Royal'), but subsets are risky, so it's
    scaled to 90: below auto-accept unless the club also matches.
    """
    sort_scores = cdist(us_names, tm_names, scorer=fuzz.token_sort_ratio, workers=-1)
    set_scores = cdist(us_names, tm_names, scorer=fuzz.token_set_ratio, workers=-1)
    return np.maximum(sort_scores, 0.9 * set_scores)


def club_bonus_matrix(us_clubs: pd.Series, tm_clubs: pd.Series) -> np.ndarray:
    # Compare unique club strings once, then broadcast to the full matrix.
    us_unique = list(us_clubs.fillna("").unique())
    tm_unique = list(tm_clubs.fillna("").unique())
    small = np.array([[clubs_match(a, b) for b in tm_unique] for a in us_unique])
    us_idx = us_clubs.fillna("").map({c: i for i, c in enumerate(us_unique)}).to_numpy()
    tm_idx = tm_clubs.fillna("").map({c: i for i, c in enumerate(tm_unique)}).to_numpy()
    return small[np.ix_(us_idx, tm_idx)] * CLUB_BONUS


def match_season(us: pd.DataFrame, tm: pd.DataFrame, overrides: pd.DataFrame,
                 use_club: bool) -> list[dict]:
    us = us.reset_index(drop=True)
    tm = tm.reset_index(drop=True)
    us_norm = us["name"].map(normalize_name).tolist()
    tm_norm = tm["name"].map(normalize_name).tolist()

    scores = name_score_matrix(us_norm, tm_norm)
    if use_club:
        scores = scores + club_bonus_matrix(us["club"], tm["club"])
    scores = scores - position_penalty_matrix(us["position_raw"], tm["position_raw"])

    used_us, used_tm, matches = set(), set(), []

    # 1. Forced pairs from manual_overrides.csv. Compared after
    #    normalization so accents/case in the CSV don't matter.
    us_norm_s, tm_norm_s = pd.Series(us_norm), pd.Series(tm_norm)
    for _, row in overrides.iterrows():
        us_hits = us_norm_s.index[us_norm_s == normalize_name(row["understat_name"])].tolist()
        tm_hits = tm_norm_s.index[tm_norm_s == normalize_name(row["tm_name"])].tolist()
        if not us_hits or not tm_hits:
            side = "Understat" if not us_hits else "Transfermarkt"
            print(f"  WARNING: override {row['season']} "
                  f"'{row['understat_name']}' -> '{row['tm_name']}' not found on the {side} side.")
            continue
        i, j = us_hits[0], tm_hits[0]
        if i in used_us or j in used_tm:
            continue
        used_us.add(i); used_tm.add(j)
        matches.append({"us_idx": i, "tm_idx": j, "score": scores[i, j], "match_type": "manual"})

    # 2. Greedy one-to-one assignment, best scores first
    pairs = np.argwhere(scores >= REVIEW_FLOOR_SCORE)
    pair_scores = scores[pairs[:, 0], pairs[:, 1]]
    for k in np.argsort(-pair_scores, kind="stable"):
        i, j = pairs[k]
        if i in used_us or j in used_tm:
            continue
        used_us.add(i); used_tm.add(j)
        if us_norm[i] == tm_norm[j]:
            match_type = "exact"
        elif pair_scores[k] >= AUTO_ACCEPT_SCORE:
            match_type = "fuzzy"
        else:
            match_type = "review"
        matches.append({"us_idx": i, "tm_idx": j, "score": pair_scores[k], "match_type": match_type})

    # Two Understat players with the same name in one season (e.g. Emerson
    # Palmieri and Emerson Royal, both listed as "Emerson") can't be told
    # apart by name alone. Without club data, their matches may be swapped,
    # so they're flagged and kept out of training until club data exists.
    name_counts = Counter(us_norm)
    ambiguous_us = set() if use_club else {i for i, n in enumerate(us_norm) if name_counts[n] > 1}

    rows = []
    for m in matches:
        u, t = us.loc[m["us_idx"]], tm.loc[m["tm_idx"]]
        rows.append({
            "season": u["season"],
            "name": u["name"],
            "tm_name": t["name"],
            "club_understat": u.get("club"),
            "club_tm": t["club"],
            "position_understat": u["position_raw"],
            "position_tm": t["position_raw"],
            "position_group": tm_primary_group(t["position_raw"]),
            "age": t["age"],
            "appearances": u["appearances"],
            "minutes": u["minutes"],
            "goals": u["goals"],
            "assists": u["assists"],
            "market_value_eur": t["market_value_eur"],
            "match_type": "ambiguous" if m["us_idx"] in ambiguous_us else m["match_type"],
            "match_score": round(float(m["score"]), 1),
        })

    # Unmatched Understat players are recorded so they can be reviewed.
    for i in range(len(us)):
        if i not in used_us:
            u = us.loc[i]
            rows.append({
                "season": u["season"], "name": u["name"], "tm_name": None,
                "club_understat": u.get("club"), "minutes": u["minutes"],
                "match_type": "unmatched", "match_score": None,
            })
    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tm = load_transfermarkt()
    us = load_understat()
    overrides = load_overrides()

    use_club = "club" in us.columns and "club" in tm.columns
    print(f"Club signal: {'ON' if use_club else 'OFF (Understat CSVs have no club column yet)'}\n")

    all_rows = []
    for season in SEASONS:
        us_s = us[us["season"] == season]
        tm_s = tm[tm["season"] == season]
        if us_s.empty or tm_s.empty:
            continue
        ov_s = overrides[overrides["season"] == season]
        all_rows.extend(match_season(us_s, tm_s, ov_s, use_club))

    result = pd.DataFrame(all_rows)
    accepted_types = ["exact", "fuzzy", "manual"] + (["review"] if INCLUDE_REVIEW_MATCHES else [])
    accepted = result[result["match_type"].isin(accepted_types)]
    review = result[
        result["match_type"].isin(["review", "ambiguous"])
        | ((result["match_type"] == "unmatched") & (result["minutes"] >= REGULAR_MINUTES))
    ]

    accepted.drop(columns=["match_type", "match_score"]).to_csv(
        PROCESSED_DIR / "merged.csv", index=False)
    review.sort_values(["season", "match_type", "minutes"], ascending=[True, True, False]).to_csv(
        PROCESSED_DIR / "match_review.csv", index=False)

    print(f"{'season':<8}{'understat':>10}{'matched':>9}{'rate':>7}{'rate 900+ min':>15}"
          f"{'exact':>7}{'fuzzy':>7}{'review':>8}{'manual':>8}")
    for season in SEASONS:
        s = result[result["season"] == season]
        if s.empty:
            continue
        ok = s["match_type"].isin(accepted_types)
        regulars = s["minutes"] >= REGULAR_MINUTES
        rate_reg = ok[regulars].mean() if regulars.any() else float("nan")
        counts = s["match_type"].value_counts()
        print(f"{season:<8}{len(s):>10}{ok.sum():>9}{ok.mean():>7.1%}{rate_reg:>15.1%}"
              f"{counts.get('exact', 0):>7}{counts.get('fuzzy', 0):>7}{counts.get('review', 0):>8}"
              f"{counts.get('manual', 0):>8}")

    print(f"\nSaved {len(accepted)} rows to {PROCESSED_DIR / 'merged.csv'}")
    print(f"Saved {len(review)} rows to review in {PROCESSED_DIR / 'match_review.csv'}")


if __name__ == "__main__":
    main()