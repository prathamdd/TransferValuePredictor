"""
Streamlit app: search any Premier League player from 2019/20 to 2023/24 and
see the model's expected market value next to their real one, or value a
made-up player (yourself) from a few sliders.

Run locally:
    streamlit run app.py

Needs, committed to the repo:
    data/processed/merged.csv
    data/raw/team_points_*.csv

The model is retrained from merged.csv when the app starts. A linear
regression on ~2,000 rows takes well under a second, and it means no saved
model file has to match the hosting server's scikit-learn version.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # the src/ scripts use paths relative to the repo root
sys.path.insert(0, str(ROOT / "src"))

from train_model import (  # noqa: E402
    CATEGORICAL_FEATURES, MERGED_PATH, MIN_MINUTES, POSITION_NAMES, TARGET,
    TEST_SEASON, V1_NUMERIC, V2_NUMERIC, add_features, build_pipeline,
    evaluate, load_data,
)

CHART_PATH = ROOT / "outputs" / "predicted_vs_actual.png"
DEFAULT_CLUB_POINTS = 55  # a typical mid-table club


# ---------- data and model ----------

@st.cache_resource
def get_model():
    """Score the model on the held-out season, then refit on every season
    for the lookups. Same procedure as train_model.py."""
    df = load_data()
    numeric = V2_NUMERIC if "club_points" in df.columns else V1_NUMERIC
    features = numeric + CATEGORICAL_FEATURES

    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    holdout = build_pipeline(numeric).fit(train[features], np.log(train[TARGET]))
    metrics = evaluate("model", test[TARGET].values,
                       np.exp(holdout.predict(test[features])))

    final = build_pipeline(numeric).fit(df[features], np.log(df[TARGET]))
    return final, features, metrics, len(train), len(test)


@st.cache_data
def get_players() -> pd.DataFrame:
    """Every player-season in merged.csv with the model's value attached,
    including short seasons the model wasn't trained on (flagged in the UI)."""
    model, features, *_ = get_model()
    df = add_features(pd.read_csv(MERGED_PATH))
    usable = df[features].notna().all(axis=1) & (df[TARGET] > 0)
    df["predicted_eur"] = np.nan
    df.loc[usable, "predicted_eur"] = np.exp(model.predict(df.loc[usable, features]))
    # Transfermarkt's name identifies the player (Understat lists both
    # Emerson Royal and Emerson Palmieri as just "Emerson").
    df["player"] = df["tm_name"].fillna(df["name"])
    return df


# ---------- formatting helpers ----------

def fmt_eur(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"€{value / 1e6:.1f}m" if value >= 1e6 else f"€{value / 1e3:.0f}k"


def season_label(season) -> str:
    return f"{int(season)}/{str(int(season) + 1)[2:]}"


def verdict(actual, predicted) -> str:
    if pd.isna(actual) or pd.isna(predicted):
        return "—"
    ratio = predicted / actual
    if ratio < 0.67:
        return "Market values him higher"
    if ratio > 1.5:
        return "Model values him higher"
    return "Roughly agree"


# ---------- page ----------

st.set_page_config(page_title="Transfer Value Predictor", page_icon="⚽", layout="centered")

model, features, metrics, n_train, n_test = get_model()
players = get_players()
has_club = "club_points" in features

st.title("Premier League Transfer Value Predictor")
st.caption(
    f"Expected market value for {players['player'].nunique():,} Premier League players, "
    f"2019/20–2023/24, from goals, assists, minutes, age, position"
    f"{' and club strength' if has_club else ''}. "
    f"On a season it never saw, the model explains {metrics['R² (log scale)']:.0%} "
    f"of the variation in market value."
)

tab_lookup, tab_worth, tab_how = st.tabs(["Look up a player", "What are you worth?", "How it works"])

# ----- Tab 1: player lookup -----
with tab_lookup:
    names = sorted(players["player"].dropna().unique())
    choice = st.selectbox("Player", names, index=None,
                          placeholder="Start typing a name, e.g. Saka")

    if choice:
        rows = players[players["player"] == choice].sort_values("season")
        latest = rows.iloc[-1]
        position = POSITION_NAMES.get(latest["position_group"], "")

        st.subheader(choice)
        st.caption(f"{position} · {latest['club_tm']}")

        c1, c2 = st.columns(2)
        c1.metric(f"Actual value, {season_label(latest['season'])}",
                  fmt_eur(latest[TARGET]))
        delta = None
        if pd.notna(latest["predicted_eur"]):
            delta = f"{(latest['predicted_eur'] / latest[TARGET] - 1) * 100:+.0f}% vs actual"
        c2.metric("Model's expected value", fmt_eur(latest["predicted_eur"]),
                  delta=delta, delta_color="off")
        st.markdown(f"**Verdict:** {verdict(latest[TARGET], latest['predicted_eur'])}")

        if len(rows) > 1:
            chart = pd.DataFrame({
                "Season": rows["season"].map(season_label),
                "Actual": rows[TARGET] / 1e6,
                "Model": rows["predicted_eur"] / 1e6,
            }).set_index("Season")
            st.line_chart(chart, color=["#2a78d6", "#eb6834"])
            st.caption("Market value in € millions, by season.")

        table = pd.DataFrame({
            "Season": rows["season"].map(season_label),
            "Club": rows["club_tm"],
            "Age": rows["age"].astype("Int64"),
            "Minutes": [f"{m:,.0f}" + ("*" if m < MIN_MINUTES else "") for m in rows["minutes"]],
            "Goals": rows["goals"].astype("Int64"),
            "Assists": rows["assists"].astype("Int64"),
            "Actual": rows[TARGET].map(fmt_eur),
            "Model": rows["predicted_eur"].map(fmt_eur),
            "Verdict": [verdict(a, p) for a, p in zip(rows[TARGET], rows["predicted_eur"])],
        })
        st.dataframe(table, hide_index=True, width="stretch")

        notes = []
        if (rows["minutes"] < MIN_MINUTES).any():
            notes.append(f"\\* Under {MIN_MINUTES} minutes. The model wasn't trained on "
                         "seasons this short, so treat that value loosely.")
        notes.append("The model was trained on all five seasons, including this player's, "
                     "so this shows how it reads him rather than a test score.")
        st.caption(" ".join(notes))

# ----- Tab 2: custom stats -----
with tab_worth:
    st.write("Set a season's stats and see what the model thinks that player is worth.")
    c1, c2 = st.columns(2)
    with c1:
        position = st.selectbox("Position", list(POSITION_NAMES), index=3,
                                format_func=POSITION_NAMES.get)
        age = st.slider("Age", 16, 40, 23)
        minutes = st.slider("Minutes played (a full season is 3,420)", 0, 3420, 2400, step=30)
    with c2:
        goals = st.slider("Goals", 0, 40, 10)
        assists = st.slider("Assists", 0, 25, 5)
        club_points = None
        if has_club:
            club_points = st.slider(
                "Club's league points", 15, 100, DEFAULT_CLUB_POINTS,
                help="About 90 for a title winner, 55-60 mid-table, 30 or less relegated.")

    row = {"goals": goals, "assists": assists, "minutes": minutes, "age": age,
           "position_group": position}
    if has_club:
        row["club_points"] = club_points
    value = float(np.exp(model.predict(add_features(pd.DataFrame([row]))[features]))[0])

    st.metric("Model's expected market value", fmt_eur(value))
    if minutes < MIN_MINUTES:
        st.caption(f"Under {MIN_MINUTES} minutes is outside what the model was trained on, "
                   "so this is a rough guess.")

    # Real players the market actually valued at about this level.
    recent = players[
        (players["season"] == players["season"].max())
        & (players["position_group"] == position)
        & (players["minutes"] >= MIN_MINUTES)
    ]
    if not recent.empty:
        closest = recent.iloc[(np.log(recent[TARGET] / value)).abs().argsort()[:3]]
        st.write(f"**{POSITION_NAMES[position]}s actually valued around {fmt_eur(value)} "
                 f"in {season_label(players['season'].max())}:**")
        for _, p in closest.iterrows():
            st.write(f"- {p['player']} ({p['club_tm']}): {fmt_eur(p[TARGET])}")

# ----- Tab 3: about -----
with tab_how:
    st.subheader("How well it works")
    st.write(
        f"Trained on 2019/20–{season_label(TEST_SEASON - 1)} ({n_train:,} player-seasons) "
        f"and tested on {season_label(TEST_SEASON)} ({n_test:,}), a season the model never saw. "
        f"Only players with {MIN_MINUTES}+ minutes count."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (log scale)", f"{metrics['R² (log scale)']:.2f}")
    m2.metric("Median error", f"{metrics['median % error']:.0f}%")
    m3.metric("Within 50% of actual", f"{metrics['within 50%']:.0f}%")

    if CHART_PATH.exists():
        st.image(str(CHART_PATH), width="stretch")

    st.subheader("What it can't see")
    st.write(
        "- **Injuries and bans.** The market prices ability, not one season's availability, "
        "so stars who missed much of a season come out low.\n"
        "- **Potential.** Young players are priced partly on what they might become.\n"
        "- **Relegation.** A bottom club's squad players are worth less than their "
        "points alone suggest.\n\n"
        "Market values are Transfermarkt's crowd-sourced estimates, not transfer fees."
    )