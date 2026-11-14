"""
Trains a linear regression model to predict a player's Transfermarkt market
value from season stats, then evaluates it on a held-out season.

Design decisions (each explained where it's implemented):
  - Test set is the whole 2023/24 season. The model trains on 2019-2022
    and is scored on a season it has never seen. A random split would let
    the same player appear in both train and test (different seasons),
    which inflates the score.
  - Players under MIN_MINUTES are dropped: a 20-minute cameo says nothing
    about a player's ability, so those rows are mostly noise.
  - The target is log(market value). Values are heavily right-skewed (a few
    players worth 10-50x the median), and a log makes the model predict
    proportional differences, e.g. "each goal adds ~X%" not "+€Y".
  - age and (age - 25)^2 are both features, because value rises with age,
    peaks in the mid-20s, then falls. A straight line in age can't bend like
    that. Squaring distance from 25 (not raw age) keeps the two features
    from being near-duplicates, so both coefficients stay readable.
  - Position is one-hot encoded, with defenders as the reference category.
  - A naive baseline (predict each position's median value) is reported
    alongside the model, so the score has something to be compared against.
  - v2 adds club strength: the club's league points that season, from
    fetch_team_points.py. If those files exist, v1 (stats only) and v2 are
    trained on the same rows and scored side by side; otherwise only v1 runs.

Outputs:
  models/transfer_model.joblib     fitted pipeline (used by predict.py)
  outputs/predictions_2023.csv     test-season predictions per player
  outputs/predicted_vs_actual.png  the main chart

Usage:
    python src/train_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # render to file, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MERGED_PATH = Path("data/processed/merged.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TEST_SEASON = 2023
MIN_MINUTES = 450  # about five full matches

V1_NUMERIC = ["goals", "assists", "minutes", "age", "age_from_25_sq"]
V2_NUMERIC = V1_NUMERIC + ["club_points"]
AGE_CENTER = 25
TEAM_POINTS_DIR = Path("data/raw")
CATEGORICAL_FEATURES = ["position_group"]
TARGET = "market_value_eur"

POSITION_NAMES = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}
# Validated categorical palette (fixed order). Marker shapes are a second
# cue, since aqua and yellow sit below 3:1 contrast on a white background.
POSITION_STYLE = {
    "G": {"color": "#eda100", "marker": "s"},
    "D": {"color": "#2a78d6", "marker": "o"},
    "M": {"color": "#eb6834", "marker": "^"},
    "F": {"color": "#1baf7a", "marker": "D"},
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(MERGED_PATH)
    before = len(df)
    df = df[df["minutes"] >= MIN_MINUTES]
    df = df.dropna(subset=["goals", "assists", "minutes", "age"] + CATEGORICAL_FEATURES + [TARGET])
    df = df[df[TARGET] > 0].copy()
    print(f"Loaded {before} rows, kept {len(df)} with {MIN_MINUTES}+ minutes and complete data.")
    df = add_features(df)
    if "club_points" in df.columns:
        missing = df["club_points"].isna()
        if missing.any():
            print(f"Dropped {missing.sum()} rows with no club points "
                  f"(clubs: {sorted(df.loc[missing, 'club_tm'].dropna().unique())[:5]}).")
            df = df[~missing]
    return df


def load_team_points() -> dict[tuple[int, str], float] | None:
    """(season, Transfermarkt club name) -> league points, or None if
    fetch_team_points.py hasn't been run."""
    files = sorted(TEAM_POINTS_DIR.glob("team_points_*.csv"))
    if not files:
        return None
    tp = pd.concat([pd.read_csv(f) for f in files])
    return {(int(r.season), r.club_tm): float(r.points) for r in tp.itertuples()}


def club_points_for(season: int, clubs) -> float:
    """Mid-season transfers have club_tm like 'Chelsea FC,Manchester City';
    they get the average of their clubs' points."""
    points = _TEAM_POINTS or {}
    values = [points.get((int(season), c)) for c in str(clubs).split(",")]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else np.nan


_TEAM_POINTS = load_team_points()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. predict.py imports this, so training and prediction
    always compute them the same way."""
    df = df.copy()
    df["age_from_25_sq"] = (df["age"] - AGE_CENTER) ** 2
    if (_TEAM_POINTS is not None and "club_points" not in df.columns
            and {"season", "club_tm"} <= set(df.columns)):
        df["club_points"] = [club_points_for(s, c) for s, c in zip(df["season"], df["club_tm"])]
    return df


def build_pipeline(numeric_features: list[str]) -> Pipeline:
    # Scaling the numeric features doesn't change a linear model's
    # predictions, but it puts the coefficients on a comparable scale, so
    # they can be ranked by how much each feature matters.
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        # Categories listed explicitly so drop="first" removes defenders
        # (the largest group), which becomes the reference the other
        # positions are compared to.
        ("pos", OneHotEncoder(categories=[["D", "G", "M", "F"]], drop="first"),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def evaluate(name: str, y_true_eur: np.ndarray, y_pred_eur: np.ndarray) -> dict:
    pct_err = np.abs(y_pred_eur - y_true_eur) / y_true_eur
    metrics = {
        "model": name,
        "R² (log scale)": r2_score(np.log(y_true_eur), np.log(y_pred_eur)),
        "MAE (€m)": mean_absolute_error(y_true_eur, y_pred_eur) / 1e6,
        "median % error": np.median(pct_err) * 100,
        "within 50%": np.mean(pct_err <= 0.5) * 100,
    }
    return metrics


def print_coefficients(pipeline: Pipeline) -> None:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    print("\nCoefficients (target is log value, numeric features standardized):")
    print("  Numeric: effect of +1 standard deviation. Position: vs defenders.")
    for name, coef in sorted(zip(names, coefs), key=lambda x: -abs(x[1])):
        label = name.split("__", 1)[1]
        # exp(coef) - 1 turns a log-scale coefficient into a % change in value.
        print(f"  {label:<20}{coef:+.3f}   ({np.expm1(coef) * 100:+.0f}% value)")
    print("  age_from_25_sq is the curve: negative means value falls off on both sides of 25.")


def plot_predicted_vs_actual(test: pd.DataFrame, r2: float, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    actual = test[TARGET] / 1e6
    predicted = test["predicted_eur"] / 1e6

    for group, style in POSITION_STYLE.items():
        mask = test["position_group"] == group
        ax.scatter(
            actual[mask], predicted[mask],
            c=style["color"], marker=style["marker"], s=36, alpha=0.75,
            edgecolors="white", linewidths=0.8,  # surface ring on overlapping marks
            label=f"{POSITION_NAMES[group]} ({mask.sum()})",
        )

    lo = min(actual.min(), predicted.min()) * 0.8
    hi = max(actual.max(), predicted.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", linewidth=1.5, linestyle="--", zorder=0)
    ax.text(hi * 0.62, hi * 0.88, "perfect prediction", color="#6b6b66", fontsize=9, rotation=45,
            ha="center", va="center")

    # Label the five largest misses so readers can see who the model gets wrong.
    test = test.assign(miss=np.abs(np.log(test["predicted_eur"] / test[TARGET])))
    for _, row in test.nlargest(5, "miss").iterrows():
        ax.annotate(row["name"], (row[TARGET] / 1e6, row["predicted_eur"] / 1e6),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color="#3d3d3a")

    ax.set_xscale("log")
    ax.set_yscale("log")
    euro_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"€{v:g}m")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(euro_fmt)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual market value (log scale)", color="#3d3d3a")
    ax.set_ylabel("Predicted market value (log scale)", color="#3d3d3a")
    ax.set_title(f"Predicted vs actual market value, {TEST_SEASON}/{str(TEST_SEASON + 1)[2:]} season"
                 f"\n{label} · trained on 2019–{TEST_SEASON - 1} · R² = {r2:.2f} (log scale)",
                 loc="left", fontsize=12, color="#1a1a19")
    ax.grid(True, which="major", color="#e8e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#6b6b66")
    ax.legend(title="Position", frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON].copy()
    print(f"Train: {len(train)} rows (2019–{TEST_SEASON - 1}). Test: {len(test)} rows ({TEST_SEASON}).")

    versions = [("v1: stats only", V1_NUMERIC)]
    if "club_points" in df.columns:
        versions.append(("v2: + club points", V2_NUMERIC))
    else:
        print("No team_points_*.csv found, so only v1 runs. "
              "Run src/fetch_team_points.py to add club strength.")

    # Baseline: every player is worth his position's median training value.
    position_medians = train.groupby("position_group")[TARGET].median()
    test["baseline_eur"] = test["position_group"].map(position_medians)
    results = [evaluate("Baseline (position median)", test[TARGET].values, test["baseline_eur"].values)]

    # Every version is trained and scored on exactly the same rows.
    for label, numeric in versions:
        features = numeric + CATEGORICAL_FEATURES
        pipeline = build_pipeline(numeric).fit(train[features], np.log(train[TARGET]))
        # Predictions come out in log space; exp() converts back to euros.
        test["predicted_eur"] = np.exp(pipeline.predict(test[features]))
        results.append(evaluate(label, test[TARGET].values, test["predicted_eur"].values))

    # The last version is the one reported, plotted and saved.
    label, numeric = versions[-1]
    features = numeric + CATEGORICAL_FEATURES

    results = pd.DataFrame(results)
    print(f"\nTest-season results ({TEST_SEASON}):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n{label}")
    print_coefficients(pipeline)

    test["ratio"] = test["predicted_eur"] / test[TARGET]
    cols = ["name", "tm_name", "club_tm", "position_group", "age", "minutes", "goals", "assists",
            TARGET, "predicted_eur", "ratio"]
    test[cols].sort_values("ratio").to_csv(OUTPUT_DIR / f"predictions_{TEST_SEASON}.csv", index=False)

    show = test.assign(actual_m=test[TARGET] / 1e6, predicted_m=test["predicted_eur"] / 1e6)
    show_cols = ["name", "club_tm", "age", "goals", "assists", "actual_m", "predicted_m"]
    print("\nMost overvalued by the market (model predicts far less than actual):")
    print(show.nsmallest(5, "ratio")[show_cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\nMost undervalued by the market (model predicts far more than actual):")
    print(show.nlargest(5, "ratio")[show_cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    r2 = results.loc[results["model"] == label, "R² (log scale)"].iloc[0]
    plot_predicted_vs_actual(test, r2, label, OUTPUT_DIR / "predicted_vs_actual.png")

    # Refit on all seasons for predict.py, so the deployed model uses every row.
    # The scores above come from the train-only fit and stay honest.
    final = build_pipeline(numeric).fit(df[features], np.log(df[TARGET]))
    joblib.dump({"pipeline": final, "features": features, "min_minutes": MIN_MINUTES},
                MODEL_DIR / "transfer_model.joblib")

    print(f"\nSaved predictions to {OUTPUT_DIR / f'predictions_{TEST_SEASON}.csv'}")
    print(f"Saved chart to {OUTPUT_DIR / 'predicted_vs_actual.png'}")
    print(f"Saved model to {MODEL_DIR / 'transfer_model.joblib'}")


if __name__ == "__main__":
    main()