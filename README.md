# Premier League Transfer Value Predictor

A linear regression model that predicts a Premier League player's market value from five simple stats (goals, assists, minutes, age and position), trained on five seasons of scraped data and tested on a season it never saw.

**Result:** on the held-out 2023/24 season the model explains 55% of the variation in (log) market value, against 5% for a naive baseline. Its biggest misses show that **club strength**, which the model can't see, drives value more than any stat it can.

![Predicted vs actual market value, 2023/24](outputs/predicted_vs_actual.png)

## Results

Trained on 2019/20–2022/23 (1,591 player-seasons), tested on 2023/24 (400 player-seasons). Only players with 450+ minutes are included.

| Model | R² (log scale) | Mean abs. error | Median % error | Within 50% of actual |
|---|---|---|---|---|
| Baseline: median value for the player's position | 0.05 | €16.4m | 60% | 42% |
| **Linear regression** | **0.55** | **€12.1m** | **46%** | **55%** |

### What the model learned

Coefficients are on the log scale, so they read as percentage changes. Each is the effect of a one-standard-deviation increase.

| Feature | Effect on value |
|---|---|
| Age curve, (age − 25)² | −30%: value peaks in the mid-20s and falls on both sides |
| Goals | +30% |
| Minutes | +25% |
| Assists | +19% |

Age matters most. The goalkeeper coefficient (+17% vs defenders) is likely noise, since only about 30 keepers qualify each season.

### Where it's wrong, and why

**Most overvalued by the market** (actual far above predicted): John Stones, Jack Grealish, Rúben Dias and Kevin De Bruyne of Manchester City, plus Lisandro Martínez of Manchester United. Several played limited minutes (Stones 1,060, Martínez 656) but are priced as elite players.

**Most undervalued** (predicted far above actual): all from Sheffield United and Luton Town, the two promoted clubs that finished at the bottom.

The model only sees minutes and goals, so it can't tell a Luton regular from a City regular. **Club strength is the largest missing feature.**

Two smaller effects:

- **Potential.** Players aged 21 or under get a median predicted/actual ratio of 0.74, compared with 0.83 overall. In 2020 the market priced Bukayo Saka at €65m on 5 goals, while the model said €26m. It caught up as his output grew.
- **Market drift.** Predictions run low across the board (median ratio 0.83). The likely cause is market values rising over time, which a model trained on earlier seasons can't anticipate.

## How it works

```
Understat (stats) ──┐
                    ├─► merge_data.py ─► train_model.py ─► predict.py
Transfermarkt ──────┘    name matching     regression       "what are you worth?"
(market values)
```

### 1. Data collection, and what broke

Getting clean historical data took more work than the model did.

| Source | Attempt | Problem | Fix |
|---|---|---|---|
| football-data.org | Original plan | Free tier has no assists, minutes or market values | Dropped |
| FBref (stats) | `requests` + BeautifulSoup | Cloudflare JS challenge: HTTP 403 on every request, even with `cloudscraper` and TLS impersonation | Switched to Understat |
| Understat (stats) | Parse `playersData` JSON embedded in the page | The site moved player data out of the HTML | Found the internal AJAX endpoint (`/getLeagueData/EPL/{season}`) in browser dev tools and called it directly |
| Transfermarkt (values) | Global "most valuable players" ranking | Ignores its `saison_id` parameter, so every "season" returned today's values; also only ~100 unique players | Scraped each club's squad page per season instead (20 clubs × 5 seasons) |
| Transfermarkt (values) | Value parser | Site switched to `€35.00M`; parser expected lowercase `m`, so €35m was saved as 35 | Case-insensitive parsing, full re-scrape |

**Validation that values are really historical:** Erling Haaland first appears in 2022, when he joined Manchester City. Of the 255 players present in both 2019 and 2023, only 0.8% have the same value.

### 2. Matching players across sources

The two sites spell names differently (Martin Ødegaard / Martin Odegaard, Son Heung-Min / Heung-min Son, Kepa / Kepa Arrizabalaga), so `merge_data.py`:

1. Normalizes names (lowercase, accents and letters like `ø` stripped).
2. Fuzzy-scores every pair within a season with `rapidfuzz`.
3. Penalizes position disagreement. A goalkeeper can't match an outfielder, which stopped Liverpool's keeper Adrián matching a 19-year-old City midfielder with the same first name.
4. Assigns matches one-to-one, best score first.
5. Applies manual overrides for names too different to fuzzy-match (Max Kilman / Maximilian Kilman, Franck Zambo / Frank Anguissa).
6. Holds out players it can't tell apart: Understat lists both Emerson Royal and Emerson Palmieri as "Emerson", so those rows are excluded rather than risk a swap.

Every uncertain match is written to `match_review.csv`, and I checked them by hand. **Match rate among regulars (900+ minutes): 99.4–100% per season.**

### 3. Modeling choices

- **Test on an unseen season, not a random split.** A random split puts the same player in both train and test (as different seasons), which inflates the score.
- **Log-transformed target.** Values are heavily skewed, and the log makes effects proportional ("+30% value") rather than flat euro amounts.
- **Age curve.** `(age − 25)²` lets a linear model capture value rising and then falling with age.
- **Baseline comparison.** An R² means little on its own, so the model is scored against a position-median guess.

## Run it

```bash
pip install -r requirements.txt

python src/fetch_understat_stats.py          # stats, 5 seasons
python src/scrape_transfermarkt.py --test    # check one club first
python src/scrape_transfermarkt.py           # values, ~100 pages, ~5 min
python src/merge_data.py                     # match players across sources
python src/train_model.py                    # train, evaluate, plot

python src/predict.py --player "Saka"
python src/predict.py --goals 12 --assists 6 --minutes 2400 --age 23 --position F
```

Player lookups use a model refit on all five seasons, so they show how the model reads a player, not a test score. Out-of-sample predictions are in `outputs/predictions_2023.csv`.

## Project structure

```
src/
  fetch_understat_stats.py   stats from Understat's JSON endpoint
  scrape_transfermarkt.py    market values from club squad pages
  merge_data.py              cross-source name matching
  train_model.py             features, regression, evaluation, chart
  predict.py                 "what are you worth?" tool
data/processed/
  manual_overrides.csv       hand-checked name pairs
```

## Limitations and next steps

- **Add club strength.** The biggest available gain. The plan is to use the club's league points that season (available from Understat's team data). The club's squad market value would leak the target, so it's ruled out.
- **Account for market drift**, for example with a season trend term, so predictions for later seasons aren't biased low.
- **Market value isn't a transfer fee.** Transfermarkt values are crowd-sourced estimates. They're the right target here, since most players aren't transferred in a given season, but they're not prices anyone paid.
- **Stats are basic by design.** Goals and assists undersell defenders and keepers. Understat also provides xG and xA, which would be a natural extension.