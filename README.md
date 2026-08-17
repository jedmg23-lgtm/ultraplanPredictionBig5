# Big 5 Soccer League Prediction Model

A Streamlit app that predicts match outcomes for Europe's Big 5 leagues
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1) using **Poisson
regression** and **Monte Carlo simulation**.

## How it works

1. **Data** — [`src/data.py`](src/data.py) downloads season result CSVs from
   [football-data.co.uk](https://www.football-data.co.uk/) (free, no API key)
   and caches them under `data/`. The in-progress season is refreshed on load.
2. **Model** — [`src/model.py`](src/model.py) fits a Poisson GLM
   (`statsmodels`) in the classic Maher/Dixon-Coles form:
   `log E[goals] = intercept + home_advantage + attack(team) − defence(opponent)`.
   Each match contributes two observations (each side's goals), weighted by an
   exponential time decay so recent form counts more. Every team also gets a
   few pseudo-matches against a synthetic league-average opponent, which keeps
   estimates finite for promoted teams with little history.
3. **Simulation** — [`src/simulate.py`](src/simulate.py) draws tens of
   thousands of Poisson scorelines from the fixture's expected-goals rates and
   tabulates win/draw/loss probabilities, fair odds, scoreline distribution,
   over/under lines, and both-teams-to-score.
4. **App** — [`app.py`](app.py) ties it together: pick a league and fixture,
   tune history depth / form half-life / simulation count, and view outcome
   probabilities, a scoreline heatmap, goals markets, and model team ratings.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/streamlit run app.py
```

The first load per league downloads a few hundred KB of CSVs and fits the
model (a couple of seconds); both are cached.

## Disclaimer

For education and entertainment. The model knows nothing about injuries,
lineups, or motivation — not betting advice.
