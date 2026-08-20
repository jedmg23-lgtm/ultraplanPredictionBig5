"""Hyperparameter tuning via walk-forward backtest.

Grid-searches form half-life and shrinkage strength, scoring each combo
by mean RPS on replayed history. Run occasionally (e.g. once a month or
after big data updates) and put the winning values into the app defaults.

Usage:
    python -m scripts.tune                      # Premier League, default grid
    python -m scripts.tune --league "La Liga (Spain)"
    python -m scripts.tune --all                # every league (slow)
"""

from __future__ import annotations

import argparse
import itertools

from src.data import LEAGUES, load_matches
from src.evaluate import backtest

HALF_LIVES = [120.0, 200.0, 300.0, 450.0, 650.0]
SHRINKAGES = [1.5, 3.0, 6.0]


def tune_league(league_name: str, n_seasons: int = 3) -> None:
    top_code, second_code = LEAGUES[league_name]
    matches = load_matches((top_code, second_code), n_seasons=n_seasons)
    print(f"\n=== {league_name}: {len(matches)} matches ===")
    print(f"{'half-life':>10} {'shrinkage':>10} {'RPS':>8} {'log loss':>9} {'n':>5}")
    best = None
    for hl, sh in itertools.product(HALF_LIVES, SHRINKAGES):
        result = backtest(matches, top_code, half_life_days=hl, shrinkage_matches=sh)
        score = result.metrics.loc["Poisson model", "RPS"]
        ll = result.metrics.loc["Poisson model", "Log loss"]
        print(f"{hl:>10.0f} {sh:>10.1f} {score:>8.4f} {ll:>9.4f} {result.n_matches:>5}")
        if best is None or score < best[0]:
            best = (score, hl, sh)
    print(f"--> best: half-life {best[1]:.0f} days, shrinkage {best[2]:.1f} (RPS {best[0]:.4f})")
    if "Bookmaker (Bet365)" in result.metrics.index:
        print(f"    bookmaker benchmark RPS: {result.metrics.loc['Bookmaker (Bet365)', 'RPS']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="Premier League (England)", choices=list(LEAGUES))
    parser.add_argument("--all", action="store_true", help="tune every league")
    parser.add_argument("--seasons", type=int, default=3)
    args = parser.parse_args()

    for name in LEAGUES if args.all else [args.league]:
        tune_league(name, n_seasons=args.seasons)


if __name__ == "__main__":
    main()
