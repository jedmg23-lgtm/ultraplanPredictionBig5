"""Poisson regression model for match goal rates.

Fits a goals-scored GLM in the classic Maher/Dixon-Coles form:

    log E[goals] = intercept + home_advantage + attack(team) - defence(opponent)

Each match contributes two observations (home side's goals, away side's
goals). Matches are weighted with an exponential time decay so recent form
counts more than results from two seasons ago.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class PoissonModel:
    result: object
    teams: list[str]
    train_matches: int
    league_avg_goals: float
    home_advantage: float  # multiplicative, e.g. 1.25 = +25% goals at home
    ratings: pd.DataFrame = field(repr=False)

    def predict_rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals (lambda_home, lambda_away) for a fixture."""
        for t in (home_team, away_team):
            if t not in self.teams:
                raise ValueError(f"Unknown team: {t}")
        fixture = pd.DataFrame(
            {
                "team": [home_team, away_team],
                "opponent": [away_team, home_team],
                "home": [1, 0],
            }
        )
        lam = self.result.predict(fixture)
        return float(lam.iloc[0]), float(lam.iloc[1])


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team": matches["HomeTeam"],
            "opponent": matches["AwayTeam"],
            "goals": matches["FTHG"],
            "home": 1,
            "Date": matches["Date"],
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["AwayTeam"],
            "opponent": matches["HomeTeam"],
            "goals": matches["FTAG"],
            "home": 0,
            "Date": matches["Date"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def time_decay_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay weight per observation; most recent match weighs 1."""
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / half_life_days)


PSEUDO_TEAM = "_league_average_"


def _pseudo_observations(teams: list[str], avg_goals: float, strength: float) -> pd.DataFrame:
    """Weak shrinkage toward league average.

    Teams with very little history (promoted sides early in a season) can have
    degenerate MLEs — e.g. attack -> -inf after one scoreless match. Give every
    team two pseudo-matches against a synthetic average opponent, worth
    ``strength`` match-weights each, so all estimates stay finite and
    small-sample teams start near league average.
    """
    rows = []
    for t in teams:
        rows.append({"team": t, "opponent": PSEUDO_TEAM, "goals": avg_goals, "home": 0})
        rows.append({"team": PSEUDO_TEAM, "opponent": t, "goals": avg_goals, "home": 0})
    df = pd.DataFrame(rows)
    df["weight"] = strength
    return df


def fit_poisson_model(
    matches: pd.DataFrame, half_life_days: float = 120.0, shrinkage_matches: float = 1.5
) -> PoissonModel:
    """Fit the Poisson GLM on a league's match history."""
    long_df = _long_format(matches)
    long_df["weight"] = time_decay_weights(long_df["Date"], half_life_days)

    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    avg_goals = float(np.average(long_df["goals"], weights=long_df["weight"]))
    pseudo = _pseudo_observations(teams, avg_goals, strength=shrinkage_matches)
    train = pd.concat([long_df.drop(columns=["Date"]), pseudo], ignore_index=True)

    result = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=train,
        family=sm.families.Poisson(),
        freq_weights=train["weight"].to_numpy(),
    ).fit()

    ratings = _team_ratings(result, teams)
    return PoissonModel(
        result=result,
        teams=teams,
        train_matches=len(matches),
        league_avg_goals=float(matches[["FTHG", "FTAG"]].to_numpy().mean()),
        home_advantage=float(np.exp(result.params["home"])),
        ratings=ratings,
    )


def _team_ratings(result, teams: list[str]) -> pd.DataFrame:
    """Attack/defence multipliers per team, relative to the (alphabetical) baseline team."""
    params = result.params
    rows = []
    for t in teams:
        atk = params.get(f"C(team)[T.{t}]", 0.0)
        dfc = params.get(f"C(opponent)[T.{t}]", 0.0)
        rows.append({"Team": t, "Attack": np.exp(atk), "Defence": np.exp(dfc)})
    df = pd.DataFrame(rows)
    # Normalise so 1.00 = league average, which reads better than
    # "relative to whichever team sorts first alphabetically".
    df["Attack"] /= df["Attack"].mean()
    df["Defence"] /= df["Defence"].mean()
    return df.sort_values("Attack", ascending=False).reset_index(drop=True)
