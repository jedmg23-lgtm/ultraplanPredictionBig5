"""Monte Carlo simulation of a fixture from Poisson goal rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_GOALS_GRID = 8  # scoreline heatmap covers 0..8 goals per side


@dataclass
class SimulationResult:
    n_sims: int
    lambda_home: float
    lambda_away: float
    p_home: float
    p_draw: float
    p_away: float
    avg_home_goals: float
    avg_away_goals: float
    p_over: dict  # line -> P(total goals > line), e.g. {1.5: .., 2.5: .., 3.5: ..}
    p_btts: float
    score_matrix: pd.DataFrame  # P(home=i, away=j) grid from the simulations
    top_scores: pd.DataFrame  # most frequent scorelines

    @property
    def fair_odds(self) -> dict:
        """Fair (no-margin) decimal odds implied by the simulated probabilities."""
        return {
            k: (1.0 / p if p > 0 else float("inf"))
            for k, p in [("home", self.p_home), ("draw", self.p_draw), ("away", self.p_away)]
        }


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    n_sims: int = 20_000,
    seed: int | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(lambda_home, n_sims)
    away_goals = rng.poisson(lambda_away, n_sims)

    diff = home_goals - away_goals
    total = home_goals + away_goals

    grid = np.zeros((MAX_GOALS_GRID + 1, MAX_GOALS_GRID + 1))
    h_cap = np.minimum(home_goals, MAX_GOALS_GRID)
    a_cap = np.minimum(away_goals, MAX_GOALS_GRID)
    np.add.at(grid, (h_cap, a_cap), 1)
    grid /= n_sims
    score_matrix = pd.DataFrame(
        grid,
        index=[str(i) for i in range(MAX_GOALS_GRID + 1)],
        columns=[str(j) for j in range(MAX_GOALS_GRID + 1)],
    )

    scores, counts = np.unique(np.stack([home_goals, away_goals], axis=1), axis=0, return_counts=True)
    order = np.argsort(-counts)[:8]
    top_scores = pd.DataFrame(
        {
            "Scoreline": [f"{h}–{a}" for h, a in scores[order]],
            "Probability": counts[order] / n_sims,
        }
    )

    return SimulationResult(
        n_sims=n_sims,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        p_home=float((diff > 0).mean()),
        p_draw=float((diff == 0).mean()),
        p_away=float((diff < 0).mean()),
        avg_home_goals=float(home_goals.mean()),
        avg_away_goals=float(away_goals.mean()),
        p_over={line: float((total > line).mean()) for line in (1.5, 2.5, 3.5)},
        p_btts=float(((home_goals > 0) & (away_goals > 0)).mean()),
        score_matrix=score_matrix,
        top_scores=top_scores,
    )
