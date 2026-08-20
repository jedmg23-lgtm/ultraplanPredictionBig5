"""Walk-forward backtesting and calibration for the Poisson model.

The honest way to know whether a forecasting change helps: replay history,
predicting each match using only data available before it, and score the
predicted probabilities against what actually happened.

Metrics (lower is better for all):

- **RPS** (ranked probability score) — the standard score for ordered
  H/D/A football outcomes; punishes putting probability far from the
  actual result.
- **Brier** — mean squared error of the probability vector.
- **Log loss** — heavily punishes confident wrong predictions.

Benchmarks: the bookmaker's implied probabilities (Bet365 odds with the
margin removed) — the strongest freely available forecast — and the
uniform 1/3 forecast as a floor.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import fit_poisson_model


def outcome_probs(lambda_home: float, lambda_away: float, max_goals: int = 12):
    """Analytic P(home win), P(draw), P(away win) from independent Poissons."""
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lambda_home)
    pa = poisson.pmf(g, lambda_away)
    grid = np.outer(ph, pa)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """Ranked probability score per match; probs columns ordered [H, D, A]."""
    outcomes = np.zeros_like(probs)
    outcomes[np.arange(len(probs)), outcome_idx] = 1.0
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(outcomes, axis=1)
    return (cum_diff[:, :-1] ** 2).sum(axis=1) / (probs.shape[1] - 1)


@dataclass
class BacktestResult:
    frame: pd.DataFrame  # one row per evaluated match
    metrics: pd.DataFrame  # forecaster x (RPS, Brier, log loss)
    n_matches: int
    n_refits: int

    def calibration(self, bins: int = 10) -> pd.DataFrame:
        """Pooled reliability curve: predicted probability vs observed frequency."""
        probs = self.frame[["p_home", "p_draw", "p_away"]].to_numpy().ravel()
        hits = np.zeros((len(self.frame), 3))
        hits[np.arange(len(self.frame)), self.frame["outcome_idx"]] = 1.0
        hits = hits.ravel()
        cut = pd.cut(probs, np.linspace(0, 1, bins + 1), include_lowest=True)
        df = pd.DataFrame({"bin": cut, "prob": probs, "hit": hits})
        out = df.groupby("bin", observed=True).agg(
            predicted=("prob", "mean"), observed=("hit", "mean"), n=("hit", "size")
        )
        return out.reset_index(drop=True)


def _score_block(probs: np.ndarray, outcome_idx: np.ndarray) -> dict:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcome_idx] = 1.0
    eps = 1e-12
    return {
        "RPS": float(rps(probs, outcome_idx).mean()),
        "Brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "Log loss": float(
            -np.log(np.clip(probs[np.arange(len(probs)), outcome_idx], eps, None)).mean()
        ),
    }


def backtest(
    matches: pd.DataFrame,
    top_code: str,
    half_life_days: float = 300.0,
    shrinkage_matches: float = 3.0,
    refit_days: int = 28,
    min_train_days: int = 365,
) -> BacktestResult:
    """Replay history: refit every ``refit_days``, predict the top-division
    matches in the following window, score against actual outcomes.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    start = matches["Date"].min() + pd.Timedelta(days=min_train_days)
    end = matches["Date"].max()

    rows = []
    n_refits = 0
    cutoff = start
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while cutoff <= end:
            window_end = cutoff + pd.Timedelta(days=refit_days)
            eval_mask = (
                (matches["Div"] == top_code)
                & (matches["Date"] >= cutoff)
                & (matches["Date"] < window_end)
            )
            eval_matches = matches[eval_mask]
            if not eval_matches.empty:
                train = matches[matches["Date"] < cutoff]
                model = fit_poisson_model(
                    train, half_life_days=half_life_days, shrinkage_matches=shrinkage_matches
                )
                n_refits += 1
                known = set(model.teams)
                for _, mrow in eval_matches.iterrows():
                    if mrow["HomeTeam"] not in known or mrow["AwayTeam"] not in known:
                        continue
                    lh, la = model.predict_rates(mrow["HomeTeam"], mrow["AwayTeam"])
                    p_h, p_d, p_a = outcome_probs(lh, la)
                    outcome_idx = 0 if mrow["FTHG"] > mrow["FTAG"] else (1 if mrow["FTHG"] == mrow["FTAG"] else 2)
                    row = {
                        "Date": mrow["Date"],
                        "HomeTeam": mrow["HomeTeam"],
                        "AwayTeam": mrow["AwayTeam"],
                        "outcome_idx": outcome_idx,
                        "p_home": p_h,
                        "p_draw": p_d,
                        "p_away": p_a,
                    }
                    if all(c in mrow and pd.notna(mrow[c]) for c in ("B365H", "B365D", "B365A")):
                        inv = np.array([1 / mrow["B365H"], 1 / mrow["B365D"], 1 / mrow["B365A"]])
                        market = inv / inv.sum()  # strip the bookmaker margin
                        row.update(m_home=market[0], m_draw=market[1], m_away=market[2])
                    rows.append(row)
            cutoff = window_end

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Backtest produced no evaluated matches — not enough history.")

    outcome_idx = frame["outcome_idx"].to_numpy()
    model_probs = frame[["p_home", "p_draw", "p_away"]].to_numpy()
    metrics = {"Poisson model": _score_block(model_probs, outcome_idx)}

    has_market = frame[["m_home", "m_draw", "m_away"]].notna().all(axis=1) if "m_home" in frame else pd.Series(False, index=frame.index)
    if has_market.any():
        mkt = frame.loc[has_market, ["m_home", "m_draw", "m_away"]].to_numpy()
        metrics["Bookmaker (Bet365)"] = _score_block(mkt, outcome_idx[has_market.to_numpy()])
    metrics["Uniform 1/3"] = _score_block(
        np.full_like(model_probs, 1 / 3), outcome_idx
    )

    return BacktestResult(
        frame=frame,
        metrics=pd.DataFrame(metrics).T,
        n_matches=len(frame),
        n_refits=n_refits,
    )
