"""Data sourcing for Big 5 league match results.

Downloads season CSVs from football-data.co.uk (free, no API key) and
caches them locally under ``data/``. Each CSV holds one league-season of
final scores plus stats/odds columns; we keep only what the model needs.

Each country contributes its top TWO divisions. Training on both gives
newly promoted teams (e.g. a side entering the Premier League from the
Championship) a rating earned in their old division, linked to the top
flight through teams that moved between the two.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# league name -> (top-division code, second-division code)
LEAGUES = {
    "Premier League (England)": ("E0", "E1"),
    "La Liga (Spain)": ("SP1", "SP2"),
    "Bundesliga (Germany)": ("D1", "D2"),
    "Serie A (Italy)": ("I1", "I2"),
    "Ligue 1 (France)": ("F1", "F2"),
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ODDS_COLS = ["B365H", "B365D", "B365A"]  # kept when present; backtest benchmark


def current_season_start_year(today: dt.date | None = None) -> int:
    """Return the start year of the season in progress (seasons roll over in July)."""
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    """2024 -> '2425' as used in football-data.co.uk URLs."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def recent_seasons(n: int, today: dt.date | None = None) -> list[int]:
    """Start years of the ``n`` most recent seasons, oldest first, including the one in progress."""
    latest = current_season_start_year(today)
    return list(range(latest - n + 1, latest + 1))


def _download_csv(div_code: str, season: str, dest: Path) -> bool:
    url = BASE_URL.format(season=season, code=div_code)
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException:
        return False
    # football-data returns 200 with an error page or an empty file for
    # missing seasons; require a plausible CSV header.
    if resp.status_code != 200 or b"HomeTeam" not in resp.content[:2048]:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def _read_csv(path: Path, div_code: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return None
    if not set(REQUIRED_COLS + ["Div"]).issubset(df.columns):
        return None
    # Early-season files are sometimes placeholders holding a different
    # division's fixtures (e.g. E0.csv serving National League rows).
    df = df[df["Div"] == div_code]
    keep = REQUIRED_COLS + ["Div"] + [c for c in ODDS_COLS if c in df.columns]
    df = df[keep].dropna(subset=REQUIRED_COLS)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def load_division(div_code: str, n_seasons: int = 3, refresh_current: bool = False) -> pd.DataFrame:
    """Load the last ``n_seasons`` of results for one division, downloading as needed.

    Completed seasons are cached forever; the in-progress season is
    re-downloaded when ``refresh_current`` is set (or missing locally).
    Seasons that are unavailable (e.g. not started yet) are skipped.
    """
    frames = []
    latest = current_season_start_year()
    for start_year in recent_seasons(n_seasons):
        season = season_code(start_year)
        path = DATA_DIR / f"{div_code}_{season}.csv"
        is_current = start_year == latest
        if not path.exists() or (is_current and refresh_current):
            if not _download_csv(div_code, season, path) and not path.exists():
                continue
        df = _read_csv(path, div_code)
        if df is None or df.empty:
            continue
        df["Season"] = f"{start_year}/{(start_year + 1) % 100:02d}"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLS + ["Div", "Season"])
    return pd.concat(frames, ignore_index=True)


def load_matches(
    league: tuple[str, str] | str, n_seasons: int = 3, refresh_current: bool = False
) -> pd.DataFrame:
    """Load pooled top + second division history for one country.

    ``league`` is a ``(top_code, second_code)`` pair from :data:`LEAGUES`
    (a bare top-division string also works and loads that division only).
    """
    codes = (league,) if isinstance(league, str) else tuple(league)
    frames = [load_division(c, n_seasons=n_seasons, refresh_current=refresh_current) for c in codes]
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise RuntimeError(
            f"No match data available for {codes}. "
            "Check your internet connection (data comes from football-data.co.uk)."
        )
    return out.sort_values("Date").reset_index(drop=True)


def _season_points(matches: pd.DataFrame) -> pd.Series:
    """Points per team from a set of results."""
    home_pts = matches["FTHG"].gt(matches["FTAG"]) * 3 + matches["FTHG"].eq(matches["FTAG"]) * 1
    away_pts = matches["FTAG"].gt(matches["FTHG"]) * 3 + matches["FTHG"].eq(matches["FTAG"]) * 1
    pts = pd.concat(
        [
            pd.DataFrame({"team": matches["HomeTeam"], "pts": home_pts}),
            pd.DataFrame({"team": matches["AwayTeam"], "pts": away_pts}),
        ]
    )
    return pts.groupby("team")["pts"].sum().sort_values(ascending=False)


def top_flight_teams(matches: pd.DataFrame, top_code: str) -> list[str]:
    """Best guess at the current top-division roster.

    Start from the most recent top-division season in the data, drop teams
    that have since shown up in the second division (relegated), and add
    likely-promoted teams: sides that left the second division after
    finishing in its top half. Early in a season, before the new files
    exist, this degrades gracefully to last season's roster.
    """
    top = matches[matches["Div"] == top_code]
    if top.empty:
        return []
    last_top_season = top["Season"].max()
    roster = set(top.loc[top["Season"] == last_top_season, "HomeTeam"]) | set(
        top.loc[top["Season"] == last_top_season, "AwayTeam"]
    )

    second = matches[matches["Div"] != top_code]
    newer_second = second[second["Season"] > last_top_season]
    if not newer_second.empty:
        # Relegated: were in last season's top flight, now playing second division.
        roster -= set(newer_second["HomeTeam"]) | set(newer_second["AwayTeam"])
        # Promoted: finished top-half of last season's second division and left it.
        prev = second[second["Season"] == last_top_season]
        if not prev.empty:
            table = _season_points(prev)
            still_there = set(newer_second["HomeTeam"]) | set(newer_second["AwayTeam"])
            leavers = [t for t in table.index if t not in still_there]
            top_half = set(table.index[: len(table) // 2])
            roster |= {t for t in leavers if t in top_half}
    return sorted(roster)
