"""Data sourcing for Big 5 league match results.

Downloads season CSVs from football-data.co.uk (free, no API key) and
caches them locally under ``data/``. Each CSV holds one league-season of
final scores plus stats/odds columns; we keep only what the model needs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "Premier League (England)": "E0",
    "La Liga (Spain)": "SP1",
    "Bundesliga (Germany)": "D1",
    "Serie A (Italy)": "I1",
    "Ligue 1 (France)": "F1",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


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


def _download_csv(league_code: str, season: str, dest: Path) -> bool:
    url = BASE_URL.format(season=season, code=league_code)
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


def _read_csv(path: Path, league_code: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return None
    if not set(REQUIRED_COLS + ["Div"]).issubset(df.columns):
        return None
    # Early-season files are sometimes placeholders holding a different
    # division's fixtures (e.g. E0.csv serving National League rows).
    df = df[df["Div"] == league_code]
    df = df[REQUIRED_COLS].dropna()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def load_matches(league_code: str, n_seasons: int = 3, refresh_current: bool = False) -> pd.DataFrame:
    """Load the last ``n_seasons`` of results for one league, downloading as needed.

    Completed seasons are cached forever; the in-progress season is
    re-downloaded when ``refresh_current`` is set (or missing locally).
    Seasons that are unavailable (e.g. not started yet) are skipped.
    """
    frames = []
    latest = current_season_start_year()
    for start_year in recent_seasons(n_seasons):
        season = season_code(start_year)
        path = DATA_DIR / f"{league_code}_{season}.csv"
        is_current = start_year == latest
        if not path.exists() or (is_current and refresh_current):
            if not _download_csv(league_code, season, path) and not path.exists():
                continue
        df = _read_csv(path, league_code)
        if df is None or df.empty:
            continue
        df["Season"] = f"{start_year}/{(start_year + 1) % 100:02d}"
        frames.append(df)
    if not frames:
        raise RuntimeError(
            f"No match data available for league {league_code}. "
            "Check your internet connection (data comes from football-data.co.uk)."
        )
    out = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out
