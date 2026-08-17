"""Big 5 Leagues — soccer match outcome predictor.

Streamlit app: Poisson regression on football-data.co.uk results,
Monte Carlo simulation of the selected fixture.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import LEAGUES, load_matches
from src.model import fit_poisson_model
from src.simulate import simulate_match

# Validated chart palette (dataviz reference instance, light mode)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
HOME_COLOR = "#2a78d6"  # categorical slot 1 (blue)
AWAY_COLOR = "#eb6834"  # categorical slot 2 (orange)
DRAW_COLOR = "#898781"  # neutral
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Big 5 Match Predictor", page_icon="⚽", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading match data…")
def get_matches(league_code: str, n_seasons: int) -> pd.DataFrame:
    return load_matches(league_code, n_seasons=n_seasons, refresh_current=True)


@st.cache_resource(show_spinner="Fitting Poisson regression…")
def get_model(league_code: str, n_seasons: int, half_life_days: float, data_version: str):
    # data_version keys the cache so the model refits when new results arrive
    matches = get_matches(league_code, n_seasons)
    return fit_poisson_model(matches, half_life_days=half_life_days)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def outcome_bar(p_home: float, p_draw: float, p_away: float, home: str, away: str) -> go.Figure:
    fig = go.Figure()
    labels = [f"{home} win", "Draw", f"{away} win"]
    values = [p_home, p_draw, p_away]
    colors = [HOME_COLOR, DRAW_COLOR, AWAY_COLOR]
    fig.add_bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[pct(v) for v in values],
        textposition="outside",
        textfont=dict(color=INK_2, size=13),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        width=0.55,
    )
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, max(values) * 1.25], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color=INK, size=13)),
        showlegend=False,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    )
    return fig


def score_heatmap(score_matrix: pd.DataFrame, home: str, away: str) -> go.Figure:
    cap = 5  # display 0..5+; probability mass beyond is tiny
    m = score_matrix.to_numpy()[: cap + 1, : cap + 1].copy()
    m[cap, :] += score_matrix.to_numpy()[cap + 1 :, : cap + 1].sum(axis=0)
    m[:, cap] += score_matrix.to_numpy()[: cap + 1, cap + 1 :].sum(axis=1)
    m[cap, cap] += score_matrix.to_numpy()[cap + 1 :, cap + 1 :].sum()
    labels = [str(i) for i in range(cap)] + [f"{cap}+"]

    fig = go.Figure(
        go.Heatmap(
            z=m,
            x=labels,
            y=labels,
            colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
            zmin=0,
            hovertemplate=(
                f"{home} %{{y}} – %{{x}} {away}<br>Probability: %{{z:.1%}}<extra></extra>"
            ),
            colorbar=dict(
                tickformat=".0%", thickness=12, outlinewidth=0, tickfont=dict(color=MUTED)
            ),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        xaxis=dict(
            title=f"{away} goals",
            type="category",
            tickfont=dict(color=INK_2),
            title_font=dict(color=INK_2),
        ),
        yaxis=dict(
            title=f"{home} goals",
            type="category",
            tickfont=dict(color=INK_2),
            title_font=dict(color=INK_2),
        ),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Big 5 Match Predictor")
    league_name = st.selectbox("League", list(LEAGUES))
    n_seasons = st.slider(
        "Seasons of history", 1, 5, 3, help="How many recent seasons to train on."
    )
    half_life = st.slider(
        "Form half-life (days)",
        60,
        720,
        300,
        step=30,
        help="A match this many days old counts half as much as one played today.",
    )
    n_sims = st.select_slider(
        "Monte Carlo simulations", options=[5_000, 10_000, 20_000, 50_000, 100_000], value=20_000
    )
    st.caption(
        "Data: [football-data.co.uk](https://www.football-data.co.uk/). "
        "Model: Poisson regression (attack/defence + home advantage) "
        "with exponential time decay."
    )

league_code = LEAGUES[league_name]

try:
    matches = get_matches(league_code, n_seasons)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

data_version = f"{matches['Date'].max():%Y-%m-%d}-{len(matches)}"
model = get_model(league_code, n_seasons, float(half_life), data_version)

# Current-season teams first in the pickers; older relegated teams after.
latest_season = matches["Season"].max()
current = sorted(
    set(matches.loc[matches["Season"] == latest_season, "HomeTeam"])
    | set(matches.loc[matches["Season"] == latest_season, "AwayTeam"])
)
older = [t for t in model.teams if t not in current]
team_options = current + older

# ── Fixture selection ────────────────────────────────────────────────────────
st.subheader(league_name)
c1, c2, c3 = st.columns([3, 3, 2])
home_team = c1.selectbox("Home team", team_options, index=0)
away_default = 1 if len(team_options) > 1 else 0
away_team = c2.selectbox("Away team", team_options, index=away_default)
c3.markdown("<br>", unsafe_allow_html=True)
run = c3.button("Simulate match", type="primary", use_container_width=True)

if home_team == away_team:
    st.warning("Pick two different teams.")
    st.stop()

if run:
    st.session_state["fixture"] = (home_team, away_team, n_sims)

# Drop a stored fixture that doesn't belong to the currently selected league.
if "fixture" in st.session_state and not set(st.session_state["fixture"][:2]) <= set(model.teams):
    del st.session_state["fixture"]

if "fixture" not in st.session_state:
    st.info(
        f"Trained on **{model.train_matches:,} matches** "
        f"({matches['Date'].min():%b %Y} – {matches['Date'].max():%b %Y}). "
        f"League average: **{model.league_avg_goals:.2f}** goals per team per match; "
        f"home advantage: **{(model.home_advantage - 1) * 100:+.0f}%** goals. "
        "Pick a fixture and hit **Simulate match**."
    )
else:
    home_team, away_team, n_sims_used = st.session_state["fixture"]
    lam_h, lam_a = model.predict_rates(home_team, away_team)
    sim = simulate_match(lam_h, lam_a, n_sims=n_sims_used, seed=42)

    st.markdown(f"### {home_team} vs {away_team}")
    st.caption(
        f"{n_sims_used:,} simulated matches · expected goals "
        f"{home_team} **{lam_h:.2f}** – **{lam_a:.2f}** {away_team}"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home_team} win", pct(sim.p_home), f"fair odds {sim.fair_odds['home']:.2f}", delta_color="off")
    m2.metric("Draw", pct(sim.p_draw), f"fair odds {sim.fair_odds['draw']:.2f}", delta_color="off")
    m3.metric(f"{away_team} win", pct(sim.p_away), f"fair odds {sim.fair_odds['away']:.2f}", delta_color="off")

    st.plotly_chart(
        outcome_bar(sim.p_home, sim.p_draw, sim.p_away, home_team, away_team),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    left, right = st.columns([5, 4])
    with left:
        st.markdown("#### Scoreline probabilities")
        st.plotly_chart(
            score_heatmap(sim.score_matrix, home_team, away_team),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.markdown("#### Most likely scorelines")
        top = sim.top_scores.copy()
        top["Probability"] = top["Probability"].map(pct)
        st.dataframe(top, hide_index=True, use_container_width=True)

        st.markdown("#### Goals markets")
        markets = pd.DataFrame(
            {
                "Market": [f"Over {line}" for line in sim.p_over]
                + [f"Under {line}" for line in sim.p_over]
                + ["Both teams to score"],
                "Probability": [pct(p) for p in sim.p_over.values()]
                + [pct(1 - p) for p in sim.p_over.values()]
                + [pct(sim.p_btts)],
            }
        )
        st.dataframe(markets, hide_index=True, use_container_width=True)

    with st.expander("Team ratings (model attack / defence strength)"):
        st.caption(
            "1.00 = league average. Attack > 1 scores more than average; "
            "Defence > 1 concedes more than average (lower is better)."
        )
        ratings = model.ratings.copy()
        ratings["Attack"] = ratings["Attack"].round(2)
        ratings["Defence"] = ratings["Defence"].round(2)
        st.dataframe(ratings, hide_index=True, use_container_width=True, height=420)

    st.caption(
        "For education and entertainment — a Poisson model knows nothing about "
        "injuries, lineups, or motivation. Not betting advice."
    )
