"""
FORENSICS XG: INTELLIGENCE UNIT
Football Analytics — Interactive data analysis tool.
Ask questions and get charts, stats, and insights from Europe's top 7 leagues.
Data source: Opta 2025-2026 season.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
import re
import unicodedata
import urllib.request
import urllib.parse
import requests
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FORENSICS XG: INTELLIGENCE UNIT",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Opta data directory — check repo root first (Streamlit Cloud), then local path
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIR = os.path.join(_REPO_DIR, "..", "..", "Forensics xG Opta Data")
OPTA_DIR = _REPO_DIR if os.path.isdir(os.path.join(_REPO_DIR, "Bundesliga")) else _LOCAL_DIR

LEAGUE_FOLDERS = {
    "Premier League": "English Premier League",
    "LaLiga": "LaLiga",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Serie A": "Serie A",
    "Primeira Liga": "Primeira Liga",
    "Eredivisie": "Eredivisie",
}

# Squad roster overrides are no longer needed — Opta data includes team info directly.
SQUAD_ROSTER_OVERRIDES = {}

CHART_COLORS = px.colors.qualitative.Vivid

# ── Player Financials CSV ────────────────────────────────────────────────────
_FINANCIALS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Player Financials", "player_financials.csv")

# ── Player Photos CSV ────────────────────────────────────────────────────────
_PHOTOS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_photos.csv")

# ── Player Footedness CSV (from Transfermarkt, keyed by Opta id) ──────────────
_FOOT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_footedness.csv")

# ── GK PSxG CSV (true shot-level post-shot xG, keyed by Opta id + season) ─────
# Pre-computed locally by build_gk_psxg.py from the raw Opta event JSONs (which
# are too large to commit).  Replaces the season-aggregate PSxG approximation in
# _compute_gk_derived for any keeper-season present here.
_GK_PSXG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gk_psxg.csv")

# ── Player xG CSV (pre-shot expected goals per shooter, keyed by id + season) ─
# Pre-computed locally by build_player_xg.py from the same event JSONs.  Powers
# npxG and finishing (npG − xG) for outfield players.
_PLAYER_XG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_xg.csv")

# ── Team xT CSVs (open-play expected threat, keyed by liga + temporada + equipo)
# Built by build_team_xt.py from the same raw Opta event JSONs.  team_xt.csv is
# one row per team-season; team_xt_grid.csv breaks that total down by the pitch
# zone each action started from, which powers the Team Profile heatmap.
_TEAM_XT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt.csv")
_TEAM_XT_GRID_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt_grid.csv")

# ── xT the other way (build_spatial.py): threat PREVENTED by defensive actions,
# and per-player xT generated + prevented.  Same builder→CSV→app pattern.
_TEAM_XTP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt_prevented.csv")
_TEAM_XTP_GRID_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt_prevented_grid.csv")
_PLAYER_XT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_xt.csv")
# Per-player xT by zone (current season) — powers the profile's xT gen/prevented maps.
_PLAYER_XT_GRID_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_xt_grid.csv")

# ── Heat maps & pass sonars (build_maps.py, current season only) ─────────────
_TEAM_HEATMAP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_heatmap.csv")
_PLAYER_HEATMAP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_heatmap.csv")
_TEAM_SONAR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_sonar.csv")
_PLAYER_SONAR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_sonar.csv")

# ── xT CONCEDED (build_xt_conceded.py) — the threat opponents create against
# this team, charged to the defending side and mirrored into its own attacking
# orientation, so the map reads with its own goal on the left like every other
# defensive map here.  Lower is better, unambiguously (unlike xT prevented,
# which partly measures defensive workload).
_TEAM_XTC_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt_conceded.csv")
_TEAM_XTC_GRID_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_xt_conceded_grid.csv")

# ── Pivot Index (build_pivot_index.py) — deep-lying playmaker scoring, one row
# per midfielder-season.  CONTROL / PROGRESSION / ANCHOR percentiles plus the
# combined PIVOT; powers the Team Profile's midfield-archetype quadrant.
_PLAYER_PIVOT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_pivot.csv")


def _csv_mtime(path):
    """Return CSV file modification time as int (for cache-busting)."""
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


@st.cache_data(ttl=3600, show_spinner=False)
def _load_photos_csv(_bust=0):
    """Load manual photo URL overrides from player_photos.csv (if it exists).
    CSV format: short_name, photo_url
    short_name must match the Opta 'nombre' field exactly (e.g. 'E. Haaland')."""
    if not os.path.exists(_PHOTOS_CSV):
        return {}
    df = pd.read_csv(_PHOTOS_CSV, encoding="utf-8-sig")
    lookup = {}
    for _, r in df.iterrows():
        key = str(r.get("short_name", "")).strip()
        url = str(r.get("photo_url", "")).strip()
        if key and url and url.lower() not in ("", "nan", "none"):
            lookup[key] = url
    return lookup


@st.cache_data(ttl=3600, show_spinner=False)
def _load_financials_csv(_bust=0):
    """Load pre-fetched market values and salaries from CSV (if it exists.)."""
    if not os.path.exists(_FINANCIALS_CSV):
        return {}
    df = pd.read_csv(_FINANCIALS_CSV, encoding="utf-8-sig")
    lookup = {}
    for _, r in df.iterrows():
        key = str(r.get("short_name", "")).strip()
        if key:
            mv = r.get("market_value", "")
            sal = r.get("salary", "")
            age = r.get("age", "") if "age" in df.columns else ""
            team = r.get("team", "") if "team" in df.columns else ""
            lookup[key] = {
                "market_value": mv if pd.notna(mv) and mv != "" else None,
                "salary": sal if pd.notna(sal) and sal != "" else None,
                "age": age if pd.notna(age) and str(age).strip() != "" else None,
                "team": team if pd.notna(team) and str(team).strip() != "" else None,
            }
    return lookup


def _fin_rec(financials, name, team=None):
    """Financials row for a player, guarding against shared-name collisions.

    The CSV is keyed by short name, so a famous player's value can sit under a
    name an obscure player also uses.  When the stored row carries a team and it
    doesn't match this player's team, treat it as a miss (no value) rather than
    showing the wrong one."""
    rec = financials.get(name)
    if not rec:
        return {}
    rec_team = rec.get("team")
    if team and rec_team and _canon_team(rec_team) != _canon_team(team):
        return {}
    return rec


def _mv_millions(s):
    """Parse a Transfermarkt value string ('€75.00m', '€500k', '€1.20bn') to a
    float in € millions, or NaN."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    t = str(s).strip().lower().replace("€", "").replace("€", "").replace(",", "").strip()
    if not t or t in ("-", "nan", "none"):
        return np.nan
    mult = 1.0
    if t.endswith("bn") or t.endswith("b"):
        mult, t = 1000.0, t.rstrip("bn")
    elif t.endswith("m"):
        mult, t = 1.0, t[:-1]
    elif t.endswith("k"):
        mult, t = 0.001, t[:-1]
    try:
        return round(float(t) * mult, 3)
    except ValueError:
        return np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def _load_footedness_csv(_bust=0):
    """Preferred foot (left/right/both) from Transfermarkt, keyed by Opta id."""
    if not os.path.exists(_FOOT_CSV):
        return {}
    df = pd.read_csv(_FOOT_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        pid = (r.get("id") or "").strip()
        foot = (r.get("foot") or "").strip().lower()
        if pid and foot in ("left", "right", "both"):
            out[pid] = foot
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_gk_psxg_csv(_bust=0):
    """True post-shot xG per goalkeeper-season, keyed by (Opta id, temporada).

    Built by build_gk_psxg.py from the raw event JSONs.  Returns
    {(id, temporada): {"PSxG", "PSxG+/-", "shots", "goals"}}.
    """
    if not os.path.exists(_GK_PSXG_CSV):
        return {}
    df = pd.read_csv(_GK_PSXG_CSV, encoding="utf-8-sig", low_memory=False)
    out = {}
    for _, r in df.iterrows():
        pid = str(r.get("id", "")).strip()
        season = str(r.get("temporada", "")).strip()
        if not pid or not season:
            continue
        out[(pid, season)] = {
            "PSxG": float(r.get("PSxG", 0) or 0),
            "PSxG+/-": float(r.get("PSxG+/-", 0) or 0),
            "shots": float(r.get("shots_on_target_faced", 0) or 0),
            "goals": float(r.get("psxg_goals_faced", 0) or 0),
            "soft": float(r.get("soft_goals_conceded", 0) or 0),
        }
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_xg_csv(_bust=0):
    """Pre-shot xG per shooter-season, keyed by (Opta id, temporada).

    Built by build_player_xg.py.  Returns
    {(id, temporada): {"xG", "npxG", "npG-xG", "shots", "np_shots"}}.
    """
    if not os.path.exists(_PLAYER_XG_CSV):
        return {}
    df = pd.read_csv(_PLAYER_XG_CSV, encoding="utf-8-sig", low_memory=False)
    out = {}
    for _, r in df.iterrows():
        pid = str(r.get("id", "")).strip()
        season = str(r.get("temporada", "")).strip()
        if not pid or not season:
            continue
        out[(pid, season)] = {
            "xG": float(r.get("xG", 0) or 0),
            "npxG": float(r.get("npxG", 0) or 0),
            "npG-xG": float(r.get("npG-xG", 0) or 0),
            "shots": float(r.get("shots", 0) or 0),
            "np_shots": float(r.get("np_shots", 0) or 0),
            "xA": float(r.get("xA", 0) or 0),
            "key_passes": float(r.get("key_passes", 0) or 0),
        }
    return out


def _read_team_xt(path):
    """Shared reader for the two team-xT CSVs; empty frame if not built yet."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()
    for col in ("liga", "temporada", "equipo"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xt_csv(_bust=0):
    """Open-play xT per team-season: liga, temporada, equipo, matches,
    xt_total, xt_per_match, moves.  Built by build_team_xt.py."""
    return _read_team_xt(_TEAM_XT_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xt_grid_csv(_bust=0):
    """Per-match xT by originating pitch zone (zx 0-11 own goal -> opponent
    goal, zy 0-7 across).  Long form, one row per team-season-zone."""
    return _read_team_xt(_TEAM_XT_GRID_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xtp_csv(_bust=0):
    """xT prevented per team-season: liga, temporada, equipo, matches,
    xtp_total, xtp_per_match.  Built by build_spatial.py."""
    return _read_team_xt(_TEAM_XTP_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xtp_grid_csv(_bust=0):
    """xT prevented per match by the zone the defensive action happened in
    (team's own orientation, so its defensive third is on the left)."""
    return _read_team_xt(_TEAM_XTP_GRID_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xtc_csv(_bust=0):
    """xT conceded per team-season: liga, temporada, equipo, matches,
    xtc_total, xtc_per_match.  Built by build_xt_conceded.py."""
    return _read_team_xt(_TEAM_XTC_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_xtc_grid_csv(_bust=0):
    """xT conceded per match by the zone the opponent's action started in,
    mirrored into this team's own orientation (own goal on the left)."""
    return _read_team_xt(_TEAM_XTC_GRID_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_xt_csv(_bust=0):
    """Per-player xT, keyed by (Opta id, temporada):
    {(id, temporada): {"gen", "prevented", "moves", "def"}}."""
    if not os.path.exists(_PLAYER_XT_CSV):
        return {}
    df = pd.read_csv(_PLAYER_XT_CSV, encoding="utf-8-sig", low_memory=False)
    out = {}
    for _, r in df.iterrows():
        pid = str(r.get("id", "")).strip()
        season = str(r.get("temporada", "")).strip()
        if not pid or not season:
            continue
        out[(pid, season)] = {
            "gen": float(r.get("xt_gen", 0) or 0),
            "prevented": float(r.get("xt_prevented", 0) or 0),
            "moves": float(r.get("moves", 0) or 0),
            "def": float(r.get("def_actions", 0) or 0),
        }
    return out


def _read_id_csv(path):
    """Reader for the id-keyed heat/sonar CSVs; empty frame if not built."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()
    for col in ("id", "temporada"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_heatmap_csv(_bust=0):
    return _read_team_xt(_TEAM_HEATMAP_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_heatmap_csv(_bust=0):
    return _read_id_csv(_PLAYER_HEATMAP_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_xt_grid_csv(_bust=0):
    """Per-player xT by zone (current season): id, temporada, kind (gen/prev),
    zx, zy, xt.  Powers the Player Profile xT maps."""
    return _read_id_csv(_PLAYER_XT_GRID_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_team_sonar_csv(_bust=0):
    return _read_team_xt(_TEAM_SONAR_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_sonar_csv(_bust=0):
    return _read_id_csv(_PLAYER_SONAR_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_player_pivot_csv(_bust=0):
    """Pivot Index per midfielder-season: id, temporada, liga, equipo, minutes,
    CONTROL / PROGRESSION / ANCHOR / PIVOT and the raw components.  Built by
    build_pivot_index.py; empty frame if it hasn't been run."""
    return _read_id_csv(_PLAYER_PIVOT_CSV)


META_COLS = {"nombre", "posicion", "posicion_detail", "league_display", "Player",
             "equipo", "Appearances", "Time Played", "estimated_90s"}

# ── Metric groupings (Opta column names) ─────────────────────────────────────

OFFENSIVE_METRICS = [
    "Goals", "Goals Openplay", "npxG", "npG-xG", "xG/Shot",
    "Total Shots", "Shots On Target ( inc goals )",
    "Goals from Inside Box", "Goals from Outside Box", "Headed Goals",
    "Total Touches In Opposition Box", "Total Big Chances Scored",
    "Total Big Chances Missed", "Shots Created",
    "Total Fouls Won",
    "Dribble %",
]

DEFENSIVE_METRICS = [
    "Total Tackles", "Tackles Won", "Interceptions", "Total Clearances", "Recoveries",
    "Blocked Shots", "Blocks", "Aerial Duels won", "Ground Duels won",
    "Duels won",
    "Tackles Lost",
    "Tackle Win %", "Aerial Win %", "Ground Duel %", "Duel %",
]

BALL_PROGRESSION_METRICS = [
    "Progressive Carries", "Carries",
    "Forward Passes", "Through balls", "Final Third Touches",
]

PASSING_METRICS = [
    "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
    "Key Passes (Attempt Assists)", "Goal Assists", "xA",
    "Successful Long Passes", "Forward Passes",
    "Successful Open Play Passes", "Through balls",
    "Successful Crosses & Corners",
    "Pass %", "Cross %", "Long Pass %", "Short Pass %",
]

DRIBBLING_METRICS = [
    "Successful Dribbles", "Unsuccessful Dribbles",
    "Progressive Carries", "Carries",
    "Overruns",
    "Dribble %",
]

GK_METRICS = [
    "Saves Made", "Clean Sheets",
    "Penalties Saved", "Catches", "Punches",
    "Total Big Chances Saved",
    "GK Successful Distribution", "GK Unsuccessful Distribution",
    "Save %", "Launch %", "Goals Prevented",
    "PSxG", "PSxG+/-", "PSxG/Shot",
    "Penalty Save %", "Caught %", "Claim %",
]

# ── GK PSxG (post-shot xG) calibration weights ───────────────────────────────
# Shots counted are on-target (saves + goals conceded).  Post-shot conversion
# rates run higher than pre-shot xG, so these weights are calibrated to
# on-target shots by location.  Big chances saved are priced as premium chances.
PSXG_W_INSIDE = 0.34       # normal on-target shot, inside box (non-penalty)
PSXG_W_BIG_CHANCE = 0.55   # big chance — clear scoring opportunity
PSXG_W_OUTSIDE = 0.12      # on-target shot from outside the box
PSXG_W_PENALTY = 0.79      # penalty on target

# Regression-to-mean constant for the graded PSxG+/-.  A keeper's measured
# PSxG+/- is shrunk toward 0 by shots/(shots+k) so a hot/cold partial season
# can't swing the shot-stopping grade as hard as a full one.  ~30 on-target
# shots ≈ a few matches; tuned so a full season (~130 faced) keeps ~80% weight.
PSXG_SHRINK_K = 30.0

# Same idea for finishing (npG − xG): regress toward 0 by non-penalty shots so a
# small sample of hot/cold finishing can't swing the grade. ~40 shots ≈ a
# striker's half-season; a full ~100-shot season keeps ~70% weight.
XG_FINISH_SHRINK_K = 40.0

DISCIPLINE_METRICS = ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"]

# ── Possession Adjustment (Padj) metric classification ──────────────────────
# Defensive metrics: occur more when opponent has the ball.
#   Padj = raw × (50 / opponent_possession)
PADJ_DEFENSIVE_METRICS = {
    "Total Tackles", "Tackles Won", "Tackles Lost",
    "Interceptions", "Total Clearances", "Recoveries",
    "Blocked Shots", "Blocks",
    "Aerial Duels", "Aerial Duels won", "Aerial Duels lost",
    "Ground Duels", "Ground Duels won", "Ground Duels lost",
    "Duels", "Duels won", "Duels lost",
    "Saves Made", "Catches", "Punches",
    "Total Big Chances Saved", "Penalties Saved",
    "Goals Conceded", "Goals Conceded Inside Box", "Goals Conceded Outside Box",
    "Saves Made from Inside Box", "Saves Made from Outside Box",
    "Clearances Off the Line", "Foul Attempted Tackle",
    "Saves from Penalty", "Crosses not Claimed",
    "Penalties Faced",
}
# Attacking / possession-correlated metrics: occur more when team has the ball.
#   Padj = raw × (50 / team_possession)
PADJ_ATTACKING_METRICS = {
    "Goals", "Non-Penalty Goals", "Goals Openplay", "Goals from Inside Box",
    "Goals from Outside Box", "Headed Goals", "Left Foot Goals", "Right Foot Goals",
    "Penalty Goals", "Direct Setpiece Goals", "Set Pieces Goals",
    "Total Shots", "Shots On Target ( inc goals )",
    "Shots Off Target (inc woodwork)", "Shots Created",
    "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
    "Total Unsuccessful Passes ( Excl Crosses & Corners )",
    "Key Passes (Attempt Assists)", "Goal Assists", "Second Goal Assists",
    "Forward Passes", "Backward Passes", "Leftside Passes", "Rightside Passes",
    "Successful Long Passes", "Unsuccessful Long Passes",
    "Successful Open Play Passes", "Open Play Passes",
    "Through balls", "Final Third Touches",
    "Successful Crosses & Corners", "Unsuccessful Crosses & Corners",
    "Successful Crosses open play", "Unsuccessful Crosses open play",
    "Successful Short Passes", "Unsuccessful Short Passes",
    "Successful Dribbles", "Unsuccessful Dribbles",
    "Progressive Carries", "Carries", "Overruns",
    "Total Touches In Opposition Box", "Touches",
    "Total Big Chances Scored", "Total Big Chances Missed",
    "Total Big Chances Created",
    "Total Fouls Won", "Offsides",
    "GK Successful Distribution", "GK Unsuccessful Distribution",
    "Successful Launches", "Unsuccessful Launches",
    "Successful Lay-offs", "Unsuccessful lay-offs",
    "Successful Passes Opposition Half", "Unsuccessful Passes Opposition Half",
    "Successful Passes Own Half", "Unsuccessful Passes Own Half",
    "Corners Taken (incl short corners)", "Successful Corners into Box",
    "Unsuccessful Corners into Box", "Corners Won",
    "Total Losses Of Possession",
}

# Position mapping: Opta broad position → general group
# Opta uses: Goalkeeper, Defender, Midfielder, Forward
# We also keep the old BeSoccer detail codes if any data still uses them.
POSITION_MAP = {
    # Opta broad positions
    "Goalkeeper": "Goalkeeper",
    "Defender": "Centre-Back",
    "Midfielder": "Central Midfield",
    "Forward": "Striker",
    # Legacy BeSoccer detail codes (kept for compatibility)
    "GK": "Goalkeeper",
    "CB": "Centre-Back", "LB": "Full-Back", "RB": "Full-Back", "RWB": "Full-Back", "LWB": "Full-Back",
    "DM": "Central Midfield",
    "CM": "Central Midfield", "LM": "Central Midfield", "RM": "Central Midfield",
    "CAM": "Attacking Midfield", "RAM": "Attacking Midfield", "LAM": "Attacking Midfield",
    "ST": "Striker", "CF": "Striker", "LW": "Wingers", "RW": "Wingers",
}
# Keep broad position groups for any code that still references them
_DEFENDER_POSITIONS = {"Centre-Back", "Full-Back"}
_MIDFIELDER_POSITIONS = {"Central Midfield", "Attacking Midfield", "Wingers"}
_FORWARD_POSITIONS = {"Wingers", "Striker"}

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="stMetric"] label { color: #b7e4c7 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: white !important; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────

_EXTRA_TRANSLIT = str.maketrans("ĐđØøŁłßÞþ", "DdOoLlsTt")

def _normalize_name(name):
    """Strip accents and lowercase for fuzzy matching."""
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFD", str(name).strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.translate(_EXTRA_TRANSLIT)
    return s.lower().strip()


def _ascii_name(name):
    """ASCII fold a name but keep its case, e.g. 'B. Šeško' -> 'B. Sesko'.
    Used as a typeable alias so accented names can be searched without the
    special characters."""
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFD", str(name).strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.translate(_EXTRA_TRANSLIT).strip()


def _make_match_key(name):
    """First-initial + last-name key for matching: 'A. Isak' -> 'a.isak'."""
    parts = name.split()
    if not parts:
        return ""
    return parts[0][0] + "." + parts[-1] if len(parts) > 1 else parts[0]


def _build_team_lookup():
    """No longer needed — Opta data includes equipo directly."""
    return pd.DataFrame()


# Current season uses jugadores_seasonstats.csv; past seasons live in
# jugadores_historical.csv (one file per league, filtered by 'temporada').
CURRENT_SEASON = "2025-2026"


def _load_league_season(folder_path, season):
    """Load one league's data for a given season.

    Current season -> jugadores_seasonstats.csv.
    Past season    -> jugadores_historical.csv filtered to that 'temporada'.
    """
    if season == CURRENT_SEASON:
        csv_path = os.path.join(folder_path, "jugadores_seasonstats.csv")
    else:
        csv_path = os.path.join(folder_path, "jugadores_historical.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    if season != CURRENT_SEASON and "temporada" in df.columns:
        df = df[df["temporada"].astype(str) == season]
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def get_available_seasons():
    """List of selectable seasons, current first then past seasons (desc)."""
    seasons = set()
    for folder_name in LEAGUE_FOLDERS.values():
        path = os.path.join(OPTA_DIR, folder_name, "jugadores_historical.csv")
        if not os.path.exists(path):
            continue
        try:
            t = pd.read_csv(path, encoding="utf-8-sig", usecols=["temporada"], low_memory=False)
            seasons.update(str(x) for x in t["temporada"].dropna().unique())
        except Exception:
            pass
    past = sorted((s for s in seasons if s and s != CURRENT_SEASON), reverse=True)
    return [CURRENT_SEASON] + past


def _compute_gk_derived(df):
    """Compute goalkeeper shot-stopping & command metrics for a DataFrame.

    Returns a dict of {column_name: Series}.  Shared by the raw and the
    possession-adjusted pipelines so the PSxG model lives in exactly one
    place and the two can never drift apart.

    PSxG (post-shot expected goals) approximates the goals an average keeper
    would concede from the on-target shots faced, using calibrated post-shot
    weights by location.  Big chances saved are priced as premium chances, so
    a keeper who repels clear openings earns more credit than one facing
    routine efforts.  PSxG+/- (PSxG minus actual goals conceded) is the
    "xG prevented" figure — positive means above-average shot-stopping.
    """
    out = {}

    _has_ib = ("Saves Made from Inside Box" in df.columns
               and "Goals Conceded Inside Box" in df.columns)
    _has_ob = ("Saves Made from Outside Box" in df.columns
               and "Goals Conceded Outside Box" in df.columns)
    if _has_ib and _has_ob:
        shots_ib = (df["Saves Made from Inside Box"].fillna(0)
                    + df["Goals Conceded Inside Box"].fillna(0))
        shots_ob = (df["Saves Made from Outside Box"].fillna(0)
                    + df["Goals Conceded Outside Box"].fillna(0))
        pens = (df["Penalties Faced"].fillna(0)
                if "Penalties Faced" in df.columns
                else pd.Series(0, index=df.index))
        big_saved = (df["Total Big Chances Saved"].fillna(0)
                     if "Total Big Chances Saved" in df.columns
                     else pd.Series(0, index=df.index))

        # Split non-penalty inside-box on-target shots into big chances vs normal.
        # Cap big chances at the inside-box count so the buckets never overlap.
        shots_ib_np = (shots_ib - pens).clip(lower=0)
        big_ib = np.minimum(big_saved, shots_ib_np)
        normal_ib = (shots_ib_np - big_ib).clip(lower=0)

        psxg = (normal_ib * PSXG_W_INSIDE
                + big_ib * PSXG_W_BIG_CHANCE
                + shots_ob * PSXG_W_OUTSIDE
                + pens * PSXG_W_PENALTY).round(1)
        out["PSxG"] = psxg
        if "Goals Conceded" in df.columns:
            out["PSxG+/-"] = (psxg - df["Goals Conceded"].fillna(0)).round(1)
        total_shots = shots_ib + shots_ob
        out["PSxG/Shot"] = (psxg / total_shots.replace(0, np.nan)).round(3)
        out["Inside Box Save %"] = (
            df["Saves Made from Inside Box"].fillna(0)
            / shots_ib.replace(0, np.nan) * 100
        ).round(1)
        out["Outside Box Save %"] = (
            df["Saves Made from Outside Box"].fillna(0)
            / shots_ob.replace(0, np.nan) * 100
        ).round(1)

    # Penalty Save % — penalty stopping isolated from open-play save %
    if "Penalties Saved" in df.columns and "Penalties Faced" in df.columns:
        out["Penalty Save %"] = (
            df["Penalties Saved"].fillna(0)
            / df["Penalties Faced"].replace(0, np.nan) * 100
        ).round(1)

    # Caught % — share of saves held cleanly vs parried (rebound risk)
    if "Saves made - caught" in df.columns and "Saves made - parried" in df.columns:
        held = df["Saves made - caught"].fillna(0)
        handled = held + df["Saves made - parried"].fillna(0)
        out["Caught %"] = (held / handled.replace(0, np.nan) * 100).round(1)

    # Claim % — command of area: crosses claimed (catches + punches) vs spilled
    if ("Catches" in df.columns and "Punches" in df.columns
            and "Crosses not Claimed" in df.columns):
        claimed = df["Catches"].fillna(0) + df["Punches"].fillna(0)
        cross_actions = claimed + df["Crosses not Claimed"].fillna(0)
        out["Claim %"] = (claimed / cross_actions.replace(0, np.nan) * 100).round(1)

    # ── Override with TRUE shot-level PSxG where available ────────────────────
    # build_gk_psxg.py measures PSxG/PSxG+/- per keeper-season from the raw event
    # JSONs.  Those measured values replace the season-aggregate approximation
    # above (which remains the fallback for any keeper-season not in the CSV).
    # A measured xG is a real quantity, so it is used verbatim in both the raw
    # and Padj pipelines (never re-scaled by possession).
    gk_psxg = _load_gk_psxg_csv(_bust=_csv_mtime(_GK_PSXG_CSV))
    if gk_psxg and "id" in df.columns and "temporada" in df.columns:
        keys = list(zip(df["id"].astype(str), df["temporada"].astype(str)))
        m_psxg = pd.Series([gk_psxg.get(k, {}).get("PSxG") for k in keys],
                           index=df.index, dtype=float)
        m_pm = pd.Series([gk_psxg.get(k, {}).get("PSxG+/-") for k in keys],
                         index=df.index, dtype=float)
        m_sh = pd.Series([gk_psxg.get(k, {}).get("shots") for k in keys],
                         index=df.index, dtype=float)
        m_soft = pd.Series([gk_psxg.get(k, {}).get("soft") for k in keys],
                           index=df.index, dtype=float)
        has = m_psxg.notna()
        if has.any():
            base_psxg = out.get("PSxG", pd.Series(np.nan, index=df.index))
            base_pm = out.get("PSxG+/-", pd.Series(np.nan, index=df.index))
            base_sps = out.get("PSxG/Shot", pd.Series(np.nan, index=df.index))
            out["PSxG"] = base_psxg.where(~has, m_psxg).round(1)
            out["PSxG+/-"] = base_pm.where(~has, m_pm).round(1)
            meas_sps = (m_psxg / m_sh.replace(0, np.nan)).round(3)
            out["PSxG/Shot"] = base_sps.where(~has, meas_sps)
            # Shrunk PSxG+/- — the graded shot-stopping signal (measured rows only).
            out["PSxG+/- (shrunk)"] = (m_pm * m_sh / (m_sh + PSXG_SHRINK_K)).round(2)
            # Saveable goals conceded — soft goals PSxG+/- nets away (see builder).
            out["Saveable Goals Conceded"] = m_soft.where(has)

    return out


def _compute_xg_derived(df):
    """Pre-shot xG columns per player from the measured model (build_player_xg.py),
    merged by (id, temporada).  Covers every shooter, not just strikers.

    Returns {col: Series}: xG, npxG (counting stats → divided in per-90 mode),
    npG-xG (finishing), xG/Shot, and a sample-shrunk finishing figure used by the
    grade.  Like the keeper PSxG, these are measured quantities and are used
    verbatim in both the raw and Padj pipelines.
    """
    out = {}
    xg_data = _load_player_xg_csv(_bust=_csv_mtime(_PLAYER_XG_CSV))
    if not xg_data or "id" not in df.columns or "temporada" not in df.columns:
        return out
    keys = list(zip(df["id"].astype(str), df["temporada"].astype(str)))
    m_xg = pd.Series([xg_data.get(k, {}).get("xG") for k in keys],
                     index=df.index, dtype=float)
    m_npxg = pd.Series([xg_data.get(k, {}).get("npxG") for k in keys],
                       index=df.index, dtype=float)
    m_fin = pd.Series([xg_data.get(k, {}).get("npG-xG") for k in keys],
                      index=df.index, dtype=float)
    m_sh = pd.Series([xg_data.get(k, {}).get("shots") for k in keys],
                     index=df.index, dtype=float)
    m_nps = pd.Series([xg_data.get(k, {}).get("np_shots") for k in keys],
                      index=df.index, dtype=float)
    m_xa = pd.Series([xg_data.get(k, {}).get("xA") for k in keys],
                     index=df.index, dtype=float)
    has = m_xg.notna()
    if has.any():
        out["xG"] = m_xg.round(2)
        out["npxG"] = m_npxg.round(2)
        out["npG-xG"] = m_fin.round(2)
        out["xG/Shot"] = (m_xg / m_sh.replace(0, np.nan)).round(3)
        # Shrunk finishing — the graded signal (measured rows only).
        out["npG-xG (shrunk)"] = (m_fin * m_nps / (m_nps + XG_FINISH_SHRINK_K)).round(2)
        # Expected assists, and combined expected goal involvement.
        out["xA"] = m_xa.round(2)
        out["npxG+xA"] = (m_npxg + m_xa).round(2)
    return out


def _compute_xt_derived(df):
    """Per-player expected threat (build_spatial.py), merged by (id, temporada):
    xT Generated (build-up threat from open-play moves) and xT Prevented (threat
    denied by ball-winning defensive actions).  Season totals — the per-90 loop
    turns them into xT/90.
    """
    out = {}
    xt_data = _load_player_xt_csv(_bust=_csv_mtime(_PLAYER_XT_CSV))
    if not xt_data or "id" not in df.columns or "temporada" not in df.columns:
        return out
    keys = list(zip(df["id"].astype(str), df["temporada"].astype(str)))
    m_gen = pd.Series([xt_data.get(k, {}).get("gen") for k in keys],
                      index=df.index, dtype=float)
    m_prev = pd.Series([xt_data.get(k, {}).get("prevented") for k in keys],
                       index=df.index, dtype=float)
    if m_gen.notna().any() or m_prev.notna().any():
        out["xT Generated"] = m_gen.round(3)
        out["xT Prevented"] = m_prev.round(3)
    return out


def _data_fingerprint():
    """Max mtime across all player CSVs — used as a cache-buster so load_data
    (and trajectories) automatically re-read when the data files change on disk,
    without needing a manual cache clear or app restart."""
    mt = 0
    for folder_name in LEAGUE_FOLDERS.values():
        for fn in ("jugadores_seasonstats.csv", "jugadores_historical.csv"):
            try:
                mt = max(mt, int(os.path.getmtime(os.path.join(OPTA_DIR, folder_name, fn))))
            except OSError:
                pass
    mt = max(mt, _csv_mtime(_GK_PSXG_CSV))   # refresh when measured PSxG changes
    mt = max(mt, _csv_mtime(_PLAYER_XG_CSV))  # …and when player xG changes
    mt = max(mt, _csv_mtime(_PLAYER_XT_CSV))  # …and when player xT changes
    mt = max(mt, _csv_mtime(_FINANCIALS_CSV))  # …and when values/ages change
    return mt


@st.cache_data(show_spinner="Loading football data...")
def _load_data_cached(season, _bust):
    """Load all Opta data for *season*.  Returns dict with 'total'/'per90' etc.

    Opta provides only season totals.  Per-90 values are computed from
    Time Played: stat_per90 = stat_total / (Time Played / 90).
    *_bust* is a data fingerprint that invalidates the cache when CSVs change.
    """
    total_frames = []

    for display_name, folder_name in LEAGUE_FOLDERS.items():
        folder_path = os.path.join(OPTA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        df = _load_league_season(folder_path, season)
        if df is not None and not df.empty:
            df["league_display"] = display_name
            total_frames.append(df)

    result = {"total": pd.DataFrame(), "per90": pd.DataFrame()}

    if not total_frames:
        return result

    combined = pd.concat(total_frames, ignore_index=True)

    # ── Normalize column names ──────────────────────────────────────
    # The Opta CSV already has 'nombre', 'equipo', 'posicion' columns.
    # Map posicion → posicion_detail (raw) and posicion → broad group
    combined["posicion_detail"] = combined["posicion"]
    combined["posicion"] = combined["posicion_detail"].map(POSITION_MAP).fillna("Unknown")

    # Auto-correct mislabeled goalkeepers
    if "Saves Made" in combined.columns:
        gk_fix = (
            (combined["posicion_detail"] != "Goalkeeper")
            & (pd.to_numeric(combined["Saves Made"], errors="coerce") > 2)
        )
        combined.loc[gk_fix, "posicion_detail"] = "Goalkeeper"
        combined.loc[gk_fix, "posicion"] = "Goalkeeper"

    # Convert stat columns to numeric
    for col in combined.columns:
        if col not in META_COLS and col not in {"liga", "temporada", "id", "dorsal"}:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["equipo"] = combined["equipo"].fillna("Unknown")
    combined["Appearances"] = pd.to_numeric(combined.get("Appearances"), errors="coerce")
    combined["Time Played"] = pd.to_numeric(combined.get("Time Played"), errors="coerce")

    # ── Compute derived stats (batched to avoid DataFrame fragmentation) ──
    _derived = {}

    # Goals minus penalties
    if "Goals" in combined.columns and "Penalty Goals" in combined.columns:
        _derived["Non-Penalty Goals"] = combined["Goals"].fillna(0) - combined["Penalty Goals"].fillna(0)

    # Dribble success %
    if "Successful Dribbles" in combined.columns and "Unsuccessful Dribbles" in combined.columns:
        total_drib = combined["Successful Dribbles"].fillna(0) + combined["Unsuccessful Dribbles"].fillna(0)
        _derived["Dribble %"] = (combined["Successful Dribbles"].fillna(0) / total_drib.replace(0, np.nan) * 100).round(1)

    # Aerial duel win %
    if "Aerial Duels won" in combined.columns and "Aerial Duels" in combined.columns:
        _derived["Aerial Win %"] = (
            combined["Aerial Duels won"].fillna(0) / combined["Aerial Duels"].replace(0, np.nan) * 100
        ).round(1)

    # Tackle win %
    if "Tackles Won" in combined.columns and "Total Tackles" in combined.columns:
        _derived["Tackle Win %"] = (
            combined["Tackles Won"].fillna(0) / combined["Total Tackles"].replace(0, np.nan) * 100
        ).round(1)

    # Pass success %
    if "Total Successful Passes ( Excl Crosses & Corners ) " in combined.columns and "Total Passes" in combined.columns:
        _derived["Pass %"] = (
            combined["Total Successful Passes ( Excl Crosses & Corners ) "].fillna(0) /
            combined["Total Passes"].replace(0, np.nan) * 100
        ).round(1)

    # GK Save %
    if "Saves Made" in combined.columns and "Shots On Target ( inc goals )" in combined.columns:
        shots_faced = combined["Saves Made"].fillna(0) + combined["Goals Conceded"].fillna(0)
        _derived["Save %"] = (combined["Saves Made"].fillna(0) / shots_faced.replace(0, np.nan) * 100).round(1)

    # GK Goals Prevented
    if "Saves Made" in combined.columns and "Goals Conceded" in combined.columns:
        _derived["Goals Prevented"] = combined["Saves Made"].fillna(0) - combined["Goals Conceded"].fillna(0)

    # ── GK shot-stopping (PSxG) & command metrics ─────────────────────────
    # Calibrated PSxG/PSxG+/-, penalty stopping, handling and command of area.
    # Computed via shared helper so raw and Padj pipelines stay in sync.
    _derived.update(_compute_gk_derived(combined))

    # ── Player pre-shot xG, npxG & finishing (npG − xG) ───────────────────
    _derived.update(_compute_xg_derived(combined))

    # ── Player expected threat (generated & prevented) ────────────────────
    _derived.update(_compute_xt_derived(combined))

    # GK Launch %
    if "Successful Launches" in combined.columns and "Unsuccessful Launches" in combined.columns:
        total_launches = combined["Successful Launches"].fillna(0) + combined["Unsuccessful Launches"].fillna(0)
        _derived["Launch %"] = (combined["Successful Launches"].fillna(0) / total_launches.replace(0, np.nan) * 100).round(1)

    # Ground Duel Win %
    if "Ground Duels won" in combined.columns and "Ground Duels" in combined.columns:
        _derived["Ground Duel %"] = (
            combined["Ground Duels won"].fillna(0) / combined["Ground Duels"].replace(0, np.nan) * 100
        ).round(1)

    # Overall Duel Win %
    if "Duels won" in combined.columns and "Duels" in combined.columns:
        _derived["Duel %"] = (
            combined["Duels won"].fillna(0) / combined["Duels"].replace(0, np.nan) * 100
        ).round(1)

    # Cross Success % (open play)
    if "Successful Crosses open play" in combined.columns and "Unsuccessful Crosses open play" in combined.columns:
        total_cross = combined["Successful Crosses open play"].fillna(0) + combined["Unsuccessful Crosses open play"].fillna(0)
        _derived["Cross %"] = (combined["Successful Crosses open play"].fillna(0) / total_cross.replace(0, np.nan) * 100).round(1)

    # Long Pass Success %
    if "Successful Long Passes" in combined.columns and "Unsuccessful Long Passes" in combined.columns:
        total_lp = combined["Successful Long Passes"].fillna(0) + combined["Unsuccessful Long Passes"].fillna(0)
        _derived["Long Pass %"] = (combined["Successful Long Passes"].fillna(0) / total_lp.replace(0, np.nan) * 100).round(1)

    # Short Pass Success %
    if "Successful Short Passes" in combined.columns and "Unsuccessful Short Passes" in combined.columns:
        total_sp = combined["Successful Short Passes"].fillna(0) + combined["Unsuccessful Short Passes"].fillna(0)
        _derived["Short Pass %"] = (combined["Successful Short Passes"].fillna(0) / total_sp.replace(0, np.nan) * 100).round(1)

    # Ball Retention % (100 - losses per touch * 100) — higher is better
    if "Total Losses Of Possession" in combined.columns and "Touches" in combined.columns:
        _derived["Retention %"] = (100 - combined["Total Losses Of Possession"].fillna(0) / combined["Touches"].replace(0, np.nan) * 100).round(1)

    # Own Half Pass % (successful own-half passes / total own-half passes * 100)
    if "Successful Passes Own Half" in combined.columns and "Unsuccessful Passes Own Half" in combined.columns:
        total_oh = combined["Successful Passes Own Half"].fillna(0) + combined["Unsuccessful Passes Own Half"].fillna(0)
        _derived["Own Half Pass %"] = (combined["Successful Passes Own Half"].fillna(0) / total_oh.replace(0, np.nan) * 100).round(1)

    # Estimated 90s
    _derived["estimated_90s"] = (combined["Time Played"].fillna(0) / 90).round(2)

    # GK rate metrics — derived as rates so they are NOT re-divided in per-90 mode
    if "Saves Made" in combined.columns:
        _e90 = (combined["Time Played"].fillna(0) / 90).replace(0, np.nan)
        _derived["Saves/90"] = (combined["Saves Made"].fillna(0) / _e90).round(2)
    if "Clean Sheets" in combined.columns and "Appearances" in combined.columns:
        _derived["Clean Sheet %"] = (
            combined["Clean Sheets"].fillna(0) / combined["Appearances"].replace(0, np.nan) * 100
        ).round(1)

    # Assign all derived columns at once (avoids DataFrame fragmentation)
    if _derived:
        combined = pd.concat([combined, pd.DataFrame(_derived, index=combined.index)], axis=1)

    # ── Sub-classify defenders into CB vs FB ───────────────────────
    # Opta only provides "Defender" — use wide-attacking vs central-defensive
    # balance to split into Centre-Back vs Full-Back.
    # Full-backs are defined by attacking the FLANK (crossing, wide dribbling) —
    # not by being a ball-player.  Key Passes / box touches were flagging
    # ball-playing centre-backs (e.g. Militão) as full-backs, so the "wide" set
    # is crossing-led, and a margin keeps borderline defenders as CB (the safer
    # default) rather than a hard 50/50 median split.
    _DEF_WIDE = ["Successful Crosses open play", "Unsuccessful Crosses open play",
                 "Successful Crosses & Corners", "Successful Dribbles"]
    _DEF_CENTRAL = ["Aerial Duels won", "Aerial Duels", "Total Clearances",
                    "Blocked Shots", "Headed Goals", "Interceptions"]
    def_mask = combined["posicion"] == "Centre-Back"
    if def_mask.sum() > 10:
        wide_avail = [m for m in _DEF_WIDE if m in combined.columns]
        central_avail = [m for m in _DEF_CENTRAL if m in combined.columns]
        if wide_avail and central_avail:
            def_idx = combined.index[def_mask]
            wide_pct = combined.loc[def_idx, wide_avail].rank(pct=True).mean(axis=1)
            central_pct = combined.loc[def_idx, central_avail].rank(pct=True).mean(axis=1)
            balance = wide_pct - central_pct
            fb_idx = balance[balance > 0.08].index
            combined.loc[fb_idx, "posicion"] = "Full-Back"
            combined.loc[fb_idx, "posicion_detail"] = "FB"

    # ── Sub-classify midfielders into CM vs CAM ─────────────────────
    # Opta only provides "Midfielder" — use attacking-vs-defensive balance
    # to split into Central Midfield (CM/DM) vs Attacking Midfield (CAM).
    _MF_ATK = ["Goals", "Total Shots", "Key Passes (Attempt Assists)",
               "Total Touches In Opposition Box", "Successful Dribbles"]
    _MF_DEF = ["Total Tackles", "Interceptions", "Recoveries", "Total Clearances"]
    mf_mask = combined["posicion"] == "Central Midfield"
    if mf_mask.sum() > 10:
        atk_avail = [m for m in _MF_ATK if m in combined.columns]
        def_avail = [m for m in _MF_DEF if m in combined.columns]
        if atk_avail and def_avail:
            mf_idx = combined.index[mf_mask]
            atk_pct = combined.loc[mf_idx, atk_avail].rank(pct=True).mean(axis=1)
            def_pct = combined.loc[mf_idx, def_avail].rank(pct=True).mean(axis=1)
            balance = atk_pct - def_pct
            cam_idx = balance[balance > 0].index
            combined.loc[cam_idx, "posicion"] = "Attacking Midfield"
            combined.loc[cam_idx, "posicion_detail"] = "CAM"

    # ── Sub-classify forwards into Striker vs Wingers ───────────────
    # Opta lumps every forward into "Forward" → "Striker", so wide players
    # (and the Wingers roles) never surface.  Split on wide-play (crossing,
    # dribbling) vs central play (box touches, headers, aerials, big chances).
    _FWD_WIDE = ["Successful Crosses open play", "Unsuccessful Crosses open play",
                 "Successful Crosses & Corners", "Successful Dribbles"]
    _FWD_CENTRAL = ["Headed Goals", "Goals from Inside Box", "Total Big Chances Scored",
                    "Aerial Duels won", "Total Touches In Opposition Box"]
    fwd_mask = combined["posicion"] == "Striker"
    if fwd_mask.sum() > 10:
        wide_avail = [m for m in _FWD_WIDE if m in combined.columns]
        central_avail = [m for m in _FWD_CENTRAL if m in combined.columns]
        if wide_avail and central_avail:
            fwd_idx = combined.index[fwd_mask]
            wide_pct = combined.loc[fwd_idx, wide_avail].rank(pct=True).mean(axis=1)
            central_pct = combined.loc[fwd_idx, central_avail].rank(pct=True).mean(axis=1)
            balance = wide_pct - central_pct
            wing_idx = balance[balance > 0].index
            combined.loc[wing_idx, "posicion"] = "Wingers"
            combined.loc[wing_idx, "posicion_detail"] = "Winger"

    # ── Drop players with zero minutes ──────────────────────────────
    combined = combined[combined["Time Played"].fillna(0) > 0].reset_index(drop=True)

    # ── Attach current-season financials (age, market value) ────────────
    # Age feeds Potential Grading and the value model; market value (€m) is the
    # target the value model is benchmarked against.  Current season only —
    # financials are scraped for the live squads.
    if season == CURRENT_SEASON:
        _fin = _load_financials_csv(_bust=_csv_mtime(_FINANCIALS_CSV))
        if _fin:
            _recs = [_fin_rec(_fin, n, t)
                     for n, t in zip(combined["nombre"], combined["equipo"])]
            combined["age"] = pd.to_numeric(
                pd.Series([r.get("age") for r in _recs], index=combined.index),
                errors="coerce")
            combined["market_value_m"] = pd.Series(
                [_mv_millions(r.get("market_value")) for r in _recs],
                index=combined.index)

    result["total"] = combined

    # ── Compute Per 90 ──────────────────────────────────────────────
    per90 = combined.copy()
    _min_minutes = 90  # need at least 90 mins to compute per-90
    for col in per90.columns:
        if col in META_COLS or col in {"liga", "temporada", "id", "dorsal",
                                        "posicion", "posicion_detail", "league_display",
                                        "equipo", "nombre", "Appearances", "Time Played",
                                        "estimated_90s",
                                        # Percentage stats should NOT be divided by 90s
                                        "Dribble %", "Aerial Win %", "Tackle Win %",
                                        "Pass %", "Save %", "Launch %",
                                        "Ground Duel %", "Duel %", "Cross %",
                                        "Long Pass %", "Short Pass %",
                                        "Retention %", "Own Half Pass %",
                                        # GK ratio metrics — already rates, do not re-divide
                                        "Inside Box Save %", "Outside Box Save %",
                                        "PSxG/Shot", "Penalty Save %", "Caught %", "Claim %",
                                        # PSxG model outputs are season-level quality figures
                                        # (like xG totals) — dividing a +/- differential by 90s
                                        # is meaningless, so keep them as-is in per-90 mode.
                                        "PSxG", "PSxG+/-", "PSxG+/- (shrunk)",
                                        "Saveable Goals Conceded",
                                        # xG finishing differentials/rates — not per-90 counts
                                        # (xG / npxG themselves DO divide → xG/90, npxG/90)
                                        "npG-xG", "npG-xG (shrunk)", "xG/Shot",
                                        # financials — not per-90 quantities
                                        "age", "market_value_m",
                                        # GK rate metrics already normalised — do not re-divide
                                        "Saves/90", "Clean Sheet %"}:
            continue
        if per90[col].dtype in ("float64", "int64", "float32", "int32"):
            per90[col] = np.where(
                per90["Time Played"].fillna(0) >= _min_minutes,
                (per90[col] / per90["estimated_90s"].replace(0, np.nan)).round(2),
                np.nan,
            )
    result["per90"] = per90

    # ── Compute Possession Adjustment (Padj) ───────────────────────────
    # Estimate team possession per league from total passes
    _PCTILE_COLS_SKIP = META_COLS | {
        "liga", "temporada", "id", "dorsal",
        "posicion", "posicion_detail", "league_display",
        "equipo", "nombre", "Appearances", "Time Played",
        "estimated_90s",
        "Dribble %", "Aerial Win %", "Tackle Win %",
        "Pass %", "Save %", "Launch %",
        "Ground Duel %", "Duel %", "Cross %",
        "Long Pass %", "Short Pass %",
        "Retention %", "Own Half Pass %",
        # GK ratio metrics — already rates, do not re-divide
        "Inside Box Save %", "Outside Box Save %",
        "PSxG/Shot", "Penalty Save %", "Caught %", "Claim %",
        "PSxG", "PSxG+/-", "PSxG+/- (shrunk)", "Saveable Goals Conceded",
        "npG-xG", "npG-xG (shrunk)", "xG/Shot",
        "age", "market_value_m",
        # GK rate metrics already normalised — do not re-divide
        "Saves/90", "Clean Sheet %",
    }
    if "Total Passes" in combined.columns:
        team_poss = {}
        for league in combined["league_display"].unique():
            lg_df = combined[combined["league_display"] == league]
            tp = lg_df.groupby("equipo")["Total Passes"].sum()
            total = tp.sum()
            n_teams = len(tp)
            if total > 0 and n_teams > 0:
                for team, passes in tp.items():
                    poss_pct = passes / total * n_teams * 50
                    # Clamp to [38,62] — keeps the adjustment factor in a sane
                    # ~0.81–1.32 range instead of up to 1.67 at the extremes.
                    poss_pct = max(38.0, min(62.0, poss_pct))
                    team_poss[(league, team)] = poss_pct

        # Build lookup DataFrame for vectorized merge
        poss_records = [
            {"league_display": k[0], "equipo": k[1], "_team_poss": v}
            for k, v in team_poss.items()
        ]
        poss_df = pd.DataFrame(poss_records) if poss_records else pd.DataFrame(
            columns=["league_display", "equipo", "_team_poss"]
        )

        padj = combined.merge(poss_df, on=["league_display", "equipo"], how="left")
        padj["_team_poss"] = padj["_team_poss"].fillna(50.0)
        padj["_opp_poss"] = 100.0 - padj["_team_poss"]

        def_factor = 50.0 / padj["_opp_poss"].replace(0, np.nan)
        atk_factor = 50.0 / padj["_team_poss"].replace(0, np.nan)

        for col in padj.columns:
            if col in PADJ_DEFENSIVE_METRICS and padj[col].dtype in ("float64", "int64", "float32", "int32"):
                padj[col] = (padj[col] * def_factor).round(2)
            elif col in PADJ_ATTACKING_METRICS and padj[col].dtype in ("float64", "int64", "float32", "int32"):
                padj[col] = (padj[col] * atk_factor).round(2)

        # Re-derive percentage stats from Padj-adjusted counts
        if "Successful Dribbles" in padj.columns and "Unsuccessful Dribbles" in padj.columns:
            td = padj["Successful Dribbles"].fillna(0) + padj["Unsuccessful Dribbles"].fillna(0)
            padj["Dribble %"] = (padj["Successful Dribbles"].fillna(0) / td.replace(0, np.nan) * 100).round(1)
        if "Aerial Duels won" in padj.columns and "Aerial Duels" in padj.columns:
            padj["Aerial Win %"] = (padj["Aerial Duels won"].fillna(0) / padj["Aerial Duels"].replace(0, np.nan) * 100).round(1)
        if "Tackles Won" in padj.columns and "Total Tackles" in padj.columns:
            padj["Tackle Win %"] = (padj["Tackles Won"].fillna(0) / padj["Total Tackles"].replace(0, np.nan) * 100).round(1)
        if "Total Successful Passes ( Excl Crosses & Corners ) " in padj.columns and "Total Passes" in padj.columns:
            padj["Pass %"] = (padj["Total Successful Passes ( Excl Crosses & Corners ) "].fillna(0) / padj["Total Passes"].replace(0, np.nan) * 100).round(1)
        if "Saves Made" in padj.columns and "Goals Conceded" in padj.columns:
            sf = padj["Saves Made"].fillna(0) + padj["Goals Conceded"].fillna(0)
            padj["Save %"] = (padj["Saves Made"].fillna(0) / sf.replace(0, np.nan) * 100).round(1)
            padj["Goals Prevented"] = padj["Saves Made"].fillna(0) - padj["Goals Conceded"].fillna(0)
        # Re-derive GK shot-stopping & command metrics from Padj-adjusted counts
        for _gk_col, _gk_val in _compute_gk_derived(padj).items():
            padj[_gk_col] = _gk_val
        # Measured xG/finishing — used verbatim (not possession-scaled)
        for _xg_col, _xg_val in _compute_xg_derived(padj).items():
            padj[_xg_col] = _xg_val
        # Player xT (generated & prevented) — measured, used verbatim
        for _xt_col, _xt_val in _compute_xt_derived(padj).items():
            padj[_xt_col] = _xt_val
        if "Successful Launches" in padj.columns and "Unsuccessful Launches" in padj.columns:
            tl = padj["Successful Launches"].fillna(0) + padj["Unsuccessful Launches"].fillna(0)
            padj["Launch %"] = (padj["Successful Launches"].fillna(0) / tl.replace(0, np.nan) * 100).round(1)
        if "Ground Duels won" in padj.columns and "Ground Duels" in padj.columns:
            padj["Ground Duel %"] = (padj["Ground Duels won"].fillna(0) / padj["Ground Duels"].replace(0, np.nan) * 100).round(1)
        if "Duels won" in padj.columns and "Duels" in padj.columns:
            padj["Duel %"] = (padj["Duels won"].fillna(0) / padj["Duels"].replace(0, np.nan) * 100).round(1)
        if "Successful Crosses open play" in padj.columns and "Unsuccessful Crosses open play" in padj.columns:
            tc = padj["Successful Crosses open play"].fillna(0) + padj["Unsuccessful Crosses open play"].fillna(0)
            padj["Cross %"] = (padj["Successful Crosses open play"].fillna(0) / tc.replace(0, np.nan) * 100).round(1)
        if "Successful Long Passes" in padj.columns and "Unsuccessful Long Passes" in padj.columns:
            tlp = padj["Successful Long Passes"].fillna(0) + padj["Unsuccessful Long Passes"].fillna(0)
            padj["Long Pass %"] = (padj["Successful Long Passes"].fillna(0) / tlp.replace(0, np.nan) * 100).round(1)
        if "Successful Short Passes" in padj.columns and "Unsuccessful Short Passes" in padj.columns:
            tsp = padj["Successful Short Passes"].fillna(0) + padj["Unsuccessful Short Passes"].fillna(0)
            padj["Short Pass %"] = (padj["Successful Short Passes"].fillna(0) / tsp.replace(0, np.nan) * 100).round(1)
        if "Goals" in padj.columns and "Penalty Goals" in padj.columns:
            padj["Non-Penalty Goals"] = padj["Goals"].fillna(0) - padj["Penalty Goals"].fillna(0)
        if "Total Losses Of Possession" in padj.columns and "Touches" in padj.columns:
            padj["Retention %"] = (100 - padj["Total Losses Of Possession"].fillna(0) / padj["Touches"].replace(0, np.nan) * 100).round(1)
        if "Successful Passes Own Half" in padj.columns and "Unsuccessful Passes Own Half" in padj.columns:
            toh = padj["Successful Passes Own Half"].fillna(0) + padj["Unsuccessful Passes Own Half"].fillna(0)
            padj["Own Half Pass %"] = (padj["Successful Passes Own Half"].fillna(0) / toh.replace(0, np.nan) * 100).round(1)

        padj = padj.drop(columns=["_team_poss", "_opp_poss"])
        result["padj"] = padj

        # Padj Per 90
        padj_per90 = padj.copy()
        for col in padj_per90.columns:
            if col in _PCTILE_COLS_SKIP:
                continue
            if padj_per90[col].dtype in ("float64", "int64", "float32", "int32"):
                padj_per90[col] = np.where(
                    padj_per90["Time Played"].fillna(0) >= _min_minutes,
                    (padj_per90[col] / padj_per90["estimated_90s"].replace(0, np.nan)).round(2),
                    np.nan,
                )
        result["padj_per90"] = padj_per90

        # Default Total/Per 90 stay RAW; possession-adjusted views are opt-in
        # (the sidebar "Possession-adjust" toggle selects result["padj"]).
    else:
        result["padj"] = combined.copy()
        result["padj_per90"] = per90.copy()

    return result


def load_data(season=CURRENT_SEASON):
    """Public entry point — re-reads automatically when the CSVs change on disk."""
    return _load_data_cached(season, _data_fingerprint())


@st.cache_data(show_spinner=False)
def _ambiguous_names(_bust=0):
    """Display names whose ACCENT-FOLDED form is shared by more than one distinct
    player id across all seasons — e.g. 'A. Onana' (André/Amadou) or 'Ederson'
    (Man City GK) vs 'Éderson' (Atalanta MF). Such names can't be resolved by the
    name-only photo override, so the photo lookup must disambiguate by team (or
    fall back to the avatar). Returns the set of original display names."""
    norm_ids, norm_names = {}, {}
    for sea in get_available_seasons():
        d = load_data(sea)["total"]
        if d.empty or "id" not in d.columns:
            continue
        for nm, pid in zip(d["nombre"], d["id"]):
            if isinstance(nm, str) and pid is not None and not pd.isna(pid):
                key = _normalize_name(nm)
                norm_ids.setdefault(key, set()).add(pid)
                norm_names.setdefault(key, set()).add(nm)
    amb = set()
    for key, ids in norm_ids.items():
        if len(ids) > 1:
            amb |= norm_names[key]
    return amb


def filter_df(df, leagues=None, positions=None, positions_detail=None, teams=None):
    """Apply common filters."""
    out = df.copy()
    if leagues:
        out = out[out["league_display"].isin(leagues)]
    if positions:
        out = out[out["posicion"].isin(positions)]
    if positions_detail:
        out = out[out["posicion_detail"].isin(positions_detail)]
    if teams:
        out = out[out["equipo"].isin(teams)]
    return out


def _safe_int(val):
    """Safely convert to int, returning 0 for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _ord_suffix(n):
    """Ordinal suffix for a number: 1->'st', 2->'nd', 3->'rd', 11->'th', 92->'nd'."""
    try:
        n = int(round(float(n)))
    except (TypeError, ValueError):
        return "th"
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _ordinal(n):
    """Format a number as an ordinal string: 92 -> '92nd', 11 -> '11th'."""
    try:
        i = int(round(float(n)))
    except (TypeError, ValueError):
        return str(n)
    return f"{i}{_ord_suffix(i)}"


# ── Squad Role (share-of-season minutes) ─────────────────────────────────────
# Classified by share of the season rather than absolute minutes, so it's fair
# across leagues that play a different number of games (33–37).
_SQUAD_ROLE_STARTER = 0.60   # >= 60% of the season's minutes
_SQUAD_ROLE_ROTATION = 0.30  # >= 30% (otherwise Depth)


def _league_season_minutes(df):
    """Approx minutes available per league (games × 90), inferred from the most
    minutes any player logged in that league (an ever-present ≈ games × 90)."""
    out = {}
    if "league_display" not in df.columns or "Time Played" not in df.columns:
        return out
    for lg, grp in df.groupby("league_display"):
        games = max(1, round((grp["Time Played"].max() or 0) / 90))
        out[lg] = games * 90
    return out


def _squad_role_label(minutes, season_minutes):
    """Starter / Rotation / Depth from a player's share of the season."""
    if not season_minutes:
        return "Depth"
    share = (minutes or 0) / season_minutes
    if share >= _SQUAD_ROLE_STARTER:
        return "Starter"
    if share >= _SQUAD_ROLE_ROTATION:
        return "Rotation"
    return "Depth"


# ── Chart Builders ───────────────────────────────────────────────────────────

def chart_bar(data, x, y, title, color=None, orientation="v", height=520):
    kw = dict(color_discrete_sequence=CHART_COLORS, height=height, title=title)
    if orientation == "h":
        fig = px.bar(data, x=y, y=x, color=color, orientation="h", **kw)
    else:
        fig = px.bar(data, x=x, y=y, color=color, **kw)
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45 if orientation == "v" else 0)
    return fig


def chart_scatter(data, x, y, title, color=None, size=None, height=520):
    fig = px.scatter(
        data, x=x, y=y, color=color, size=size, title=title,
        hover_name="nombre" if "nombre" in data.columns else None,
        color_discrete_sequence=CHART_COLORS, height=height,
    )
    fig.update_layout(template="plotly_white")
    return fig


def chart_radar(df_full, player_names, metrics, title="Player Comparison"):
    """Radar chart with values normalized to 0-100 vs selected-players max."""
    # Collect selected player rows first to compute per-metric max among them
    selected_rows = []
    for name in player_names:
        rows = df_full[df_full["nombre"].str.contains(re.escape(name), case=False, na=False)]
        if not rows.empty:
            selected_rows.append(rows.iloc[0])

    if not selected_rows:
        return go.Figure()

    # Per-metric max among selected players (floor at a small number to avoid /0)
    sel_max = {}
    for m in metrics:
        sel_max[m] = max((0 if pd.isna(r.get(m, 0)) else (r.get(m, 0) or 0)) for r in selected_rows) or 1

    fig = go.Figure()
    for row in selected_rows:
        vals = []
        raw_vals = []
        for m in metrics:
            v = row.get(m, 0)
            v = 0 if pd.isna(v) else v
            raw_vals.append(round(v, 2))
            vals.append(round(v / sel_max[m] * 100, 1) if sel_max[m] > 0 else 0)
        vals.append(vals[0])
        raw_vals.append(raw_vals[0])
        label = f"{row['nombre']} ({row.get('league_display', '')})"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]],
            fill="toself", name=label, opacity=0.6,
            customdata=raw_vals,
            hoveron="points", marker=dict(size=6),
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra>" + label + "</extra>",
        ))
    _radar_h = max(560, 420 + len(metrics) * 20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 115], dtick=20)),
        title=title, height=_radar_h, template="plotly_white", showlegend=True,
        hovermode="closest",
    )
    return fig


def chart_pie(data, names, values, title, height=500):
    fig = px.pie(data, names=names, values=values, title=title,
                 color_discrete_sequence=CHART_COLORS, height=height)
    fig.update_layout(template="plotly_white")
    return fig


# ── Photo helper ─────────────────────────────────────────────────────────────

# Opta stores full legal club names ("FC Bayern München", "FC Internazionale
# Milano"); TheSportsDB uses common names ("Bayern Munich", "Inter Milan").  A
# plain substring match fails for these, so canonicalize the ones that differ
# enough to break it — otherwise the team filter can't disambiguate same-named
# players and a wrong/stale face slips through.
_TEAM_CANON = {
    "bayern munchen": "bayern munich",
    "internazionale milano": "inter milan",
    "rasenballsport": "rb leipzig",
    "borussia 09 dortmund": "borussia dortmund",
    "vfl monchengladbach": "borussia monchengladbach",
    "atalanta bergamasca": "atalanta",
    "1. fc koln": "fc koln",
    "real club celta": "celta",
    "real sociedad": "real sociedad",
    "reial club deportiu espanyol": "espanyol",
    "real club deportivo mallorca": "mallorca",
}


def _canon_team(team):
    """Normalize a club name (accent-fold, strip FC) and apply the alias map so
    Opta and TheSportsDB names match."""
    t = _normalize_name(team).replace(" fc", "").replace("fc ", "").strip()
    for k, v in _TEAM_CANON.items():
        if k in t:
            return v
    return t


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_player_photo(player_name, team=None, ambiguous=False):
    """Return a player photo URL, preferring manual CSV overrides over TheSportsDB.

    The override CSV is keyed by name only, so for a name shared by more than
    one player (e.g. 'A. Onana' = André/Amadou, 'Ederson' = Man City GK/Atalanta)
    it would return the wrong (usually more famous) face. When *ambiguous* is set
    we skip the name-only override and require a team-matched live result —
    falling back to the initials avatar rather than showing the wrong player.
    """
    # 1. Manual override CSV — trusted only for unambiguous names.
    if not ambiguous:
        _photo_overrides = _load_photos_csv(_bust=_csv_mtime(_PHOTOS_CSV))
        if player_name in _photo_overrides:
            return _photo_overrides[player_name]
    # 2. TheSportsDB, team-disambiguated.
    try:
        q = urllib.parse.quote(player_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        players = [p for p in (data.get("player") or [])
                   if (p.get("strSport") or "").lower() == "soccer"]
        if players:
            def _photo(p):
                return p.get("strCutout") or p.get("strThumb") or p.get("strRender")
            # If team hint provided, only accept a player on that team.
            if team:
                team_lower = _canon_team(team)
                for p in players:
                    p_team = _canon_team(p.get("strTeam") or "")
                    if team_lower and p_team and (team_lower in p_team or p_team in team_lower):
                        photo = _photo(p)
                        if photo:
                            return photo
            # No confident match: only trust a unique result, and never for an
            # ambiguous name — guessing players[0] returns the wrong face
            # (e.g. every "Rodríguez"). Return None → initials avatar instead.
            if len(players) == 1 and not ambiguous:
                return _photo(players[0])
    except Exception:
        pass
    return None


# ── Market Value & Salary helpers ────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def _resolve_full_name(short_name, team=None):
    """Resolve abbrev. name (e.g. 'E. Haaland') to full name via TheSportsDB."""
    try:
        q = urllib.parse.quote(short_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        players = data.get("player")
        if not players:
            return short_name
        # If team hint provided, prefer match
        if team:
            team_lower = team.lower().replace(" fc", "").replace("fc ", "").strip()
            for p in players:
                p_team = (p.get("strTeam") or "").lower().replace(" fc", "").replace("fc ", "").strip()
                if (p.get("strSport") or "").lower() == "soccer" and (
                    team_lower in p_team or p_team in team_lower
                ):
                    return p.get("strPlayer") or short_name
        # Return first soccer player
        for p in players:
            if (p.get("strSport") or "").lower() == "soccer":
                return p.get("strPlayer") or short_name
        return players[0].get("strPlayer") or short_name
    except Exception:
        return short_name


_TM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_transfermarkt_value(player_name, team=None):
    """Fetch latest market value from Transfermarkt (€)."""
    try:
        query = urllib.parse.quote(player_name)
        url = (f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/"
               f"schnellsuche?query={query}")
        resp = requests.get(url, headers=_TM_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table", class_="items")
        if not tables:
            return None
        rows = tables[0].find_all("tr", class_=["odd", "even"])
        best = None
        for row in rows:
            value_cell = row.find("td", class_=lambda c: c and "rechts" in c and "hauptlink" in c)
            if not value_cell:
                continue
            mv = value_cell.get_text(strip=True)
            if not mv or mv == "-":
                continue
            # Try team match
            if team:
                cells = row.find_all("td")
                row_text = " ".join(c.get_text(strip=True).lower() for c in cells)
                team_norm = _normalize_name(team)
                if team_norm and team_norm in row_text:
                    return mv
            if best is None:
                best = mv
        return best
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_capology_search_index():
    """Download Capology player search index (cached 24 h)."""
    try:
        resp = requests.get(
            "https://www.capology.com/static/files/search_players.json",
            headers=_TM_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def _capology_find_slug(player_name, team=None):
    """Find the Capology player slug from the search index."""
    index = _fetch_capology_search_index()
    if not index:
        return None
    target = _normalize_name(player_name)
    candidates = []
    for entry in index:
        name = _normalize_name(entry.get("name", ""))
        if name == target:
            candidates.append(entry)
    if not candidates:
        # Fallback: match by last name + first-initial when name has "X. Surname" pattern
        parts = target.split()
        if len(parts) >= 2:
            surname = parts[-1]
            initial = parts[0].rstrip(".")
            for entry in index:
                name = _normalize_name(entry.get("name", ""))
                name_parts = name.split()
                if len(name_parts) >= 2 and name_parts[-1] == surname:
                    if name_parts[0].startswith(initial):
                        candidates.append(entry)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].get("link")
    # Disambiguate by team logo URL (contains team slug)
    if team:
        team_norm = _normalize_name(team)
        for c in candidates:
            club_url = (c.get("club") or "").lower()
            if team_norm.replace(" ", "-") in club_url or team_norm.replace(" ", "") in club_url:
                return c.get("link")
    return candidates[0].get("link")


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_capology_salary(player_name, team=None):
    """Fetch latest gross annual salary from Capology (€)."""
    try:
        slug = _capology_find_slug(player_name, team)
        if not slug:
            return None
        url = f"https://www.capology.com{slug}/"
        resp = requests.get(url, headers=_TM_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        # Extract annual_gross_eur + bonus_gross_eur from the embedded JS data_active
        m_base = re.search(r'"annual_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
        if m_base:
            raw = int(m_base.group(1))
            m_bonus = re.search(r'"bonus_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
            if m_bonus:
                raw += int(m_bonus.group(1))
            if raw >= 1_000_000:
                return f"€{raw / 1_000_000:,.2f}M"
            elif raw >= 1_000:
                return f"€{raw / 1_000:,.0f}K"
            else:
                return f"€{raw:,}"
        return None
    except Exception:
        return None


# ── Pizza Chart ──────────────────────────────────────────────────────────────

PIZZA_METRICS = {
    "Defending": [
        ("Tackles", "Total Tackles"),
        ("Tackles Won", "Tackles Won"),
        ("Interceptions", "Interceptions"),
        ("Recoveries", "Recoveries"),
        ("Clearances", "Total Clearances"),
        ("Blocks", "Blocks"),
    ],
    "Aerial": [
        ("Aerial Duels", "Aerial Duels"),
        ("Aerial Won", "Aerial Duels won"),
        ("Aerial Win %", "Aerial Win %"),
    ],
    "Passing": [
        ("Passes", "Total Passes"),
        ("Long Passes", "Successful Long Passes"),
        ("Fwd Passes", "Forward Passes"),
        ("Key Passes", "Key Passes (Attempt Assists)"),
        ("xA", "xA"),
        ("Through Balls", "Through balls"),
        ("Big Chances Created", "Total Big Chances Created"),
    ],
    "Progression": [
        ("Prog. Carries", "Progressive Carries"),
        ("Carries", "Carries"),
        ("Final 3rd Touch", "Final Third Touches"),
        ("Fwd Passes", "Forward Passes"),
    ],
    "Possession": [
        ("Dribbles", "Successful Dribbles"),
        ("Prog. Carries", "Progressive Carries"),
        ("Ground Duels", "Ground Duels won"),
        ("Duels Won", "Duels won"),
    ],
    "Attacking": [
        ("Goals", "Goals"),
        ("npxG", "npxG"),
        ("Finishing", "npG-xG (shrunk)"),
        ("Assists", "Goal Assists"),
        ("Shots on Target", "Shots On Target ( inc goals )"),
        ("Big Chances", "Total Big Chances Scored"),
    ],
}

# NOTE: These categories drive BOTH the GK pizza chart and the GK overall grade
# in the Player Profile.  Only reliable, high-sample metrics belong here.
# Penalty Save % (tiny samples — a handful of penalties), Caught % (catch-vs-parry
# style, small samples) and Claim % (clusters at 100% so it ranks poorly despite
# being "perfect") are intentionally EXCLUDED from grading — they are shown as
# informational stats in the Detailed Statistics tabs and the PSxG statline.
GK_PIZZA_METRICS = {
    "Shot-Stopping": [
        # Core: goals prevented vs a real post-shot xG model (build_gk_psxg.py),
        # sample-shrunk so partial seasons don't swing the grade.  Weighted as the
        # dominant metric in the grade via GK_SHOTSTOP_GRADE_WEIGHTS.
        ("PSxG+/-", "PSxG+/- (shrunk)"),
        ("Saves/90", "Saves/90"),          # rate — fair for GKs on dominant teams
        ("Save %", "Save %"),
        ("Big Chances Saved", "Total Big Chances Saved"),
        ("Clean Sheet %", "Clean Sheet %"),
    ],
    "Distribution": [
        ("GK Distribution", "GK Successful Distribution"),
        ("Launches", "Successful Launches"),
        ("Launch %", "Launch %"),
    ],
    "Command": [
        ("Catches", "Catches"),
        ("Punches", "Punches"),
    ],
    "Sweeping": [
        ("Recoveries", "Recoveries"),
        ("Clearances", "Total Clearances"),
        ("Interceptions", "Interceptions"),
    ],
}

# PSxG+/- is the primary shot-stopping signal: weight it 4× the supporting
# metrics so it accounts for ~half the Shot-Stopping category grade.  Applied in
# the GK grade path only (the pizza chart still shows each slice equally).
GK_SHOTSTOP_GRADE_WEIGHTS = {"PSxG+/- (shrunk)": 4.0}

PIZZA_CATEGORY_COLORS = {
    "Defending": "#457b9d",
    "Aerial": "#f4a261",
    "Passing": "#2a9d8f",
    "Progression": "#264653",
    "Possession": "#e9c46a",
    "Attacking": "#e63946",
    "Shot-Stopping": "#e63946",
    "Distribution": "#2a9d8f",
    "Command": "#f4a261",
    "Sweeping": "#457b9d",
}


def _build_pizza_chart(player_row, df_peers, player_name, position, is_gk=False):
    """Nightingale rose (pizza) chart showing per-metric percentiles."""
    labels, values, colors, category_labels = [], [], [], []

    # GKs must be compared only against other GKs — not outfield players.
    # Without this filter, GK-specific stats (Saves, etc.) would rank near
    # 100th percentile because every outfield player has 0 saves, producing
    # inflated pizza numbers that don't reflect true GK quality.
    if is_gk and "posicion" in df_peers.columns:
        df_peers = df_peers[df_peers["posicion"] == "Goalkeeper"]

    pizza_src = GK_PIZZA_METRICS if is_gk else PIZZA_METRICS
    for cat, metric_list in pizza_src.items():
        color = PIZZA_CATEGORY_COLORS.get(cat, "#999")
        for display_name, col_name in metric_list:
            if col_name not in df_peers.columns:
                continue
            val = player_row.get(col_name, 0)
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
            peer_vals = df_peers[col_name].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            labels.append(display_name)
            values.append(round(pct, 1))
            colors.append(color)
            category_labels.append(cat)

    if not labels:
        return None

    n = len(labels)
    slice_angle = 360 / n
    theta_centers = [i * slice_angle for i in range(n)]

    fig = go.Figure()

    cats_seen = set()
    for i, (lbl, val, clr, cat) in enumerate(zip(labels, values, colors, category_labels)):
        show_legend = cat not in cats_seen
        cats_seen.add(cat)
        fig.add_trace(go.Barpolar(
            r=[val], theta=[theta_centers[i]], width=[slice_angle - 1],
            marker=dict(color=clr, opacity=0.85, line=dict(color="#1a1a2e", width=1.5)),
            name=cat if show_legend else None,
            legendgroup=cat, showlegend=show_legend,
            hovertemplate=f"<b>{lbl}</b><br>{cat}<br>Percentile: {val:.0f}<extra></extra>",
        ))

    text_r = [max(v, 10) + 10 for v in values]
    fig.add_trace(go.Scatterpolar(
        r=text_r, theta=theta_centers, mode="text",
        text=[f"<b>{v:.0f}</b>" for v in values],
        textfont=dict(size=11, color="white"), showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 115], showticklabels=False, showline=False,
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickvals=theta_centers, ticktext=labels,
                             tickfont=dict(size=9, color="#ccc"),
                             gridcolor="rgba(255,255,255,0.05)", direction="clockwise"),
            bgcolor="#1a1a2e",
        ),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font=dict(color="#eee"),
        title=dict(
            text=f"<b>{player_name}</b> vs {position}s<br>"
                 f"<span style='font-size:12px;color:#aaa'>Percentile rankings · 0-100 scale · 50 = average</span>",
            font=dict(size=16, color="#f4a261"),
        ),
        height=650, showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                    font=dict(size=12, color="#eee")),
        margin=dict(t=80, b=60, l=80, r=80),
    )
    return fig


# ── FBref-style Scouting Report ──────────────────────────────────────────────

FBREF_TEMPLATES = {
    "Wingers": {
        "Goals": ["Goals"],
        "Non-Penalty Goals": ["Non-Penalty Goals"],
        "Assists": ["Goal Assists"],
        "Key Passes": ["Key Passes (Attempt Assists)"],
        "Dribble %": ["Dribble %"],
        "Dribbles Completed": ["Successful Dribbles"],
        "Progressive Carries": ["Progressive Carries"],
        "Cross %": ["Cross %"],
        "Crosses Completed": ["Successful Crosses & Corners"],
        "Shots": ["Total Shots"],
        "Shots on Target": ["Shots On Target ( inc goals )"],
        "Touches in Box": ["Total Touches In Opposition Box"],
        "Tackle Win %": ["Tackle Win %"],
    },
    "Attacking Midfield": {
        "Goals": ["Goals"],
        "Non-Penalty Goals": ["Non-Penalty Goals"],
        "Assists": ["Goal Assists"],
        "Key Passes": ["Key Passes (Attempt Assists)"],
        "Through Balls": ["Through balls"],
        "Dribble %": ["Dribble %"],
        "Dribbles Completed": ["Successful Dribbles"],
        "Progressive Carries": ["Progressive Carries"],
        "Shots": ["Total Shots"],
        "Touches in Box": ["Total Touches In Opposition Box"],
        "Pass %": ["Pass %"],
        "Tackle Win %": ["Tackle Win %"],
    },
    "Striker": {
        "Non-Penalty Goals": ["Non-Penalty Goals"],
        "Goals": ["Goals"],
        "Shots": ["Total Shots"],
        "Shots on Target": ["Shots On Target ( inc goals )"],
        "Assists": ["Goal Assists"],
        "Key Passes": ["Key Passes (Attempt Assists)"],
        "Touches in Box": ["Total Touches In Opposition Box"],
        "Aerial Win %": ["Aerial Win %"],
        "Dribble %": ["Dribble %"],
        "Dribbles Completed": ["Successful Dribbles"],
        "Progressive Carries": ["Progressive Carries"],
    },
    "Central Midfield": {
        "Goals": ["Goals"],
        "Assists": ["Goal Assists"],
        "Key Passes": ["Key Passes (Attempt Assists)"],
        "Pass %": ["Pass %"],
        "Passes": ["Total Passes"],
        "Long Pass %": ["Long Pass %"],
        "Long Passes": ["Successful Long Passes"],
        "Progressive Carries": ["Progressive Carries"],
        "Tackle Win %": ["Tackle Win %"],
        "Tackles": ["Total Tackles"],
        "Interceptions": ["Interceptions"],
        "Recoveries": ["Recoveries"],
        "Dribble %": ["Dribble %"],
        "Ground Duel %": ["Ground Duel %"],
    },
    "Centre-Back": {
        "Passes": ["Total Passes"],
        "Long Pass %": ["Long Pass %"],
        "Long Passes": ["Successful Long Passes"],
        "Tackle Win %": ["Tackle Win %"],
        "Tackles": ["Total Tackles"],
        "Interceptions": ["Interceptions"],
        "Clearances": ["Total Clearances"],
        "Blocks": ["Blocks"],
        "Aerial Win %": ["Aerial Win %"],
        "Aerial Duels Won": ["Aerial Duels won"],
        "Recoveries": ["Recoveries"],
        "Duel %": ["Duel %"],
    },
    "Full-Back": {
        "Goals": ["Goals"],
        "Assists": ["Goal Assists"],
        "Key Passes": ["Key Passes (Attempt Assists)"],
        "Cross %": ["Cross %"],
        "Crosses": ["Successful Crosses & Corners"],
        "Dribble %": ["Dribble %"],
        "Progressive Carries": ["Progressive Carries"],
        "Tackle Win %": ["Tackle Win %"],
        "Tackles": ["Total Tackles"],
        "Interceptions": ["Interceptions"],
        "Aerial Win %": ["Aerial Win %"],
        "Recoveries": ["Recoveries"],
    },
    "Goalkeeper": {
        "Save %": ["Save %"],
        "Saves": ["Saves Made"],
        "Goals Prevented": ["Goals Prevented"],
        "Catches": ["Catches"],
        "Punches": ["Punches"],
        "Launch %": ["Launch %"],
        "GK Distribution": ["GK Successful Distribution"],
        "Big Chances Saved": ["Total Big Chances Saved"],
    },
}


def _build_fbref_bar_chart(player_row, df_peers, player_name, position, selected_metrics=None):
    """Horizontal bar chart -- FBref-style percentile scouting report."""
    template = FBREF_TEMPLATES.get(position, FBREF_TEMPLATES["Central Midfield"])
    if selected_metrics is not None:
        template = {k: v for k, v in template.items() if k in selected_metrics}
    labels, pct_values, bar_colors = [], [], []

    for display_name, cols in template.items():
        avail = [c for c in cols if c in df_peers.columns]
        if not avail:
            continue
        player_val = sum(player_row.get(c, 0) or 0 for c in avail)
        peer_vals = df_peers[avail].fillna(0).sum(axis=1)
        pct = round((peer_vals < player_val).sum() / max(len(peer_vals), 1) * 100, 1)
        labels.append(display_name)
        pct_values.append(pct)
        bar_colors.append("#2d6a4f" if pct >= 66 else "#e9c46a" if pct >= 33 else "#e63946")

    if not labels:
        return None

    labels.reverse()
    pct_values.reverse()
    bar_colors.reverse()

    fig = go.Figure(go.Bar(
        y=labels, x=pct_values, orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#1a1a2e", width=1)),
        text=[f"{v:.0f}" for v in pct_values], textposition="outside",
        textfont=dict(color="#eee", size=12),
        hovertemplate="<b>%{y}</b><br>Percentile: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b> -- Scouting Report vs {position}s<br>"
                 f"<span style='font-size:12px;color:#aaa'>Percentile 0-100 compared to same position</span>",
            font=dict(size=15, color="#f4a261"),
        ),
        xaxis=dict(range=[0, 110], title="Percentile", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(tickfont=dict(size=12, color="#ccc")),
        height=max(380, 36 * len(labels)),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"), margin=dict(l=160, r=40, t=80, b=40),
    )
    return fig


# ── Scatter Plot ─────────────────────────────────────────────────────────────

SCATTER_METRIC_OPTIONS = [
    "Goals", "Non-Penalty Goals", "Goal Assists", "Total Shots",
    "Shots On Target ( inc goals )", "Shots Created",
    "Key Passes (Attempt Assists)",
    "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
    "Successful Long Passes", "Forward Passes",
    "Through balls", "Successful Crosses & Corners",
    "Successful Dribbles", "Progressive Carries", "Carries",
    "Total Tackles", "Tackles Won", "Interceptions", "Total Clearances", "Recoveries",
    "Aerial Duels won", "Aerial Win %", "Ground Duels won", "Duels won",
    "Total Touches In Opposition Box", "Final Third Touches",
    "Total Big Chances Scored", "Total Big Chances Created",
    "Blocked Shots", "Blocks",
    "Total Fouls Won", "Total Fouls Conceded",
]

GK_SCATTER_METRIC_OPTIONS = [
    "Saves Made", "Goals Prevented", "Save %",
    "Catches", "Punches", "Total Big Chances Saved",
    "GK Successful Distribution", "Successful Launches",
    "Launch %", "Recoveries", "Total Clearances", "Interceptions",
    "Clean Sheets", "Penalties Saved",
]

SCATTER_DEFAULTS = {
    "Wingers": ("Successful Dribbles", "Goal Assists"),
    "Attacking Midfield": ("Key Passes (Attempt Assists)", "Goals"),
    "Striker":             ("Total Shots", "Goals"),
    "Central Midfield":    ("Total Passes", "Total Tackles"),
    "Centre-Back":         ("Total Tackles", "Aerial Duels won"),
    "Full-Back":           ("Successful Crosses & Corners", "Total Tackles"),
    "Goalkeeper":          ("Saves Made", "Goals Prevented"),
}


def _build_scatter_plot(player_row, df_peers, player_name, position, x_col, y_col, scope_label="Across Europe"):
    """Scatter plot with the player highlighted among same-position peers."""
    if x_col not in df_peers.columns or y_col not in df_peers.columns:
        return None

    pos_peers = df_peers[df_peers["posicion"] == position].copy()
    if pos_peers.empty:
        return None

    pos_peers = pos_peers[(pos_peers[x_col].notna()) & (pos_peers[y_col].notna())].copy()
    if pos_peers.empty:
        return None

    pos_peers["is_selected"] = pos_peers["nombre"] == player_name

    def _iqr_outliers(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return series > (q3 + 1.5 * iqr)

    pos_peers["_outlier"] = _iqr_outliers(pos_peers[x_col]) | _iqr_outliers(pos_peers[y_col])

    fig = go.Figure()

    # Regular peers
    others = pos_peers[~pos_peers["is_selected"] & ~pos_peers["_outlier"]]
    fig.add_trace(go.Scatter(
        x=others[x_col], y=others[y_col], mode="markers",
        marker=dict(size=7, color="#555", opacity=0.45),
        text=others["nombre"] + " (" + others["league_display"] + ")",
        hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
        name=f"Other {position}s", showlegend=True,
    ))

    # Outliers (labeled)
    outliers = pos_peers[~pos_peers["is_selected"] & pos_peers["_outlier"]]
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers[x_col], y=outliers[y_col], mode="markers+text",
            marker=dict(size=9, color="#e9c46a", opacity=0.8, line=dict(width=1, color="#f4a261")),
            text=outliers["nombre"], textposition="top center",
            textfont=dict(size=9, color="#e9c46a"),
            hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
            name="Outliers", showlegend=True,
        ))

    # Selected player
    selected = pos_peers[pos_peers["is_selected"]]
    if not selected.empty:
        fig.add_trace(go.Scatter(
            x=selected[x_col], y=selected[y_col], mode="markers+text",
            marker=dict(size=16, color="#2d6a4f", symbol="circle", line=dict(width=2.5, color="white")),
            text=selected["nombre"], textposition="top center",
            textfont=dict(size=13, color="#52b788", family="Arial Black"),
            hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
            name=player_name, showlegend=True,
        ))

    # Average quadrant lines
    avg_x, avg_y = pos_peers[x_col].mean(), pos_peers[y_col].mean()
    fig.add_vline(x=avg_x, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text=f"Avg {x_col}", annotation_font_color="#888", annotation_font_size=10)
    fig.add_hline(y=avg_y, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text=f"Avg {y_col}", annotation_font_color="#888", annotation_font_size=10)

    fig.update_layout(
        title=dict(text=f"<b>{player_name}</b> vs All {position}s {scope_label}",
                   font=dict(size=16, color="#f4a261")),
        xaxis_title=x_col, yaxis_title=y_col, height=560,
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font=dict(color="#eee"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        legend=dict(font=dict(color="#eee")),
    )
    return fig


# ── Profile Helpers ──────────────────────────────────────────────────────────

PROFILE_CATEGORIES = {
    "Offensive": ["Goals", "Goal Assists", "Total Shots", "Shots On Target ( inc goals )",
                  "Key Passes (Attempt Assists)", "Total Touches In Opposition Box"],
    "Defensive": ["Total Tackles", "Tackles Won", "Tackle Win %",
                  "Interceptions", "Total Clearances",
                  "Blocked Shots", "Recoveries"],
    "Passing": ["Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
                "Pass %", "Successful Long Passes", "Long Pass %",
                "Forward Passes"],
    "Ball Progression": ["Progressive Carries", "Carries",
                         "Through balls", "Final Third Touches"],
    "Possession": ["Successful Dribbles", "Dribble %",
                   "Duels won", "Duel %",
                   "Ground Duels won", "Ground Duel %"],
    "Aerial": ["Aerial Duels won", "Aerial Duels", "Aerial Win %"],
    "Discipline": ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"],
}

GK_PROFILE_CATEGORIES = {
    "Shot-Stopping": ["PSxG+/- (shrunk)", "Save %", "Saves Made",
                      "Total Big Chances Saved"],
    "Command": ["Catches", "Punches", "Penalties Saved"],
    "Distribution": ["GK Successful Distribution", "Successful Launches",
                     "Launch %"],
    "Sweeping": ["Recoveries", "Total Clearances", "Interceptions"],
}

STRIKER_PROFILE_CATEGORIES = {
    "Finishing": ["Goals", "Non-Penalty Goals", "npxG", "npG-xG (shrunk)",
                  "Total Shots",
                  "Shots On Target ( inc goals )", "Total Touches In Opposition Box",
                  "Total Big Chances Scored"],
    "Chance Creation": ["Goal Assists", "xA", "Key Passes (Attempt Assists)",
                        "Total Big Chances Created"],
    "Aerial": ["Aerial Duels won", "Aerial Duels", "Aerial Win %"],
    "Possession": ["Successful Dribbles", "Dribble %",
                   "Progressive Carries", "Total Fouls Won"],
    "Pressing": ["Recoveries", "Interceptions"],
    "Discipline": ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"],
}

WINGER_PROFILE_CATEGORIES = {
    "Offensive": ["Goals", "Non-Penalty Goals", "Total Shots",
                  "Shots On Target ( inc goals )",
                  "Total Touches In Opposition Box"],
    "Chance Creation": ["Goal Assists", "xA", "Key Passes (Attempt Assists)",
                        "Successful Crosses & Corners", "Cross %",
                        "Total Big Chances Created"],
    "Dribbling & Carrying": ["Successful Dribbles", "Dribble %",
                             "Progressive Carries", "Carries"],
    "Passing": ["Through balls", "Final Third Touches",
                "Forward Passes"],
    "Pressing": ["Recoveries", "Total Tackles", "Tackle Win %",
                 "Interceptions"],
    "Discipline": ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"],
}

AM_PROFILE_CATEGORIES = {
    "Offensive": ["Goals", "Non-Penalty Goals", "Total Shots",
                  "Shots On Target ( inc goals )",
                  "Total Touches In Opposition Box",
                  "Total Big Chances Scored"],
    "Chance Creation": ["Goal Assists", "xA", "Key Passes (Attempt Assists)",
                        "Through balls", "Total Big Chances Created"],
    "Dribbling & Carrying": ["Successful Dribbles", "Dribble %",
                             "Progressive Carries", "Carries"],
    "Passing": ["Total Passes", "Pass %", "Forward Passes",
                "Successful Long Passes", "Long Pass %",
                "Final Third Touches"],
    "Pressing & Defence": ["Recoveries", "Total Tackles", "Tackle Win %",
                           "Interceptions", "Ground Duels won", "Ground Duel %"],
    "Discipline": ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"],
}

# ── Attribute-based grade categories (used for Player Profile header grades) ─
ATTRIBUTE_GRADE_CATEGORIES = {
    "Attacking": ["Goals", "Non-Penalty Goals", "npxG", "npG-xG (shrunk)",
                  "Goal Assists", "xA",
                  "Total Shots",
                  "Shots On Target ( inc goals )", "Goals from Inside Box",
                  "Total Touches In Opposition Box",
                  "Total Big Chances Scored", "Shots Created",
                  "Total Fouls Won"],
    "Defending": ["Total Tackles", "Tackles Won", "Tackle Win %",
                  "Interceptions", "Total Clearances", "Recoveries",
                  "Blocked Shots", "Blocks",
                  "Aerial Duels won", "Aerial Win %",
                  "Ground Duels won", "Ground Duel %",
                  "Duels won", "Duel %"],
    "Passing": ["Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
                "Pass %", "Key Passes (Attempt Assists)", "Goal Assists",
                "Total Big Chances Created",
                "Successful Long Passes", "Long Pass %",
                "Successful Crosses & Corners", "Cross %",
                "Forward Passes", "Short Pass %"],
    "Dribbling & Carrying": ["Successful Dribbles", "Dribble %",
                              "Unsuccessful Dribbles",
                              "Progressive Carries", "Carries", "Overruns"],
    "Ball Progression": ["Progressive Carries", "Carries",
                         "Through balls", "Final Third Touches",
                         "Forward Passes"],
    # Pass % lives in "Passing" above — kept out here to avoid double-counting.
    "Passing Safety": ["Retention %", "Short Pass %",
                       "Long Pass %", "Own Half Pass %"],
}

# Categories where LOWER values are better (inverted percentile)
_INVERTED_GRADE_CATS = {"Discipline"}

# ── Position-specific category weights for the overall grade ─────────────────
# Weights reflect what matters most for each position.
# Keys must match ATTRIBUTE_GRADE_CATEGORIES (outfield) or GK categories.
# Position weights are used as fallback when no role is known.
_POSITION_GRADE_WEIGHTS = {
    "Striker": {
        "Attacking": 0.30, "Defending": 0.05, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.05,
    },
    "Wingers": {
        "Attacking": 0.25, "Defending": 0.05, "Passing": 0.20,
        "Dribbling & Carrying": 0.20, "Ball Progression": 0.15,
        "Passing Safety": 0.05,
    },
    "Attacking Midfield": {
        "Attacking": 0.20, "Defending": 0.05, "Passing": 0.25,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
    "Central Midfield": {
        "Attacking": 0.10, "Defending": 0.20, "Passing": 0.25,
        "Dribbling & Carrying": 0.10, "Ball Progression": 0.10,
        "Passing Safety": 0.15,
    },
    "Centre-Back": {
        "Attacking": 0.05, "Defending": 0.40, "Passing": 0.15,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.10,
        "Passing Safety": 0.15,
    },
    "Full-Back": {
        "Attacking": 0.10, "Defending": 0.25, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
}

# ── Role-specific weight overrides (take priority over position weights) ─────
_ROLE_GRADE_WEIGHTS = {
    # --- Striker roles ---
    "Prolific Striker": {
        "Attacking": 0.40, "Defending": 0.02, "Passing": 0.10,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.10,
        "Passing Safety": 0.03,
    },
    "Target Man": {
        "Attacking": 0.35, "Defending": 0.10, "Passing": 0.10,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.10,
        "Passing Safety": 0.05,
    },
    "False 9": {
        "Attacking": 0.20, "Defending": 0.03, "Passing": 0.25,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.10,
    },
    "Pressing Forward": {
        "Attacking": 0.30, "Defending": 0.15, "Passing": 0.10,
        "Dribbling & Carrying": 0.10, "Ball Progression": 0.10,
        "Passing Safety": 0.05,
    },
    # --- Winger roles ---
    "Inverted Winger": {
        "Attacking": 0.35, "Defending": 0.03, "Passing": 0.10,
        "Dribbling & Carrying": 0.25, "Ball Progression": 0.15,
        "Passing Safety": 0.02,
    },
    "Classic Winger": {
        "Attacking": 0.15, "Defending": 0.05, "Passing": 0.25,
        "Dribbling & Carrying": 0.20, "Ball Progression": 0.20,
        "Passing Safety": 0.05,
    },
    "Creative Winger": {
        "Attacking": 0.15, "Defending": 0.03, "Passing": 0.30,
        "Dribbling & Carrying": 0.20, "Ball Progression": 0.15,
        "Passing Safety": 0.07,
    },
    "Pressing Winger": {
        "Attacking": 0.25, "Defending": 0.15, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.05,
    },
    # --- CAM roles ---
    "Classic 10": {
        "Attacking": 0.12, "Defending": 0.03, "Passing": 0.33,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.22,
        "Passing Safety": 0.15,
    },
    "Advanced Playmaker": {
        "Attacking": 0.28, "Defending": 0.03, "Passing": 0.22,
        "Dribbling & Carrying": 0.22, "Ball Progression": 0.18,
        "Passing Safety": 0.07,
    },
    "Shadow Striker": {
        "Attacking": 0.35, "Defending": 0.05, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.05,
    },
    "Trequartista": {
        "Attacking": 0.20, "Defending": 0.02, "Passing": 0.20,
        "Dribbling & Carrying": 0.25, "Ball Progression": 0.20,
        "Passing Safety": 0.08,
    },
    "Mezzala": {
        "Attacking": 0.25, "Defending": 0.10, "Passing": 0.20,
        "Dribbling & Carrying": 0.20, "Ball Progression": 0.20,
        "Passing Safety": 0.05,
    },
    # --- Central Midfield roles ---
    "Defensive Midfielder": {
        "Attacking": 0.03, "Defending": 0.40, "Passing": 0.15,
        "Dribbling & Carrying": 0.02, "Ball Progression": 0.05,
        "Passing Safety": 0.20,
    },
    "Box-to-Box Midfielder": {
        "Attacking": 0.15, "Defending": 0.20, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
    "Deep-Lying Playmaker": {
        "Attacking": 0.05, "Defending": 0.10, "Passing": 0.35,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.20,
        "Passing Safety": 0.20,
    },
    "Central Midfielder": {
        "Attacking": 0.12, "Defending": 0.18, "Passing": 0.20,
        "Dribbling & Carrying": 0.10, "Ball Progression": 0.15,
        "Passing Safety": 0.15,
    },
    "Ball Winning Midfielder": {
        "Attacking": 0.05, "Defending": 0.45, "Passing": 0.10,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    # --- Centre-Back roles ---
    "Ball Playing": {
        "Attacking": 0.05, "Defending": 0.25, "Passing": 0.25,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.15,
        "Passing Safety": 0.20,
    },
    "Duelist": {
        "Attacking": 0.02, "Defending": 0.50, "Passing": 0.10,
        "Dribbling & Carrying": 0.02, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    # --- Full-Back roles ---
    "Inverted": {
        "Attacking": 0.10, "Defending": 0.15, "Passing": 0.25,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.10,
    },
    "Defensive": {
        "Attacking": 0.05, "Defending": 0.40, "Passing": 0.15,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    "Attacking": {
        "Attacking": 0.20, "Defending": 0.10, "Passing": 0.25,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.05,
    },
    "All-Around": {
        "Attacking": 0.12, "Defending": 0.25, "Passing": 0.20,
        "Dribbling & Carrying": 0.12, "Ball Progression": 0.12,
        "Passing Safety": 0.10,
    },
    # --- Goalkeeper roles ---
    # Shot-stopping is the core job of every keeper, so it carries a floor of
    # 0.35 across all GK roles.  Each role still keeps its signature emphasis
    # (Sweeper → sweeping, Ball-Playing → distribution), but no role lets its
    # speciality outweigh shot-stopping.
    "Shot-Stopper": {
        "Shot-Stopping": 0.50, "Command": 0.20,
        "Distribution": 0.15, "Sweeping": 0.15,
    },
    "Sweeper Keeper": {
        "Shot-Stopping": 0.35, "Sweeping": 0.30,
        "Distribution": 0.20, "Command": 0.15,
    },
    "Ball-Playing Goalkeeper": {
        "Shot-Stopping": 0.35, "Distribution": 0.35,
        "Command": 0.15, "Sweeping": 0.15,
    },
}

# ── Role-specific KPI profiles (curated key metrics per role) ────────────────
# Each role maps category_name → (weight, [key_metrics]).
# Overall grade = weighted avg of KPI category percentiles.
# Ball Security now uses rate stats (higher = better) — NOT inverted.
_KPI_INVERTED_CATS = set()

# Bump this string whenever role names/definitions change to invalidate the
# 24-hour Player Lab cache immediately on redeployment.
_ROLE_SCHEMA_VERSION = "v14"  # Player Lab: Foot + Top Strength filters

_ROLE_KPI_PROFILES = {
    # --- Striker roles ---
    "Prolific Striker": {
        "Finishing":     (0.35, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )", "Total Big Chances Scored"]),
        "Shot Volume":   (0.20, ["Total Shots", "Total Big Chances Created"]),
        "Box Presence":  (0.25, ["Goals from Inside Box", "Total Touches In Opposition Box"]),
        "Link-Up":       (0.15, ["Goal Assists", "Key Passes (Attempt Assists)"]),
        "Ball Security": (0.05, ["Retention %", "Dribble %"]),
    },
    "Target Man": {
        "Finishing":     (0.25, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )"]),
        "Aerial Threat": (0.25, ["Aerial Duels won", "Aerial Win %"]),
        "Hold-Up Play":  (0.20, ["Total Fouls Won", "Key Passes (Attempt Assists)", "Goal Assists"]),
        "Box Presence":  (0.20, ["Goals from Inside Box", "Total Touches In Opposition Box"]),
        "Ball Security": (0.10, ["Retention %", "Dribble %"]),
    },
    "False 9": {
        "Creativity":      (0.30, ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"]),
        "Finishing":        (0.15, ["Goals", "Non-Penalty Goals"]),
        "Progression":     (0.25, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Passing Quality": (0.20, ["Pass %", "Total Successful Passes ( Excl Crosses & Corners ) "]),
        "Ball Security":   (0.10, ["Retention %", "Pass %", "Dribble %"]),
    },
    "Pressing Forward": {
        "Finishing":     (0.25, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )"]),
        "Pressing":      (0.25, ["Recoveries", "Tackles Won", "Interceptions"]),
        "Duels":         (0.20, ["Duels won", "Duel %", "Ground Duels won"]),
        "Link-Up":       (0.20, ["Goal Assists", "Key Passes (Attempt Assists)"]),
        "Ball Security": (0.10, ["Retention %", "Dribble %"]),
    },
    # --- Winger roles ---
    "Inverted Winger": {
        "Finishing":     (0.30, ["Non-Penalty Goals", "Goals", "Shots On Target ( inc goals )"]),
        "Dribbling":     (0.25, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Shot Creation": (0.20, ["Total Big Chances Created", "Key Passes (Attempt Assists)"]),
        "Creativity":    (0.20, ["Goal Assists", "Through balls"]),
        "Ball Security": (0.05, ["Retention %", "Dribble %"]),
    },
    "Classic Winger": {
        "Crossing":      (0.30, ["Successful Crosses & Corners", "Cross %"]),
        "Creativity":    (0.25, ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"]),
        "Dribbling":     (0.25, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Progression":   (0.15, ["Forward Passes", "Final Third Touches"]),
        "Ball Security": (0.05, ["Retention %", "Dribble %"]),
    },
    "Creative Winger": {
        "Creativity":      (0.30, ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"]),
        "Crossing":        (0.20, ["Successful Crosses & Corners", "Cross %"]),
        "Dribbling":       (0.20, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Passing Quality": (0.15, ["Pass %", "Forward Passes"]),
        "Ball Security":   (0.15, ["Retention %", "Pass %"]),
    },
    "Pressing Winger": {
        "Pressing":      (0.25, ["Recoveries", "Tackles Won", "Interceptions"]),
        "Attacking":     (0.25, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)"]),
        "Dribbling":     (0.20, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Duels":         (0.20, ["Duels won", "Duel %", "Ground Duels won"]),
        "Ball Security": (0.10, ["Retention %", "Dribble %"]),
    },
    # --- CAM roles ---
    "Classic 10": {
        "Creativity":      (0.35, ["Total Big Chances Created", "Key Passes (Attempt Assists)", "Goal Assists", "Through balls"]),
        "Passing Quality": (0.25, ["Pass %", "Forward Passes", "Final Third Touches"]),
        "Progression":     (0.20, ["Progressive Carries", "Through balls"]),
        "Ball Security":   (0.15, ["Retention %", "Pass %", "Dribble %"]),
        "Finishing":       (0.05, ["Non-Penalty Goals", "Shots On Target ( inc goals )"]),
    },
    "Advanced Playmaker": {
        "Finishing":    (0.30, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )", "Total Big Chances Scored"]),
        "Creativity":   (0.28, ["Key Passes (Attempt Assists)", "Goal Assists", "Total Big Chances Created", "Through balls"]),
        "Dribbling":    (0.22, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Ball Security":(0.15, ["Retention %", "Pass %", "Dribble %"]),
        "Progression":  (0.05, ["Final Third Touches", "Total Touches In Opposition Box"]),
    },
    "Shadow Striker": {
        "Finishing":     (0.35, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )", "Total Big Chances Scored"]),
        "Movement":      (0.25, ["Goals from Inside Box", "Total Touches In Opposition Box"]),
        "Dribbling":     (0.20, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Creativity":    (0.15, ["Key Passes (Attempt Assists)", "Goal Assists"]),
        "Ball Security": (0.05, ["Retention %", "Dribble %"]),
    },
    "Trequartista": {
        "Creativity":      (0.30, ["Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created", "Goal Assists"]),
        "Dribbling":       (0.25, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Progression":     (0.20, ["Final Third Touches", "Forward Passes"]),
        "Ball Security":   (0.15, ["Retention %", "Dribble %"]),
        "Finishing":       (0.10, ["Goals", "Non-Penalty Goals"]),
    },
    "Mezzala": {
        "Progression":      (0.30, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Attacking Output": (0.25, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)", "Total Big Chances Created"]),
        "Dribbling":        (0.20, ["Successful Dribbles", "Dribble %"]),
        "Defensive Work":   (0.15, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Ball Security":    (0.10, ["Retention %", "Pass %"]),
    },
    # --- Central Midfield roles ---
    "Defensive Midfielder": {
        "Defensive Shield": (0.35, ["Interceptions", "Tackles Won", "Tackle Win %", "Recoveries"]),
        "Duels":            (0.25, ["Ground Duels won", "Ground Duel %", "Aerial Duels won", "Aerial Win %"]),
        "Distribution":     (0.20, ["Pass %", "Total Passes", "Own Half Pass %"]),
        "Ball Security":    (0.15, ["Retention %", "Pass %"]),
        "Positioning":      (0.05, ["Blocked Shots", "Blocks", "Total Clearances"]),
    },
    "Box-to-Box Midfielder": {
        "Defensive Work":   (0.25, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Attacking Output": (0.20, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)", "Total Big Chances Created"]),
        "Progression":      (0.20, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Duels":            (0.20, ["Duels won", "Duel %", "Ground Duels won", "Ground Duel %"]),
        "Ball Security":    (0.15, ["Retention %", "Pass %"]),
    },
    "Deep-Lying Playmaker": {
        "Distribution":        (0.30, ["Pass %", "Total Passes", "Successful Long Passes", "Short Pass %"]),
        "Progression":         (0.25, ["Forward Passes", "Final Third Touches", "Progressive Carries"]),
        "Ball Security":       (0.20, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Defensive Awareness": (0.15, ["Interceptions", "Recoveries"]),
        "Creativity":          (0.10, ["Key Passes (Attempt Assists)", "Total Big Chances Created"]),
    },
    "Central Midfielder": {
        "Passing Quality":  (0.25, ["Pass %", "Total Passes", "Forward Passes"]),
        "Defensive Work":   (0.20, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Progression":      (0.20, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Attacking Output": (0.15, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)", "Total Big Chances Created"]),
        "Ball Security":    (0.20, ["Retention %", "Pass %"]),
    },
    "Ball Winning Midfielder": {
        "Tackling & Pressing": (0.35, ["Tackles Won", "Tackle Win %", "Total Tackles", "Interceptions"]),
        "Duels":               (0.30, ["Ground Duels won", "Ground Duel %", "Duels won", "Duel %"]),
        "Recoveries":          (0.20, ["Recoveries", "Total Clearances", "Blocked Shots"]),
        "Ball Security":       (0.15, ["Retention %", "Pass %"]),
    },
    # --- Centre-Back roles ---
    "Ball Playing": {
        "Distribution":       (0.30, ["Pass %", "Successful Long Passes", "Long Pass %", "Forward Passes"]),
        "Progression":        (0.20, ["Progressive Carries", "Through balls"]),
        "Defensive Solidity": (0.20, ["Interceptions", "Tackles Won", "Total Clearances"]),
        "Ball Security":      (0.20, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Aerial":             (0.10, ["Aerial Duels won", "Aerial Win %"]),
    },
    "Duelist": {
        "Tackling":           (0.25, ["Tackles Won", "Tackle Win %", "Total Tackles", "Interceptions"]),
        "Duels":              (0.25, ["Ground Duels won", "Ground Duel %", "Duels won", "Duel %"]),
        "Aerial":             (0.20, ["Aerial Duels won", "Aerial Win %"]),
        "Defensive Solidity": (0.20, ["Total Clearances", "Blocked Shots", "Blocks", "Recoveries"]),
        "Ball Security":      (0.10, ["Retention %", "Pass %", "Own Half Pass %"]),
    },
    "Libero": {
        "Ball Progression":   (0.30, ["Progressive Carries", "Through balls", "Forward Passes"]),
        "Passing Quality":    (0.25, ["Pass %", "Total Successful Passes ( Excl Crosses & Corners ) ", "Short Pass %"]),
        "Defensive Solidity": (0.20, ["Tackles Won", "Interceptions", "Total Clearances"]),
        "Dribbling":          (0.15, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Ball Security":      (0.10, ["Retention %", "Own Half Pass %"]),
    },
    # --- Full-Back roles ---
    "Inverted": {
        "Passing Quality": (0.30, ["Pass %", "Forward Passes", "Short Pass %", "Through balls"]),
        "Progression":     (0.25, ["Progressive Carries", "Successful Dribbles"]),
        "Distribution":    (0.20, ["Successful Long Passes", "Long Pass %"]),
        "Defensive Duty":  (0.15, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Ball Security":   (0.10, ["Retention %", "Pass %"]),
    },
    "Defensive": {
        "Defensive Solidity": (0.30, ["Tackles Won", "Tackle Win %", "Interceptions"]),
        "Aerial":             (0.15, ["Aerial Duels won", "Aerial Win %"]),
        "Duels":              (0.25, ["Ground Duels won", "Ground Duel %", "Duels won"]),
        "Distribution":       (0.15, ["Pass %", "Successful Long Passes"]),
        "Ball Security":      (0.15, ["Retention %", "Pass %"]),
    },
    "Attacking": {
        "Attacking Output":      (0.30, ["Goal Assists", "Key Passes (Attempt Assists)", "Successful Crosses & Corners", "Cross %"]),
        "Crossing & Creativity": (0.20, ["Through balls", "Total Big Chances Created", "Successful Crosses open play"]),
        "Progression":           (0.20, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Dribbling":             (0.15, ["Successful Dribbles", "Dribble %"]),
        "Ball Security":         (0.15, ["Retention %", "Pass %"]),
    },
    "All-Around": {
        "Defensive Duty":    (0.25, ["Tackles Won", "Interceptions", "Duels won", "Ground Duels won"]),
        "Attacking Output":  (0.20, ["Goal Assists", "Successful Crosses & Corners", "Key Passes (Attempt Assists)"]),
        "Passing Quality":   (0.20, ["Pass %", "Forward Passes", "Total Passes"]),
        "Progression":       (0.20, ["Progressive Carries", "Successful Dribbles"]),
        "Ball Security":     (0.15, ["Retention %", "Pass %"]),
    },
    # --- Goalkeeper roles ---
    "Shot-Stopper": {
        "Shot-Stopping": (0.45, ["Saves Made", "Save %", "Goals Prevented"]),
        "Penalties":     (0.10, ["Penalties Saved"]),
        "Command":       (0.25, ["Catches", "Punches", "Aerial Duels won"]),
        "Distribution":  (0.20, ["Launch %"]),
    },
    "Sweeper Keeper": {
        "Sweeping":      (0.35, ["Goalkeeper Smother", "Recoveries", "Total Clearances", "Interceptions"]),
        "Shot-Stopping": (0.30, ["Goals Prevented", "Save %"]),
        "Command":       (0.20, ["Catches", "Aerial Duels won"]),
        "Distribution":  (0.15, ["Successful Launches", "Launch %"]),
    },
    "Ball-Playing Goalkeeper": {
        "Distribution":  (0.40, ["GK Successful Distribution", "Successful Launches", "Launch %"]),
        "Shot-Stopping": (0.30, ["Goals Prevented", "Save %"]),
        "Sweeping":      (0.15, ["Goalkeeper Smother", "Recoveries"]),
        "Command":       (0.15, ["Catches", "Punches"]),
    },
}

# Inject the measured xG/xA metrics into every role's attacking categories so a
# striker/winger/CAM is graded on chance quality (npxG), finishing skill
# (sample-shrunk npG − xG) and chance creation (xA) — not just raw goals/shots/
# assists.  Done programmatically so all attacking roles stay consistent without
# editing each profile by hand.
_CREATION_CAT_HINTS = ("Creat", "Link-Up", "Playmak", "Chance", "Shot Creation",
                       "Crossing", "Hold-Up")
for _role_prof in _ROLE_KPI_PROFILES.values():
    for _cat, (_w, _metrics) in _role_prof.items():
        if _cat == "Finishing":
            for _m in ("npxG", "npG-xG (shrunk)"):
                if _m not in _metrics:
                    _metrics.append(_m)
        if any(_h in _cat for _h in _CREATION_CAT_HINTS) and "xA" not in _metrics:
            _metrics.append("xA")

# ── Ball Security refinement ─────────────────────────────────────────────────
# (1) "Ball Security" measures ball RETENTION, not pass completion.  Pass % is
#     already graded under each role's passing/distribution category, so strip
#     it here to avoid double-counting a player's completion rate.
# (2) Playmakers are paid to take risks — their lower retention should not
#     dominate — so down-weight Ball Security for creative/playmaking roles.
_PLAYMAKER_ROLES = {
    "Central Midfielder", "Deep-Lying Playmaker", "Advanced Playmaker",
    "Classic 10", "Trequartista", "Creative Winger", "Mezzala",
}
for _role, _cats in _ROLE_KPI_PROFILES.items():
    if "Ball Security" in _cats:
        _w, _mets = _cats["Ball Security"]
        _mets = [m for m in _mets if m != "Pass %"]
        if _role in _PLAYMAKER_ROLES:
            _w = min(_w, 0.10)
        _cats["Ball Security"] = (_w, _mets)

GK_ATTRIBUTE_GRADE_CATEGORIES = {
    "Shot-Stopping": ["Saves/90", "Save %",
                      "PSxG+/-", "PSxG/Shot",
                      "Goals Prevented",
                      "Total Big Chances Saved",
                      "Clean Sheet %",
                      "Penalties Saved"],
    "Command": ["Catches", "Punches", "Aerial Duels won",
                "Aerial Duels", "Aerial Win %"],
    "Distribution": ["GK Successful Distribution", "Successful Launches",
                     "Launch %"],
    "Sweeping": ["Recoveries", "Total Clearances", "Interceptions"],
}

# ── Role Exceptional Contribution Bonuses ────────────────────────────────────
# Maps a role → (bonus_label, [out-of-role metrics that signal exceptional value]).
# When a player ranks highly vs same-position peers on these *unexpected* metrics,
# their overall grade receives a small bonus (up to +8 percentile pts).
# Classic example: Casemiro (Anchor Man) scoring goals at the 90th+ percentile.
_ROLE_EXCEPTIONAL_CONTRIBUTIONS = {
    # ── Central Midfielders ───────────────────────────────────────────────
    "Defensive Midfielder":  ("Attacking Output",    ["Goals", "Goal Assists", "Key Passes (Attempt Assists)", "Shots On Target ( inc goals )"]),
    "Box-to-Box Midfielder": ("Elite Finishing",     ["Goals", "Non-Penalty Goals", "Total Big Chances Scored"]),
    "Deep-Lying Playmaker":  ("Goal Threat",         ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )", "Total Touches In Opposition Box"]),
    "Central Midfielder":    ("Elite Finishing",     ["Goals", "Non-Penalty Goals", "Total Big Chances Scored"]),
    "Ball Winning Midfielder": ("Attacking Output",  ["Goals", "Goal Assists", "Key Passes (Attempt Assists)", "Shots On Target ( inc goals )"]),
    "Ball Playing":          ("Defensive Dominance", ["Total Tackles", "Interceptions", "Aerial Duels won", "Aerial Win %"]),
    "Duelist":               ("Ball Distribution",   ["Total Passes", "Forward Passes", "Successful Long Passes", "Pass %"]),
    "Libero":               ("Defensive Dominance",  ["Total Tackles", "Interceptions", "Aerial Duels won", "Total Clearances"]),
    "Duelist":              ("Ball Distribution",    ["Total Passes", "Forward Passes", "Successful Long Passes", "Pass %"]),
    # ── Full-Backs ───────────────────────────────────────────────────────
    "Inverted":              ("Attacking Threat",    ["Goal Assists", "Goals", "Key Passes (Attempt Assists)", "Total Shots"]),
    "Defensive":             ("Attacking Threat",    ["Goal Assists", "Goals", "Successful Crosses & Corners"]),
    "Attacking":             ("Defensive Solidity",  ["Total Tackles", "Interceptions", "Aerial Duels won", "Tackle Win %"]),
    "All-Around":            ("Goal Threat",         ["Goals", "Non-Penalty Goals", "Total Shots", "Shots On Target ( inc goals )"]),
    # ── Strikers ─────────────────────────────────────────────────────────
    "Prolific Striker":     ("Pressing Work Rate",  ["Recoveries", "Total Tackles", "Interceptions", "Ground Duels won"]),
    "Target Man":           ("Creative Link-Up",    ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"]),
    "Pressing Forward":     ("Clinical Finishing",  ["Goals", "Non-Penalty Goals", "Total Big Chances Scored"]),
    "False 9":              ("Goal Scoring",        ["Goals", "Non-Penalty Goals", "Total Shots"]),
    # ── Wingers ──────────────────────────────────────────────────────────
    "Inverted Winger":       ("Defensive Contribution", ["Recoveries", "Total Tackles", "Interceptions", "Ground Duels won"]),
    "Classic Winger":       ("Goal Threat",         ["Goals", "Non-Penalty Goals", "Total Shots", "Shots On Target ( inc goals )"]),
    "Creative Winger":      ("Goal Scoring",        ["Goals", "Non-Penalty Goals", "Total Shots"]),
    "Pressing Winger":      ("Creative Output",     ["Key Passes (Attempt Assists)", "Goal Assists", "Through balls", "Total Big Chances Created"]),
    # ── CAMs ─────────────────────────────────────────────────────────────
    "Classic 10":           ("Goal Scoring",        ["Goals", "Non-Penalty Goals", "Total Shots", "Shots On Target ( inc goals )"]),
    "Advanced Playmaker":   ("Defensive Solidity",  ["Total Tackles", "Interceptions", "Recoveries", "Tackle Win %"]),
    "Shadow Striker":       ("Creative Playmaking", ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"]),
    "Trequartista":         ("Goal Scoring",        ["Goals", "Non-Penalty Goals", "Total Shots"]),
    "Mezzala":              ("Clinical Finishing",  ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )"]),
    # ── Goalkeepers ─────────────────────────────────────────────────────
    "Shot-Stopper":         ("Distribution Quality",    ["GK Successful Distribution", "Successful Launches", "Launch %"]),
    "Sweeper Keeper":       ("Shot-Stopping Ability",   ["Saves Made", "Save %", "Goals Prevented"]),
    "Ball-Playing Goalkeeper": ("Shot-Stopping Ability",["Saves Made", "Save %", "Goals Prevented", "Total Big Chances Saved"]),
}

# Bonus tiers: (min_avg_bonus_percentile, bonus_percentile_pts, tier_label)
_EXCEPTIONAL_BONUS_TIERS = [
    (92, 8, "World Class"),
    (85, 6, "Elite"),
    (75, 4, "High"),
    (65, 2, "Notable"),
]

# ── Percentile-based role profiles (Opta metrics) ───────────────────────────
WINGER_ROLE_PROFILES = {
    "Inverted Winger": [
        "Goals", "Non-Penalty Goals", "Total Shots",
        "Shots On Target ( inc goals )", "Goals from Inside Box",
        "Total Touches In Opposition Box",
        "Successful Dribbles",
        "Progressive Carries", "Carries",
        "Total Big Chances Scored",
    ],
    "Classic Winger": [
        "Successful Crosses & Corners", "Successful Crosses open play",
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Total Big Chances Created",
        "Through balls",
    ],
    "Creative Winger": [
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Total Big Chances Created",
        "Through balls", "Final Third Touches",
        "Forward Passes",
        "Successful Dribbles",
    ],
    "Pressing Winger": [
        "Recoveries",
        "Total Tackles", "Tackles Won", "Interceptions",
        "Ground Duels", "Ground Duels won",
        "Total Fouls Conceded",
    ],
}

STRIKER_ROLE_PROFILES = {
    "Prolific Striker": [
        "Goals", "Non-Penalty Goals", "Headed Goals",
        "Total Shots", "Shots On Target ( inc goals )", "Goals from Inside Box",
        "Total Touches In Opposition Box",
        "Total Big Chances Scored",
        "Total Fouls Won",
    ],
    "Target Man": [
        "Aerial Duels", "Aerial Duels won", "Aerial Win %",
        "Headed Goals", "Goals", "Non-Penalty Goals",
        "Total Shots", "Shots On Target ( inc goals )",
        "Duels", "Duels won",
        "Total Fouls Won",
    ],
    "False 9": [
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Total Big Chances Created",
        "Through balls", "Final Third Touches",
        "Successful Dribbles",
        "Progressive Carries", "Carries",
        "Goals", "Non-Penalty Goals",
    ],
    "Pressing Forward": [
        "Recoveries",
        "Total Tackles", "Tackles Won", "Interceptions",
        "Ground Duels", "Ground Duels won",
        "Total Fouls Conceded",
    ],
}

CM_ROLE_PROFILES = {
    "Defensive Midfielder": [
        "Interceptions", "Total Clearances", "Blocked Shots", "Blocks",
        "Aerial Duels", "Aerial Duels won", "Aerial Win %",
        "Total Tackles", "Tackles Won", "Recoveries",
        "Duels", "Duels won", "Ground Duels", "Ground Duels won",
    ],
    "Box-to-Box Midfielder": [
        "Goals", "Non-Penalty Goals", "Goal Assists",
        "Total Shots", "Shots On Target ( inc goals )",
        "Total Tackles", "Tackles Won", "Interceptions", "Recoveries",
        "Progressive Carries", "Carries",
        "Key Passes (Attempt Assists)",
        "Ground Duels", "Ground Duels won",
        "Duels", "Duels won",
    ],
    "Deep-Lying Playmaker": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Successful Long Passes",
        "Forward Passes",
        "Through balls", "Final Third Touches",
    ],
    "Central Midfielder": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %", "Forward Passes",
        "Total Tackles", "Tackles Won", "Interceptions", "Recoveries",
        "Progressive Carries",
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Duels", "Duels won",
    ],
    "Ball Winning Midfielder": [
        "Total Tackles", "Tackles Won", "Tackle Win %",
        "Interceptions", "Recoveries",
        "Ground Duels", "Ground Duels won", "Ground Duel %",
        "Duels", "Duels won", "Duel %",
        "Total Clearances", "Blocked Shots", "Blocks",
        "Aerial Duels won", "Aerial Win %",
    ],
}

CAM_ROLE_PROFILES = {
    "Classic 10": [
        "Key Passes (Attempt Assists)", "Goal Assists",
        "Total Big Chances Created",
        "Through balls",
        "Pass %", "Forward Passes",
        "Final Third Touches",
    ],
    "Advanced Playmaker": [
        "Goals", "Non-Penalty Goals",
        "Key Passes (Attempt Assists)", "Goal Assists",
        "Total Shots", "Shots On Target ( inc goals )",
        "Total Big Chances Created",
        "Successful Dribbles",
        "Total Touches In Opposition Box",
    ],
    "Shadow Striker": [
        "Goals", "Non-Penalty Goals", "Total Shots",
        "Shots On Target ( inc goals )",
        "Total Touches In Opposition Box",
        "Key Passes (Attempt Assists)",
        "Total Big Chances Scored",
    ],
    "Trequartista": [
        "Successful Dribbles", "Dribble %",
        "Key Passes (Attempt Assists)", "Through balls",
        "Total Big Chances Created", "Goal Assists",
        "Progressive Carries", "Final Third Touches",
    ],
    "Mezzala": [
        "Progressive Carries", "Carries",
        "Goals", "Non-Penalty Goals", "Goal Assists",
        "Successful Dribbles",
        "Key Passes (Attempt Assists)",
        "Total Touches In Opposition Box",
        "Total Tackles", "Recoveries",
    ],
}

CENTRE_BACK_ROLE_PROFILES = {
    "Ball Playing": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Successful Long Passes",
        "Forward Passes",
        "Progressive Carries", "Carries",
    ],
    "Duelist": [
        "Total Tackles", "Tackles Won", "Tackle Win %",
        "Aerial Duels", "Aerial Duels won", "Aerial Win %",
        "Ground Duels", "Ground Duels won", "Ground Duel %",
        "Total Clearances", "Blocked Shots", "Blocks",
        "Recoveries", "Duels", "Duels won",
    ],
    "Libero": [
        "Progressive Carries", "Carries", "Successful Dribbles",
        "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %", "Through balls", "Forward Passes",
        "Short Pass %",
    ],
}

FULL_BACK_ROLE_PROFILES = {
    "Inverted": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Progressive Carries", "Carries",
        "Recoveries",
        "Ground Duels", "Ground Duels won",
        "Successful Dribbles",
        "Through balls",
    ],
    "Defensive": [
        "Total Tackles", "Tackles Won", "Interceptions",
        "Total Clearances", "Blocked Shots", "Blocks",
        "Duels", "Duels won",
        "Ground Duels", "Ground Duels won",
    ],
    "Attacking": [
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Successful Crosses & Corners", "Successful Crosses open play",
        "Progressive Carries", "Carries",
        "Successful Dribbles",
        "Total Touches In Opposition Box", "Goals", "Total Shots",
        "Total Big Chances Created",
        "Forward Passes", "Through balls",
    ],
    "All-Around": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Total Tackles", "Tackles Won", "Interceptions",
        "Successful Crosses & Corners",
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Duels", "Duels won",
        "Progressive Carries",
    ],
}

GOALKEEPER_ROLE_PROFILES = {
    # Classify by shot-stopping QUALITY (PSxG+/- vs expectation), not save
    # volume.  "Saves Made"/"Goals Prevented" mostly reflect how many shots a
    # keeper's team concedes, so a busy keeper on a weak side looked like a
    # "shot-stopper" regardless of skill.  Penalties Saved is mostly zero and
    # only added noise.
    "Shot-Stopper": [
        "PSxG+/- (shrunk)", "Save %", "Total Big Chances Saved",
    ],
    "Sweeper Keeper": [
        "Recoveries", "Interceptions",
        "Total Clearances",
        "Catches", "Aerial Duels won", "Aerial Duels",
    ],
    "Ball-Playing Goalkeeper": [
        "GK Successful Distribution", "Successful Launches",
        "Launch %",
    ],
}

POSITION_ROLE_PROFILES = {
    "Wingers": WINGER_ROLE_PROFILES,
    "Attacking Midfield": CAM_ROLE_PROFILES,
    "Striker": STRIKER_ROLE_PROFILES,
    "Central Midfield": CM_ROLE_PROFILES,
    "Centre-Back": CENTRE_BACK_ROLE_PROFILES,
    "Full-Back": FULL_BACK_ROLE_PROFILES,
    "Goalkeeper": GOALKEEPER_ROLE_PROFILES,
}


def _classify_percentile_role(player_row, df_peers, position):
    """Classify a player by finding the role profile where they
    rank highest in average per-metric percentile vs same-position peers.
    Each metric is individually ranked so that percentage stats (e.g. 60-90%)
    don't dominate count stats (e.g. 2-10) when summed together."""
    profiles = POSITION_ROLE_PROFILES.get(position)
    if not profiles:
        return position or "Unknown"

    pos_peers = df_peers[df_peers["posicion"] == position]
    if pos_peers.empty:
        return position

    best_role, best_avg = position, -1.0
    for role, metrics in profiles.items():
        # Skip % success-rate stats for classification — they cluster
        # tightly around 50-60% for all players and don't discriminate
        # between roles.  Volume/count stats are the true differentiators.
        # The % stats are kept in profiles for display (scouting report).
        avail = [m for m in metrics
                 if m in pos_peers.columns and not m.startswith("% ")]
        if not avail:
            continue
        metric_pcts = []
        for m in avail:
            val = player_row.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            peer_vals = pos_peers[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            metric_pcts.append(pct)
        avg_pct = sum(metric_pcts) / len(metric_pcts)
        if avg_pct > best_avg:
            best_avg = avg_pct
            best_role = role
    return best_role


def _classify_role(row, position, df_total=None):
    """Return the most fitting role label for a player."""
    if position in POSITION_ROLE_PROFILES and df_total is not None:
        return _classify_percentile_role(dict(row), df_total, position)
    defaults = {"Wingers": "Wingers",
                "Attacking Midfield": "Attacking Midfield",
                "Striker": "Striker",
                "Central Midfield": "Central Midfield",
                "Centre-Back": "Centre-Back", "Full-Back": "Full-Back",
                "Goalkeeper": "Goalkeeper"}
    return defaults.get(position, position or "Unknown")


def _compute_overall_percentile(player_row, df_peers, categories):
    """Return the average percentile (0-100) across all categories."""
    pcts = _compute_percentiles(player_row, df_peers, categories)
    if not pcts:
        return 0.0
    return sum(pcts.values()) / len(pcts)


_GRADE_THRESHOLDS = [
    (97, "S+"), (93, "S"), (90, "S-"),
    (87, "A+"), (83, "A"), (80, "A-"),
    (77, "B+"), (73, "B"), (70, "B-"),
    (67, "C+"), (63, "C"), (60, "C-"),
    (55, "D+"), (50, "D"), (45, "D-"),
    (0,  "F"),
]

_GRADE_COLORS = {
    "S+": "#00c853", "S": "#00e676", "S-": "#69f0ae",
    "A+": "#2979ff", "A": "#448aff", "A-": "#82b1ff",
    "B+": "#aa00ff", "B": "#d500f9", "B-": "#ea80fc",
    "C+": "#ff9100", "C": "#ffab40", "C-": "#ffd180",
    "D+": "#ff3d00", "D": "#ff6e40", "D-": "#ff9e80",
    "F": "#d50000",
    "N/A": "#555555",
}

# ── Potential Grading — Age & Environment Constants ──────────────────────────

# Career phase labels (matched to age brackets in _get_age_growth)
_POTENTIAL_PHASE_COLORS = {
    "🚀 Prodigy":        "#00c853",
    "🚀 High Potential": "#00e676",
    "📈 Rising Star":    "#69f0ae",
    "📈 Developing":     "#40c4ff",
    "📈 Growing":        "#82b1ff",
    "⭐ Prime":          "#ffd740",
    "⭐ Late Prime":     "#ffab40",
    "📉 Declining":      "#ff6e40",
    "📉 Fading":         "#e53935",
    "📉 Late Career":    "#b71c1c",
}

# Club / league upgrade tiers
# ceiling_boost: extra percentile pts added to the projected ceiling over time
# year1_dip:     adaptation penalty subtracted in the first projected year
_CLUB_TIERS = {
    "No major change": {
        "ceiling_boost": 0, "year1_dip": 0,
        "desc": "Player stays at same club or moves laterally — no structural environment change.",
        "color": "#aaa",
    },
    "Elite Club upgrade (e.g. Belgian league → Man Utd, Ajax → Real Madrid)": {
        "ceiling_boost": 10, "year1_dip": 5,
        "desc": (
            "Joining a Champions League elite transforms training, coaching and daily competition. "
            "Expect a year-1 adaptation dip then accelerated growth — especially for young players."
        ),
        "color": "#2d6a4f",
    },
    "Big Club upgrade (e.g. Championship → PL Top 6, Ligue 2 → Big-6 club)": {
        "ceiling_boost": 6, "year1_dip": 3,
        "desc": "Step up to a top club in a top league — meaningful quality jump with moderate adaptation period.",
        "color": "#457b9d",
    },
    "Top League upgrade (e.g. minor league → Big 5, lower div → top div)": {
        "ceiling_boost": 5, "year1_dip": 2,
        "desc": "Moving to a stronger league raises the competitive baseline and accelerates development.",
        "color": "#f4a261",
    },
    "Same level club move": {
        "ceiling_boost": 1, "year1_dip": 1,
        "desc": "Lateral move — small fresh-start bonus, minimal structural change.",
        "color": "#e9c46a",
    },
    "Downgrade / lesser league": {
        "ceiling_boost": -4, "year1_dip": 0,
        "desc": "Moving to a weaker league typically reduces the ceiling due to lower competition quality.",
        "color": "#e63946",
    },
}


def _percentile_to_grade(pct):
    """Convert an overall percentile (0-100) to a letter grade."""
    try:
        if pct is None or (isinstance(pct, float) and (pct != pct)):  # None or NaN
            return "N/A"
        pct = float(pct)
    except (TypeError, ValueError):
        return "N/A"
    for threshold, grade in _GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return "F"


def _grade_html(grade, label, pct=None):
    """Return styled HTML for a grade card."""
    color = _GRADE_COLORS.get(grade, "#888")
    pct_html = f"<div style='font-size:10px;color:#777;margin-top:2px;'>{_ordinal(pct)}</div>" if pct is not None else ""
    return (
        f"<div style='text-align:center;'>"
        f"<div style='font-size:11px;color:#aaa;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:36px;font-weight:bold;color:{color};'>{grade}</div>"
        f"{pct_html}"
        f"</div>"
    )


@st.cache_data(ttl=86400, show_spinner="Classifying roles…")
def _classify_position_roles(df_total, position, role_schema=""):
    """Return a Series mapping df_total index → role for all players at *position*.
    Vectorized: pre-computes percentile ranks once, then averages per role profile."""
    profiles = POSITION_ROLE_PROFILES.get(position)
    if not profiles:
        return pd.Series(position or "Unknown", index=df_total.index)

    pos_df = df_total[df_total["posicion"] == position]
    if pos_df.empty:
        return pd.Series(dtype=str)

    # Pre-compute percentile ranks for ALL metrics used across all role profiles
    all_metrics = set()
    for metrics in profiles.values():
        for m in metrics:
            if m in pos_df.columns and not m.startswith("% "):
                all_metrics.add(m)
    all_metrics = sorted(all_metrics)

    if not all_metrics:
        return pd.Series(position, index=pos_df.index)

    # Vectorized rank: for each metric, rank all players at once
    pct_ranks = pos_df[all_metrics].fillna(0).rank(pct=True) * 100

    # For each role, compute mean percentile across its metrics
    role_avgs = {}
    for role, metrics in profiles.items():
        avail = [m for m in metrics if m in pct_ranks.columns and not m.startswith("% ")]
        if avail:
            role_avgs[role] = pct_ranks[avail].mean(axis=1)

    if not role_avgs:
        return pd.Series(position, index=pos_df.index)

    # Stack into a DataFrame and pick the role with highest avg per player
    role_df = pd.DataFrame(role_avgs, index=pos_df.index)
    return role_df.idxmax(axis=1)


# A player is "eligible" for every role scoring within this many percentile
# points of their best-fit role — so versatile players (e.g. a deep mid strong
# as both a Deep-Lying Playmaker and a Defensive Midfielder) surface under all
# the roles they fit, not just their single top one.
_ROLE_ELIGIBILITY_MARGIN = 6.0


def _eligible_roles_map(df_total, position):
    """{index → tuple of roles the player fits} for one position.  Mirrors the
    scoring in _classify_position_roles but keeps every role within the margin
    of the player's best (always includes their primary role)."""
    profiles = POSITION_ROLE_PROFILES.get(position)
    if not profiles:
        return pd.Series(dtype=object)
    pos_df = df_total[df_total["posicion"] == position]
    if pos_df.empty:
        return pd.Series(dtype=object)
    all_metrics = sorted({m for metrics in profiles.values() for m in metrics
                          if m in pos_df.columns and not m.startswith("% ")})
    if not all_metrics:
        return pd.Series([(position,)] * len(pos_df), index=pos_df.index)
    pct_ranks = pos_df[all_metrics].fillna(0).rank(pct=True) * 100
    role_avgs = {role: pct_ranks[[m for m in metrics
                                  if m in pct_ranks.columns and not m.startswith("% ")]].mean(axis=1)
                 for role, metrics in profiles.items()
                 if any(m in pct_ranks.columns and not m.startswith("% ") for m in metrics)}
    if not role_avgs:
        return pd.Series([(position,)] * len(pos_df), index=pos_df.index)
    role_df = pd.DataFrame(role_avgs, index=pos_df.index)
    thresh = role_df.max(axis=1) - _ROLE_ELIGIBILITY_MARGIN
    elig = role_df.ge(thresh, axis=0)
    return elig.apply(lambda r: tuple(role_df.columns[r.to_numpy()]), axis=1)


def _compute_attribute_grades(row_data, position, df_total, league=None, role=None,
                               df_role_ref=None, kpi_role=None):
    """Compute per-attribute grades for a player vs peers.
    - Always filters by same position.
    - If role is provided AND no KPI profile, further filters to same-role peers.
    - If kpi_role has a KPI profile, uses position-level peers (the curated
      metrics already encode role specificity — adding a role peer filter on top
      would double-filter and compress the grade range).
    Returns dict: {attribute_name: (grade, percentile)}."""
    _is_gk = position == "Goalkeeper"
    kpi = None
    if _is_gk:
        # GK grades always derive from the same 4 pizza chart categories so
        # the grade and the pizza chart are always consistent with each other.
        # Role-specific KPI profiles are intentionally bypassed for GKs because
        # they contain single-metric categories (e.g. "Penalties Saved" alone)
        # that push most keepers to 0th percentile and produce misleading grades.
        cats = {cat: [col for _, col in metrics] for cat, metrics in GK_PIZZA_METRICS.items()}
        inv_cats = _KPI_INVERTED_CATS
        grade_weights = GK_SHOTSTOP_GRADE_WEIGHTS   # PSxG+/- is the core of Shot-Stopping
    else:
        grade_weights = None
        kpi = _ROLE_KPI_PROFILES.get(kpi_role) if kpi_role else None
        if kpi:
            cats = {name: metrics for name, (weight, metrics) in kpi.items()}
            inv_cats = _KPI_INVERTED_CATS
        else:
            cats = ATTRIBUTE_GRADE_CATEGORIES
            inv_cats = _INVERTED_GRADE_CATS

    peers = df_total[df_total["posicion"] == position]
    # Fallback: Opta groups all forwards as "Forward" → "Striker", so
    # "Wingers" has 0 players in the data (Opta maps all forwards as
    # "Striker").  When the target position is empty, fall back.
    _PEER_FALLBACK = {"Wingers": "Striker"}
    if len(peers) < 5 and position in _PEER_FALLBACK:
        peers = df_total[df_total["posicion"] == _PEER_FALLBACK[position]]
    # Only apply role-based peer filtering when using generic categories.
    # KPI profiles already encode role specificity via curated metrics,
    # so we skip the role filter to avoid compressing the percentile range.
    if role and not kpi:
        ref = df_role_ref if df_role_ref is not None else df_total
        role_series = _classify_position_roles(ref, position, role_schema=_ROLE_SCHEMA_VERSION)
        matching_names = ref.loc[
            ref.index.intersection(role_series[role_series == role].index), "nombre"
        ]
        peers = peers[peers["nombre"].isin(matching_names)]
    if league:
        peers = peers[peers["league_display"] == league]

    if len(peers) < 5:
        return {attr: ("N/A", None) for attr in cats}

    pcts = _compute_percentiles(row_data, peers, cats, inverted_cats=inv_cats,
                                weights=grade_weights)

    result = {}
    for attr, pct in pcts.items():
        grade = _percentile_to_grade(pct)
        result[attr] = (grade, round(pct, 1))
    return result


def _compute_player_grades(row_data, position, df_total, role=None):
    """Compute composite attribute-based grades (league & overall).
    Returns (lg_grade, lg_pct, ov_grade, ov_pct) as composite of all attributes."""
    league = row_data.get("league_display", "")

    attr_lg = _compute_attribute_grades(row_data, position, df_total, league=league,
                                        kpi_role=role)
    attr_ov = _compute_attribute_grades(row_data, position, df_total, league=None,
                                        kpi_role=role)

    if position == "Goalkeeper":
        weights = _ROLE_GRADE_WEIGHTS.get(role, {})
    else:
        kpi = _ROLE_KPI_PROFILES.get(role) if role else None
        if kpi:
            weights = {name: w for name, (w, _) in kpi.items()}
        else:
            weights = _POSITION_GRADE_WEIGHTS.get(position, {})

    def _weighted_avg(attr_dict):
        pcts = {k: pct for k, (_, pct) in attr_dict.items() if pct is not None}
        if not pcts:
            return 0.0
        if weights:
            w_sum = sum(weights.get(k, 0) for k in pcts)
            if w_sum > 0:
                return sum(pcts[k] * weights.get(k, 0) for k in pcts) / w_sum
        return sum(pcts.values()) / len(pcts)

    lg_avg = _weighted_avg(attr_lg)
    ov_avg = _weighted_avg(attr_ov)

    return (_percentile_to_grade(lg_avg), round(lg_avg, 1),
            _percentile_to_grade(ov_avg), round(ov_avg, 1))


def _compute_percentiles(player_row, df_peers, categories, inverted_cats=None,
                         weights=None):
    """Per-category percentile (0-100) for a player vs peers.
    Ranks each metric individually against peers and averages the per-metric
    percentiles within each category — consistent with the pizza chart display
    so that high-magnitude stats don't dominate low-magnitude ones.

    *weights* (optional): {metric_col: weight} to weight specific metrics more
    heavily within their category (e.g. make PSxG+/- the core of Shot-Stopping).
    Metrics not listed default to weight 1.0."""
    if inverted_cats is None:
        inverted_cats = _INVERTED_GRADE_CATS
    result = {}
    for cat, metrics in categories.items():
        avail = [m for m in metrics if m in df_peers.columns]
        if not avail:
            result[cat] = 0
            continue
        metric_pcts = []
        metric_wts = []
        for m in avail:
            val = player_row.get(m, 0)
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else (val or 0)
            peer_vals = df_peers[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            metric_pcts.append(pct)
            metric_wts.append((weights or {}).get(m, 1.0))
        tot_w = sum(metric_wts) or 1.0
        cat_pct = sum(p * w for p, w in zip(metric_pcts, metric_wts)) / tot_w
        if cat in inverted_cats:
            cat_pct = 100 - cat_pct
        result[cat] = round(cat_pct, 1)
    return result


def _compute_exceptional_contribution(row_data, position, role, df_peers):
    """Return (label, avg_pct, bonus_pts, tier_label) for out-of-role exceptional metrics.

    If the player ranks above ~65th percentile vs same-position peers on metrics that are
    *outside* their primary role (e.g. an Anchor Man scoring goals), they receive a grade
    bonus of up to +8 percentile points added to their Overall Grade.
    Returns (None, 0.0, 0, '') when no exceptional contribution is detected.
    """
    entry = _ROLE_EXCEPTIONAL_CONTRIBUTIONS.get(role)
    if not entry:
        return None, 0.0, 0, ""
    label, metrics = entry

    pos_peers = df_peers[df_peers["posicion"] == position]
    if len(pos_peers) < 5:
        return None, 0.0, 0, ""

    avail = [m for m in metrics if m in pos_peers.columns]
    if not avail:
        return None, 0.0, 0, ""

    metric_pcts = []
    for m in avail:
        val = row_data.get(m, 0)
        val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else (val or 0)
        peer_vals = pos_peers[m].fillna(0)
        pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
        metric_pcts.append(pct)

    avg_pct = round(sum(metric_pcts) / len(metric_pcts), 1) if metric_pcts else 0.0

    bonus_pts, tier_label = 0, ""
    for threshold, pts, badge in _EXCEPTIONAL_BONUS_TIERS:
        if avg_pct >= threshold:
            bonus_pts, tier_label = pts, badge
            break

    return label, avg_pct, bonus_pts, tier_label


def _compute_standout_strengths(row_data, pos_peers, is_gk, n=4, min_pct=70):
    """Return a player's top-N metrics by percentile vs same-position peers.

    Works at the *metric* level (not category level) so a genuinely elite trait
    surfaces even when its parent category grades as average — e.g. a keeper with
    middling overall shot-stopping but elite Big Chances Saved / Command.

    Uses the curated, display-named pizza metric pools so labels are clean and
    position-appropriate.  Only metrics ranking at/above *min_pct* are returned,
    so a player with no standout simply yields fewer (or zero) cards.
    Returns a list of dicts: {label, value, pct, tier, color}.
    """
    pool = GK_PIZZA_METRICS if is_gk else PIZZA_METRICS
    if pos_peers is None or len(pos_peers) < 5:
        return []

    scored, seen = [], set()
    for metric_list in pool.values():
        for label, col in metric_list:
            if col in seen or col not in pos_peers.columns:
                continue
            seen.add(col)
            val = row_data.get(col)
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
            peer_vals = pos_peers[col].fillna(0)
            pct = round((peer_vals < val).sum() / len(peer_vals) * 100, 1)
            scored.append((label, val, pct))

    scored.sort(key=lambda x: x[2], reverse=True)
    out = []
    for label, val, pct in scored:
        if pct < min_pct or len(out) >= n:
            break
        if pct >= 90:
            tier, color = "Elite", "#00c853"
        elif pct >= 80:
            tier, color = "Excellent", "#2979ff"
        else:
            tier, color = "Strong", "#f4a261"
        out.append({"label": label, "value": val, "pct": pct, "tier": tier, "color": color})
    return out


# ── Key-Strengths uplift ─────────────────────────────────────────────────────
# A weighted-average grade washes out elite peaks, so a true specialist (e.g. a
# 99th-pctl creator with average retention) reads as merely "good".  This adds a
# small, capped bonus that rewards a player's elite standout metrics — the same
# metrics shown in the Standout Strengths panel — so the grade leads with what a
# player is genuinely elite at.  Average players (no metric >= 85th) are unchanged.
_KEY_STRENGTH_BONUS_TIERS = [(95, 2.0), (90, 1.0), (85, 0.5)]
_KEY_STRENGTH_BONUS_CAP = 4.0


def _key_strength_bonus_from_pcts(pcts):
    """Capped grade bonus from an iterable of metric percentiles."""
    bonus = 0.0
    for p in pcts:
        if p is None:
            continue
        for thr, pts in _KEY_STRENGTH_BONUS_TIERS:
            if p >= thr:
                bonus += pts
                break
    return min(_KEY_STRENGTH_BONUS_CAP, round(bonus, 1))


def _compute_key_strength_bonus(row_data, pos_peers, is_gk):
    """Capped overall-grade bonus from a player's elite standout metrics."""
    pool = GK_PIZZA_METRICS if is_gk else PIZZA_METRICS
    if pos_peers is None or len(pos_peers) < 5:
        return 0.0
    pcts, seen = [], set()
    for ml in pool.values():
        for _, col in ml:
            if col in seen or col not in pos_peers.columns:
                continue
            seen.add(col)
            val = row_data.get(col)
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
            pv = pos_peers[col].fillna(0)
            pcts.append((pv < val).sum() / len(pv) * 100)
    return _key_strength_bonus_from_pcts(pcts)


# ── UI: Player Lab ───────────────────────────────────────────────────────────

# Grade ordering for slider
_ALL_GRADES = ["F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+",
               "A-", "A", "A+", "S-", "S", "S+"]
_GRADE_TO_IDX = {g: i for i, g in enumerate(_ALL_GRADES)}


def _parse_market_value(mv_str):
    """Convert Transfermarkt string like '€200.00m' or '€800Th.' to float (euros)."""
    if not mv_str or mv_str == "N/A":
        return None
    s = mv_str.replace("€", "").replace(",", "").strip()
    try:
        if s.lower().endswith("m"):
            return float(s[:-1]) * 1_000_000
        elif s.lower().endswith("th."):
            return float(s[:-3]) * 1_000
        elif s.lower().endswith("k"):
            return float(s[:-1]) * 1_000
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_salary_value(sal_str):
    """Convert Capology string like '€10.54M' to float."""
    if not sal_str or sal_str == "N/A":
        return None
    s = sal_str.replace("€", "").replace(",", "").strip()
    try:
        if s.upper().endswith("M"):
            return float(s[:-1]) * 1_000_000
        elif s.upper().endswith("K"):
            return float(s[:-1]) * 1_000
        return float(s)
    except (ValueError, TypeError):
        return None


def _fmt_euros(val):
    """Format a numeric euro value for display."""
    if val is None:
        return "N/A"
    if val >= 1_000_000:
        return f"€{val / 1_000_000:,.2f}M"
    elif val >= 1_000:
        return f"€{val / 1_000:,.0f}K"
    return f"€{val:,.0f}"


@st.cache_data(ttl=86400, show_spinner=False)
def _build_player_lab_table(grade_df, role_df, mode_label="", pot_years=3, role_schema="", foot_bust=0):
    """Pre-compute grades and roles for every player (vectorized, cached 24 h).

    *grade_df* is the DataFrame used for grading AND role classification, so the
    overall grade reflects each player's best-fit role in the current lens
    (Total / Per-90 / Padj).  *role_df* is retained for signature compatibility.
    """
    financials = _load_financials_csv(_bust=_csv_mtime(_FINANCIALS_CSV))
    gdf = grade_df.copy()

    # ── Vectorized percentile ranks per position ────────────────────────
    # Europe-wide: rank within same position
    # League-scoped: rank within same position + league
    all_attr_cats = {}
    for pos in gdf["posicion"].unique():
        cats = GK_ATTRIBUTE_GRADE_CATEGORIES if pos == "Goalkeeper" else ATTRIBUTE_GRADE_CATEGORIES
        all_attr_cats[pos] = cats

    all_cat_names = set()
    for cats in all_attr_cats.values():
        all_cat_names.update(cats.keys())

    for cat_name in all_cat_names:
        gdf[f"_ov_{cat_name}"] = np.nan
        gdf[f"_lg_{cat_name}"] = np.nan

    # Europe-wide percentiles per position
    for pos, pos_group in gdf.groupby("posicion"):
        if len(pos_group) < 5:
            continue
        cats = all_attr_cats.get(pos, ATTRIBUTE_GRADE_CATEGORIES)
        for cat_name, metrics in cats.items():
            avail = [m for m in metrics if m in gdf.columns]
            if not avail:
                continue
            pct_ranks = gdf.loc[pos_group.index, avail].fillna(0).rank(pct=True) * 100
            cat_pct = pct_ranks.mean(axis=1).round(1)
            if cat_name in _INVERTED_GRADE_CATS:
                cat_pct = (100 - cat_pct).round(1)
            gdf.loc[pos_group.index, f"_ov_{cat_name}"] = cat_pct

    # League-scoped percentiles per position + league
    for (pos, league), group in gdf.groupby(["posicion", "league_display"]):
        if len(group) < 5:
            continue
        cats = all_attr_cats.get(pos, ATTRIBUTE_GRADE_CATEGORIES)
        for cat_name, metrics in cats.items():
            avail = [m for m in metrics if m in gdf.columns]
            if not avail:
                continue
            pct_ranks = gdf.loc[group.index, avail].fillna(0).rank(pct=True) * 100
            cat_pct = pct_ranks.mean(axis=1).round(1)
            if cat_name in _INVERTED_GRADE_CATS:
                cat_pct = (100 - cat_pct).round(1)
            gdf.loc[group.index, f"_lg_{cat_name}"] = cat_pct

    # ── Per-metric percentile ranks for role KPI grading ─────────────────
    _skip_meta = {"nombre", "equipo", "posicion", "posicion_detail",
                  "league_display", "estimated_90s"}
    _metric_cols = [c for c in gdf.columns if c not in _skip_meta
                    and not c.startswith("_ov_") and not c.startswith("_lg_")
                    and gdf[c].dtype.kind in ('f', 'i')]
    _ov_mpct = {}
    for pos, pos_group in gdf.groupby("posicion"):
        if len(pos_group) < 5:
            continue
        _ov_mpct[pos] = gdf.loc[pos_group.index, _metric_cols].fillna(0).rank(pct=True) * 100
    _lg_mpct = {}
    for (pos, lg), group in gdf.groupby(["posicion", "league_display"]):
        if len(group) < 5:
            continue
        _lg_mpct[(pos, lg)] = gdf.loc[group.index, _metric_cols].fillna(0).rank(pct=True) * 100

    # ── Classify roles in the CURRENT grade lens ─────────────────────────
    # Role is graded on whatever frame is being shown (Total / Per-90 / Padj) so
    # the overall grade reflects the player's best-fit role in that lens — a
    # deep mid can be a Deep-Lying Playmaker on totals but a Ball-Winner on
    # per-90 possession-adjusted, and each should grade at that best fit.
    role_map = {}
    elig_map = {}
    for position in gdf["posicion"].unique():
        if position in POSITION_ROLE_PROFILES:
            roles = _classify_position_roles(gdf, position, role_schema=_ROLE_SCHEMA_VERSION)
            for r_idx, r_role in roles.items():
                role_map[gdf.at[r_idx, "nombre"]] = r_role
            elig = _eligible_roles_map(gdf, position)
            for r_idx, r_roles in elig.items():
                elig_map[gdf.at[r_idx, "nombre"]] = r_roles

    # ── Vectorized grade conversion helper ──────────────────────────────
    _bins = [0, 45, 50, 55, 60, 63, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97, 100.01]
    _labels = ["F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+",
               "A-", "A", "A+", "S-", "S", "S+"]

    def _vec_grade(series):
        """Convert a percentile Series to grade strings (vectorized)."""
        return pd.cut(series, bins=_bins, labels=_labels, right=False,
                      include_lowest=True).astype(str).where(series.notna(), "N/A")

    # ── Build result DataFrame without iterrows ─────────────────────────
    gdf["_role"] = gdf["nombre"].map(role_map)
    gdf["_role"] = gdf["_role"].fillna(gdf["posicion"])
    gdf["_role"] = gdf["_role"].where(gdf["_role"] != "", gdf["posicion"])

    # Financials lookup (vectorized, team-guarded against shared-name collisions)
    _fin_recs = [_fin_rec(financials, n, t) for n, t in zip(gdf["nombre"], gdf.get("equipo", pd.Series("", index=gdf.index)))]
    _mv_series = pd.Series([(r.get("market_value") or "—") for r in _fin_recs], index=gdf.index)
    _sal_series = pd.Series([(r.get("salary") or "—") for r in _fin_recs], index=gdf.index)

    out = pd.DataFrame({
        "_idx": gdf.index,
        "Player": gdf["nombre"].values,
        "Team": gdf["equipo"].fillna("Unknown").values,
        "League": gdf["league_display"].values,
        "Pos": gdf["posicion_detail"].fillna("").values,
        "Position": gdf["posicion"].values,
        "Role": gdf["_role"].values,
        "Market Value": _mv_series.values,
        "Salary": _sal_series.values,
    }, index=gdf.index)

    # Roles a player also fits (within the eligibility margin) — used to make the
    # role filter surface versatile players, not just their single best-fit role.
    out["_eligible"] = gdf["nombre"].map(elig_map).apply(
        lambda x: tuple(x) if isinstance(x, (tuple, list)) else ())

    # Squad Role by share of the season (fair across 33–37 game leagues)
    _mins = gdf["Time Played"].fillna(0) if "Time Played" in gdf.columns else gdf["estimated_90s"].fillna(0) * 90
    _season = gdf["league_display"].map(_league_season_minutes(gdf)).replace(0, np.nan)
    _share = _mins / _season
    out["Squad Role"] = np.where(
        _share >= _SQUAD_ROLE_STARTER, "Starter",
        np.where(_share >= _SQUAD_ROLE_ROTATION, "Rotation", "Depth"),
    )

    # ── Top Strength: each player's #1 metric vs same-position peers ──────
    # Uses the curated pizza pools (clean labels) and the Europe-wide per-metric
    # percentile ranks already computed in _ov_mpct.
    out["Top Strength"] = ""
    for pos, mpct in _ov_mpct.items():
        pool = GK_PIZZA_METRICS if pos == "Goalkeeper" else PIZZA_METRICS
        label_map = {col: label for ml in pool.values() for (label, col) in ml}
        cols = [c for c in label_map if c in mpct.columns]
        if not cols:
            continue
        sub = mpct[cols]
        best_col = sub.idxmax(axis=1)
        best_pct = sub.max(axis=1)
        for idx in sub.index:
            out.at[idx, "Top Strength"] = f"{label_map[best_col[idx]]} ({_ordinal(best_pct[idx])})"

    # Preferred foot (Transfermarkt, joined by stable id)
    _foot = _load_footedness_csv(_bust=foot_bust)
    _foot_lbl = {"left": "Left", "right": "Right", "both": "Two-footed"}
    out["Foot"] = (gdf["id"].map(lambda p: _foot_lbl.get(_foot.get(p), ""))
                   if "id" in gdf.columns else "")

    # Top Strength metric name only (for filtering), e.g. "Big Chances Created"
    out["_top_metric"] = out["Top Strength"].str.replace(r"\s*\(.*\)$", "", regex=True)

    # Attribute grade columns (vectorized)
    _outfield_attrs = list(ATTRIBUTE_GRADE_CATEGORIES.keys())
    _gk_attrs = list(GK_ATTRIBUTE_GRADE_CATEGORIES.keys())
    _all_attrs = list(dict.fromkeys(_outfield_attrs + _gk_attrs))  # union, order preserved
    _gk_mask = gdf["posicion"] == "Goalkeeper"

    for attr in _all_attrs:
        ov_col = f"_ov_{attr}"
        lg_col = f"_lg_{attr}"
        ov_vals = gdf[ov_col] if ov_col in gdf.columns else pd.Series(np.nan, index=gdf.index)
        lg_vals = gdf[lg_col] if lg_col in gdf.columns else pd.Series(np.nan, index=gdf.index)

        out[f"{attr} Grade"] = _vec_grade(lg_vals)
        out[f"{attr} %ile"] = lg_vals.round(1)
        out[f"{attr} Grade (Europe)"] = _vec_grade(ov_vals)
        out[f"{attr} %ile (Europe)"] = ov_vals.round(1)

    # ── KPI grade columns (role-specific, populated below) ───────────────
    _all_kpi_cat_names = sorted({cat for kp in _ROLE_KPI_PROFILES.values() for cat in kp})
    for _kcat in _all_kpi_cat_names:
        out[f"{_kcat} %ile"] = np.nan

    # ── Overall grade (vectorized by role group) ─────────────────────────
    out["Overall %ile"] = 0.0
    out["League %ile"] = 0.0

    # Group by role → compute weighted avg for each group
    for role_name, group in gdf.groupby(gdf["_role"]):
        g_idx = group.index
        kpi = _ROLE_KPI_PROFILES.get(role_name)
        if kpi:
            # KPI-profile-based grading
            ov_sum = pd.Series(0.0, index=g_idx)
            lg_sum = pd.Series(0.0, index=g_idx)
            ov_wt = pd.Series(0.0, index=g_idx)
            lg_wt = pd.Series(0.0, index=g_idx)
            for cat_name, (weight, metrics) in kpi.items():
                cat_lg_vals = pd.Series(np.nan, index=g_idx)
                # Europe-wide
                for pos in group["posicion"].unique():
                    pos_mask = group["posicion"] == pos
                    p_idx = group.index[pos_mask]
                    _ov_df = _ov_mpct.get(pos)
                    if _ov_df is not None:
                        common = p_idx.intersection(_ov_df.index)
                        if len(common) > 0:
                            avail = [m for m in metrics if m in _ov_df.columns]
                            if avail:
                                cp = _ov_df.loc[common, avail].mean(axis=1)
                                if cat_name in _KPI_INVERTED_CATS:
                                    cp = 100 - cp
                                ov_sum.loc[common] += cp * weight
                                ov_wt.loc[common] += weight
                    # League-scoped
                    for lg in group.loc[p_idx, "league_display"].unique():
                        _lg_df = _lg_mpct.get((pos, lg))
                        if _lg_df is not None:
                            lg_mask = (group["posicion"] == pos) & (group["league_display"] == lg)
                            lg_idx = group.index[lg_mask].intersection(_lg_df.index)
                            if len(lg_idx) > 0:
                                avail = [m for m in metrics if m in _lg_df.columns]
                                if avail:
                                    cp = _lg_df.loc[lg_idx, avail].mean(axis=1)
                                    if cat_name in _KPI_INVERTED_CATS:
                                        cp = 100 - cp
                                    lg_sum.loc[lg_idx] += cp * weight
                                    lg_wt.loc[lg_idx] += weight
                                    cat_lg_vals.loc[lg_idx] = cp.round(1)
                # Store per-category league-scoped percentile
                out.loc[g_idx, f"{cat_name} %ile"] = cat_lg_vals.values
            out.loc[g_idx, "Overall %ile"] = (ov_sum / ov_wt.replace(0, 1)).round(1)
            out.loc[g_idx, "League %ile"] = (lg_sum / lg_wt.replace(0, 1)).round(1)
        else:
            # Position-weight-based grading
            for pos in group["posicion"].unique():
                pos_mask = group["posicion"] == pos
                p_idx = group.index[pos_mask]
                weights = _POSITION_GRADE_WEIGHTS.get(pos, {})
                attrs = list(GK_ATTRIBUTE_GRADE_CATEGORIES.keys()) if pos == "Goalkeeper" else list(ATTRIBUTE_GRADE_CATEGORIES.keys())
                if weights:
                    ov_num = pd.Series(0.0, index=p_idx)
                    ov_den = pd.Series(0.0, index=p_idx)
                    lg_num = pd.Series(0.0, index=p_idx)
                    lg_den = pd.Series(0.0, index=p_idx)
                    for a in attrs:
                        w = weights.get(a, 0)
                        if w == 0:
                            continue
                        ov_c = f"_ov_{a}"
                        lg_c = f"_lg_{a}"
                        if ov_c in gdf.columns:
                            valid = gdf.loc[p_idx, ov_c].notna()
                            ov_num.loc[valid[valid].index] += gdf.loc[valid[valid].index, ov_c] * w
                            ov_den.loc[valid[valid].index] += w
                        if lg_c in gdf.columns:
                            valid = gdf.loc[p_idx, lg_c].notna()
                            lg_num.loc[valid[valid].index] += gdf.loc[valid[valid].index, lg_c] * w
                            lg_den.loc[valid[valid].index] += w
                    out.loc[p_idx, "Overall %ile"] = (ov_num / ov_den.replace(0, 1)).round(1)
                    out.loc[p_idx, "League %ile"] = (lg_num / lg_den.replace(0, 1)).round(1)
                else:
                    # Equal-weight fallback
                    ov_cols = [f"_ov_{a}" for a in attrs if f"_ov_{a}" in gdf.columns]
                    lg_cols = [f"_lg_{a}" for a in attrs if f"_lg_{a}" in gdf.columns]
                    if ov_cols:
                        out.loc[p_idx, "Overall %ile"] = gdf.loc[p_idx, ov_cols].mean(axis=1).round(1)
                    if lg_cols:
                        out.loc[p_idx, "League %ile"] = gdf.loc[p_idx, lg_cols].mean(axis=1).round(1)

    # ── Key-Strengths uplift (vectorized) ────────────────────────────────
    # Reward elite standout metrics so specialists aren't washed out by the
    # weighted average.  Mirrors _compute_key_strength_bonus (cap +6).
    def _vec_key_bonus(mpct_dict):
        bonus = pd.Series(0.0, index=gdf.index)
        for pos, mpct in mpct_dict.items():
            pool = GK_PIZZA_METRICS if pos == "Goalkeeper" else PIZZA_METRICS
            cols = [c for ml in pool.values() for (_, c) in ml if c in mpct.columns]
            if not cols:
                continue
            sub = mpct[cols]
            b = ((sub >= 95).sum(axis=1) * 2.0
                 + ((sub >= 90) & (sub < 95)).sum(axis=1) * 1.0
                 + ((sub >= 85) & (sub < 90)).sum(axis=1) * 0.5).clip(upper=4.0)
            bonus.loc[sub.index] = b
        return bonus

    out["Overall %ile"] = (out["Overall %ile"] + _vec_key_bonus(_ov_mpct)).clip(upper=99.0).round(1)
    out["League %ile"] = (out["League %ile"] + _vec_key_bonus(_lg_mpct)).clip(upper=99.0).round(1)

    out["Overall Grade"] = _vec_grade(out["Overall %ile"])
    out["League Grade"] = _vec_grade(out["League %ile"])

    # Convert KPI %ile columns → grade strings
    for _kcat in _all_kpi_cat_names:
        _kpct_col = f"{_kcat} %ile"
        out[f"{_kcat} Grade"] = _vec_grade(out[_kpct_col])

    # ── Potential Grade (age-based projection) ───────────────────────────
    _age_col = next((c for c in ("edad", "age", "Age") if c in gdf.columns), None)
    _DEFAULT_AGE = 23
    if _age_col:
        _ages = pd.to_numeric(gdf[_age_col], errors="coerce").fillna(_DEFAULT_AGE).clip(15, 45).astype(int)
    else:
        _ages = pd.Series(_DEFAULT_AGE, index=gdf.index)

    _pot_pcts = []
    for idx in gdf.index:
        _cur_pct = float(out.at[idx, "Overall %ile"]) if idx in out.index else 0.0
        _age = int(_ages.at[idx])
        _projs = _project_potential(_cur_pct, _age, "No major change", years=pot_years)
        _pot_pcts.append(_projs[-1]["pct"] if _projs else _cur_pct)
    out["Potential %ile"] = pd.Series(_pot_pcts, index=gdf.index).round(1)
    out["Potential Grade"] = _vec_grade(out["Potential %ile"])

    # Clean up temp columns
    gdf.drop(columns=["_role"], inplace=True, errors="ignore")

    return out


# ── Value model: over/undervalued vs Transfermarkt ───────────────────────────
# Idea: fit market value on quality, age, minutes, position and league, then the
# residual (actual / model) flags over/undervaluation.  DISABLED: validated on
# real values + ages it proved unreliable — market value bakes in potential,
# hype, contract length and brand that on-pitch output + age can't reconstruct,
# so elite/young/veteran players are badly misvalued (Pedri/Yamal flagged ~8×,
# Kane → €10m) and >½ of all players land in the extreme buckets.  Kept for a
# future, better-specified model (needs potential & contract inputs).  The age &
# market-value data the refresh added still powers Potential Grading and the
# (now fresh, de-collided) market-value display.
_VALUE_MODEL_ENABLED = False


def _mv_rating(ratio):
    if ratio is None or pd.isna(ratio):
        return None
    if ratio < 0.55:
        return "Undervalued ↓↓"
    if ratio < 0.8:
        return "Undervalued ↓"
    if ratio <= 1.25:
        return "Fairly valued"
    if ratio <= 1.8:
        return "Overvalued ↑"
    return "Overvalued ↑↑"


@st.cache_data(ttl=86400, show_spinner=False)
def _compute_value_ratings(_bust):
    """Per-player market-value model for the current season.

    Returns {(nombre, equipo): {actual, predicted, ratio, rating}} in € millions,
    or {} when there isn't enough data (e.g. no market values yet)."""
    if not _VALUE_MODEL_ENABLED:
        return {}
    data = load_data(CURRENT_SEASON)
    total = data.get("total")
    if total is None or total.empty or "market_value_m" not in total.columns:
        return {}

    lab = _build_player_lab_table(total, total, role_schema=_ROLE_SCHEMA_VERSION,
                                  foot_bust=_csv_mtime(_FOOT_CSV))
    df = total.copy()
    df["_q"] = pd.to_numeric(lab["Overall %ile"].reindex(df.index), errors="coerce")
    df["_mv"] = pd.to_numeric(df["market_value_m"], errors="coerce")
    df["_min"] = pd.to_numeric(df.get("estimated_90s"), errors="coerce").fillna(0)
    df["_age"] = pd.to_numeric(df.get("age"), errors="coerce") if "age" in df.columns else np.nan

    use_age = ("age" in df.columns) and (df["_age"].notna().sum() >= 50)
    # Age is essential — without it the model misvalues older stars and can't
    # reach superstar values.  Stay dark until the financials carry ages.
    if not use_age:
        return {}

    # Design matrix over all rows (consistent dummy columns), fit on rows with a value.
    feat = pd.DataFrame({"intercept": 1.0,
                         "quality": df["_q"].fillna(df["_q"].median()),
                         "logmin": np.log1p(df["_min"])}, index=df.index)
    if use_age:
        # Gentle, capped age effect: a single "years past peak (~24)" penalty,
        # squared but bounded.  age² across the raw range over-penalised elite
        # veterans (Kane at 32) and adding interactions destabilised the fit.
        _a = df["_age"].fillna(df["_age"].median()).clip(16, 38)
        feat["age_pen"] = (_a - 24).clip(lower=0)        # only ages past peak cost value
        feat["young"] = (24 - _a).clip(lower=0)          # a small youth/upside premium
    feat = pd.concat([feat,
                      pd.get_dummies(df["posicion"], prefix="pos", drop_first=True),
                      pd.get_dummies(df["league_display"], prefix="lg", drop_first=True)],
                     axis=1)
    X = feat.to_numpy(dtype=float)

    train = df["_mv"].notna() & (df["_mv"] > 0) & df["_q"].notna()
    if use_age:
        train = train & df["_age"].notna()
    if int(train.sum()) < 50:
        return {}

    # Rank-based fit: predict where a player RANKS on value (0-100), then map the
    # predicted rank back to € via the real value quantiles.  A log-€ linear fit
    # can't reach the €100m+ tail and flagged every elite as "overvalued"; the
    # rank model keeps predictions inside the real range so the signal is honest.
    tr = train.to_numpy()
    v = df["_mv"].to_numpy()[tr]
    vrank = np.empty(len(v))
    vrank[v.argsort()] = np.linspace(0.0, 100.0, len(v))
    beta, *_ = np.linalg.lstsq(X[tr], vrank, rcond=None)
    pred_rank = np.clip(X @ beta, 0.5, 99.5)
    pred = np.percentile(v, pred_rank)            # expected € at that rank

    out = {}
    nombre = df["nombre"].to_numpy()
    equipo = df["equipo"].to_numpy()
    mv = df["_mv"].to_numpy()
    for i in range(len(df)):
        if pd.isna(mv[i]) or mv[i] <= 0:
            continue
        ratio = float(mv[i] / pred[i]) if pred[i] > 0 else None
        out[(nombre[i], equipo[i])] = {
            "actual": round(float(mv[i]), 1),
            "predicted": round(float(pred[i]), 1),
            "ratio": round(ratio, 2) if ratio is not None else None,
            "rating": _mv_rating(ratio),
        }
    return out


_STAT_MODES = ["Total", "Per 90"]

_STAT_MODE_MAP = {
    "Total": "total",
    "Per 90": "per90",
}

# When the sidebar "Possession-adjust" toggle is on, raw frames are swapped
# for their possession-adjusted equivalents.
_PADJ_KEY = {"total": "padj", "per90": "padj_per90"}


def _select_df(data, stat_mode):
    """Return the DataFrame for *stat_mode*, possession-adjusted if the toggle is on."""
    key = _STAT_MODE_MAP.get(stat_mode, "total")
    if st.session_state.get("padj_on", False):
        key = _PADJ_KEY.get(key, key)
    return data[key]


def render_player_lab(data, is_current=True):
    df_total = data["total"]
    st.subheader("🔬 Player Lab")
    if is_current:
        st.caption("Filter players by position, role, grade, market value, and salary to find your ideal targets.")
    else:
        st.caption("Filter players by position, role, and grade. (Market value, salary and "
                   "potential grade are current-season only.)")

    # Stat mode selector
    lab_stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="lab_stat_mode")
    grade_df = _select_df(data, lab_stat_mode)

    if lab_stat_mode == "Per 90" and not grade_df.empty:
        _MIN_90S_P90 = 10  # 900 min threshold
        _has_mins = grade_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
        grade_src = grade_df[_has_mins].copy()
    else:
        grade_src = grade_df

    # Potential projection years
    pot_years = st.slider("Potential projection (years)", 1, 5, 3, key="lab_pot_years",
                          help="Number of years ahead to project each player's grade potential.")

    # Build / retrieve the grade table
    with st.spinner("Computing player grades…"):
        lab_df = _build_player_lab_table(grade_src, df_total, mode_label=lab_stat_mode, pot_years=pot_years, role_schema=_ROLE_SCHEMA_VERSION, foot_bust=_csv_mtime(_FOOT_CSV))

    # ── Filters ──────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_leagues = st.multiselect("League", sorted(lab_df["League"].unique()),
                                     default=[], key="lab_leagues")
    with f2:
        pos_options = sorted(lab_df["Position"].unique())
        sel_positions = st.multiselect("Position", pos_options, default=[], key="lab_positions")
    with f3:
        # Roles depend on selected positions
        if sel_positions:
            role_pool = lab_df[lab_df["Position"].isin(sel_positions)]
        else:
            role_pool = lab_df
        # Order roles GK → CB → FB → CM → CAM → Wingers → Striker
        _ROLE_ORDER = [
            # Goalkeepers
            "Shot-Stopper", "Sweeper Keeper", "Ball-Playing Goalkeeper",
            # Centre-Backs
            "Ball Playing", "Duelist", "Libero",
            # Full-Backs
            "Defensive", "Inverted", "All-Around", "Attacking",
            # Central Midfield
            "Defensive Midfielder", "Ball Winning Midfielder", "Deep-Lying Playmaker",
            "Central Midfielder", "Box-to-Box Midfielder",
            # Attacking Midfield
            "Classic 10", "Advanced Playmaker", "Shadow Striker", "Trequartista", "Mezzala",
            # Wingers
            "Inverted Winger", "Classic Winger", "Creative Winger", "Pressing Winger",
            # Strikers
            "Prolific Striker", "Target Man", "Pressing Forward", "False 9",
        ]
        _available_in_pool = set(role_pool["Role"].unique())
        role_options = [r for r in _ROLE_ORDER if r in _available_in_pool] + \
                       sorted(_available_in_pool - set(_ROLE_ORDER))
        sel_roles = st.multiselect("Role", role_options, default=[], key="lab_roles")
        if sel_roles:
            st.caption("Includes versatile players who also fit this role. Grades reflect "
                       "each player's best-fit role in the current stat mode.")
    with f4:
        sel_squad_roles = st.multiselect("Squad Role", ["Starter", "Rotation", "Depth"],
                                         default=[], key="lab_squad_roles",
                                         help="Share of the season's minutes — Starter ≥60% · Rotation 30–60% · Depth <30%")

    # Second filter row: preferred foot + top strength
    g1, g2 = st.columns(2)
    with g1:
        _foot_opts = [f for f in ["Right", "Left", "Two-footed"]
                      if (lab_df["Foot"] == f).any()]
        sel_feet = st.multiselect("Preferred foot", _foot_opts, default=[], key="lab_feet",
                                  help="Player's stronger foot (via Transfermarkt).")
    with g2:
        _strength_pool = lab_df[lab_df["Position"].isin(sel_positions)] if sel_positions else lab_df
        _strength_opts = sorted(s for s in _strength_pool["_top_metric"].dropna().unique() if s)
        sel_strengths = st.multiselect("Top strength", _strength_opts, default=[], key="lab_strengths",
                                       help="Filter to players whose #1 percentile metric is one of these.")

    # Grade range selector
    grade_col, grade_type_col = st.columns([3, 1])
    with grade_type_col:
        _attr_grade_cols = [c for c in lab_df.columns if c.endswith(" Grade") and c not in ("Overall Grade", "League Grade", "Potential Grade")]
        _grade_basis_opts = ["Overall Grade", "League Grade"]
        if is_current:
            _grade_basis_opts.append("Potential Grade")
        _grade_basis_opts += _attr_grade_cols
        grade_basis = st.selectbox("Grade type", _grade_basis_opts,
            key="lab_grade_type",
            help="Individual attribute grades are vs same-league peers."
                 + (" 'Potential Grade' projects each player's grade forward using an age-based development curve." if is_current else ""))
    with grade_col:
        min_idx, max_idx = st.select_slider(
            "Grade range",
            options=list(range(len(_ALL_GRADES))),
            value=(0, len(_ALL_GRADES) - 1),
            format_func=lambda i: _ALL_GRADES[i],
            key="lab_grade_range",
        )
        min_grade_idx, max_grade_idx = min_idx, max_idx

    # Market Value & Salary sliders (current season only)
    if is_current:
        val1, val2 = st.columns(2)
        with val1:
            mv_range = st.slider("Transfermarkt Value (€M)", 0.0, 250.0, (0.0, 250.0),
                                  step=1.0, key="lab_mv_range")
        with val2:
            sal_range = st.slider("Gross Annual Salary (€M)", 0.0, 60.0, (0.0, 60.0),
                                   step=0.5, key="lab_sal_range")
        enable_financials = (mv_range != (0.0, 250.0)) or (sal_range != (0.0, 60.0))
    else:
        mv_range, sal_range = (0.0, 250.0), (0.0, 60.0)
        enable_financials = False

    # ── Apply filters ────────────────────────────────────────────────────
    filtered = lab_df.copy()
    if sel_leagues:
        filtered = filtered[filtered["League"].isin(sel_leagues)]
    if sel_positions:
        filtered = filtered[filtered["Position"].isin(sel_positions)]
    if sel_roles:
        # Match the player's primary role OR any role they also fit (eligibility),
        # so versatile players surface under every role they're strong in.
        _sel_set = set(sel_roles)
        if "_eligible" in filtered.columns:
            _elig_mask = filtered["_eligible"].apply(lambda rs: bool(_sel_set & set(rs)))
            filtered = filtered[filtered["Role"].isin(sel_roles) | _elig_mask]
        else:
            filtered = filtered[filtered["Role"].isin(sel_roles)]
    if sel_squad_roles:
        filtered = filtered[filtered["Squad Role"].isin(sel_squad_roles)]
    if sel_feet:
        filtered = filtered[filtered["Foot"].isin(sel_feet)]
    if sel_strengths:
        filtered = filtered[filtered["_top_metric"].isin(sel_strengths)]

    # Grade filter
    filtered["_grade_idx"] = filtered[grade_basis].map(_GRADE_TO_IDX)
    filtered = filtered[
        (filtered["_grade_idx"] >= min_grade_idx) & (filtered["_grade_idx"] <= max_grade_idx)
    ]
    filtered = filtered.drop(columns=["_grade_idx"])

    # ── Financial filters (from pre-built CSV) ───────────────────────────
    if enable_financials:
        filtered["_mv_num"] = filtered["Market Value"].apply(_parse_market_value)
        filtered["_sal_num"] = filtered["Salary"].apply(_parse_salary_value)

        if mv_range != (0.0, 250.0):
            mv_min, mv_max = mv_range[0] * 1_000_000, mv_range[1] * 1_000_000
            filtered = filtered[
                filtered["_mv_num"].isna()  # keep players with no data
                | (
                    (filtered["_mv_num"] >= mv_min)
                    & (filtered["_mv_num"] <= mv_max)
                )
            ]
        if sal_range != (0.0, 60.0):
            sal_min, sal_max = sal_range[0] * 1_000_000, sal_range[1] * 1_000_000
            filtered = filtered[
                filtered["_sal_num"].isna()  # keep players with no data
                | (
                    (filtered["_sal_num"] >= sal_min)
                    & (filtered["_sal_num"] <= sal_max)
                )
            ]
        filtered = filtered.drop(columns=["_mv_num", "_sal_num"])

    # ── Sort & display ───────────────────────────────────────────────────
    pct_col = grade_basis.replace(" Grade", " %ile")
    if pct_col not in filtered.columns:
        pct_col = "League %ile"
    filtered = filtered.sort_values(pct_col, ascending=False)

    st.markdown(f"**{len(filtered)}** players match your filters")

    # Determine which grade columns to show: role-specific KPI when a single role
    # is filtered, otherwise generic attribute grades (Attacking, Defending, etc.)
    _generic_attr_cats = set(ATTRIBUTE_GRADE_CATEGORIES.keys()) | set(GK_ATTRIBUTE_GRADE_CATEGORIES.keys())
    _generic_attr_display = [c for c in filtered.columns
                             if c.endswith(" Grade")
                             and c not in ("Overall Grade", "League Grade", "Potential Grade")
                             and not c.endswith(" Grade (Europe)")
                             and c.replace(" Grade", "") in _generic_attr_cats]
    _kpi_attr_display = []
    if sel_roles and len(sel_roles) == 1:
        _single_role_kpi = _ROLE_KPI_PROFILES.get(sel_roles[0])
        if _single_role_kpi:
            _kpi_attr_display = [f"{cat} Grade" for cat in _single_role_kpi.keys()
                                  if f"{cat} Grade" in filtered.columns]
    _attr_display = _kpi_attr_display if _kpi_attr_display else _generic_attr_display
    # Financial columns are current-season only.
    _fin_cols = ["Market Value", "Salary"] if is_current else []
    # When a specific league is selected, hide the Overall columns
    _show_potential = is_current and grade_basis == "Potential Grade"
    if sel_leagues:
        display_cols = ["Player", "Team", "League", "Pos", "Foot", "Squad Role",
                        "Top Strength", "League Grade", "League %ile"]
        if _show_potential:
            display_cols += ["Potential Grade", "Potential %ile"]
        display_cols += _attr_display + _fin_cols
    else:
        display_cols = ["Player", "Team", "League", "Pos", "Foot", "Squad Role",
                        "Top Strength", "Overall Grade", "Overall %ile",
                        "League Grade", "League %ile"]
        if _show_potential:
            display_cols += ["Potential Grade", "Potential %ile"]
        display_cols += _attr_display + _fin_cols
    show = filtered[[c for c in display_cols if c in filtered.columns]].reset_index(drop=True)
    st.dataframe(show, use_container_width=True, height=600)


# ── UI: Player Profile ──────────────────────────────────────────────────────

def render_profile(data, is_current=True):
    df_total = data["total"]
    df_per90 = data["per90"]
    st.subheader("🪪 Player Profile")

    # Selectors
    c1, c2 = st.columns(2)
    with c1:
        league_sel = st.selectbox("League", ["All"] + sorted(df_total["league_display"].unique()), key="prof_lg")
    with c2:
        pool = df_total if league_sel == "All" else df_total[df_total["league_display"] == league_sel]
        _opts = _player_options(pool)
        _labels = [o[0] for o in _opts]
        _sel_label = st.selectbox("Player", _labels, index=None,
                                  placeholder="Select a player…", key="prof_pl")

    if not _sel_label:
        return

    _, player_sel, _sel_team = _opts[_labels.index(_sel_label)]
    _prow = pool[(pool["nombre"] == player_sel) & (pool["equipo"] == _sel_team)]
    if _prow.empty:
        _prow = pool[pool["nombre"] == player_sel]
    row = _prow.iloc[0]
    _orig_position = row.get("posicion", "Unknown")
    pos_detail = row.get("posicion_detail", "")
    league = row.get("league_display", "")

    # ── Position / Role Override ─────────────────────────────────────────
    _all_positions = sorted(POSITION_ROLE_PROFILES.keys())
    _orig_idx = _all_positions.index(_orig_position) if _orig_position in _all_positions else 0
    ov1, ov2 = st.columns(2)
    with ov1:
        position = st.selectbox(
            "Position",
            _all_positions,
            index=_orig_idx,
            key="prof_pos_override",
            help="Override when a player's Opta position doesn't match their current role (e.g. formation change).",
        )
    _position_changed = position != _orig_position
    # Default role reflects the active stat lens (mode + possession-adjust) so the
    # Profile default and the Player Lab agree on a player's best-fit role — a
    # deep mid can be a Deep-Lying Playmaker on totals but a Ball-Winner on
    # per-90 possession-adjusted.  The user can still override below.
    _lens_df = _select_df(data, st.session_state.get("prof_stat_mode", "Total"))
    _lens_prow = _lens_df[(_lens_df["nombre"] == player_sel) & (_lens_df["equipo"] == _sel_team)]
    _role_src_row = _lens_prow.iloc[0] if not _lens_prow.empty else row
    role = _classify_role(_role_src_row, position, _lens_df)
    _available_roles = list(POSITION_ROLE_PROFILES.get(position, {}).keys())
    with ov2:
        if _available_roles:
            _role_idx = _available_roles.index(role) if role in _available_roles else 0
            role = st.selectbox("Role", _available_roles, index=_role_idx, key="prof_role_override")
        else:
            st.selectbox("Role", [role], key="prof_role_override")
    if _position_changed:
        st.info(f"⚠️ Position overridden from **{_orig_position}** → **{position}**. "
                f"Grades now compare against {position} peers.")

    # ── Data Controls ────────────────────────────────────────────────────
    st.markdown("---")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="prof_stat_mode")
    with ctrl2:
        scope_mode = st.radio("Scope", ["League", "Across Europe"], horizontal=True, key="prof_scope")
    with ctrl3:
        basis_mode = st.radio("Basis", ["Position", "Role"], horizontal=True, key="prof_basis")
    use_per90 = stat_mode == "Per 90"
    _active_df = _select_df(data, stat_mode)

    if use_per90 and not _active_df.empty:
        # Filter out low-minutes players whose inflated per-90 rates
        # would skew the percentile rankings for regular starters.
        _MIN_90S_P90 = 10  # 900 min threshold
        _is_sel = (_active_df["nombre"] == player_sel) & (_active_df["equipo"] == _sel_team)
        _has_mins = _active_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
        p90_filtered = _active_df[_is_sel | _has_mins]
        peers_all = p90_filtered.copy()
        p90_match = p90_filtered[(p90_filtered["nombre"] == player_sel) & (p90_filtered["equipo"] == _sel_team)]
        row_data = dict(p90_match.iloc[0]) if not p90_match.empty else dict(row)
        grade_df = p90_filtered
        grade_row = row_data
    elif stat_mode == "Total":
        _src = _active_df
        peers_all = _src.copy()
        row_match = _src[(_src["nombre"] == player_sel) & (_src["equipo"] == _sel_team)]
        row_data = dict(row_match.iloc[0]) if not row_match.empty else dict(row)
        grade_df = _src
        grade_row = row_data
    else:
        peers_all = df_total.copy()
        row_data = dict(row)
        grade_df = df_total
        grade_row = dict(row)

    # ── Compute grades ───────────────────────────────────────────────────
    _scope_league = league if scope_mode == "League" else None
    _basis_role = role if basis_mode == "Role" else None
    attr_grades = _compute_attribute_grades(grade_row, position, grade_df,
                                            league=_scope_league, role=_basis_role,
                                            df_role_ref=df_total, kpi_role=role)
    _attr_names = list(attr_grades.keys())
    _n_attrs = len(_attr_names)

    # ── Overall grade (weighted avg of category percentiles) ─────────
    if position == "Goalkeeper":
        _ov_weights = _ROLE_GRADE_WEIGHTS.get(role, {})
    else:
        _kpi_prof = _ROLE_KPI_PROFILES.get(role) if role else None
        if _kpi_prof:
            _ov_weights = {name: w for name, (w, _) in _kpi_prof.items()}
        else:
            _ov_weights = _POSITION_GRADE_WEIGHTS.get(position, {})
    _ov_pcts = {k: pct for k, (_, pct) in attr_grades.items() if pct is not None}
    if _ov_pcts and _ov_weights:
        _w_sum = sum(_ov_weights.get(k, 0) for k in _ov_pcts)
        if _w_sum > 0:
            _overall_pct = sum(_ov_pcts[k] * _ov_weights.get(k, 0) for k in _ov_pcts) / _w_sum
        else:
            _overall_pct = sum(_ov_pcts.values()) / len(_ov_pcts)
    elif _ov_pcts:
        _overall_pct = sum(_ov_pcts.values()) / len(_ov_pcts)
    else:
        _overall_pct = 0.0
    _overall_grade = _percentile_to_grade(_overall_pct)

    # ── Exceptional Contribution bonus ────────────────────────────────
    _exc_label, _exc_avg_pct, _exc_bonus_pts, _exc_tier = _compute_exceptional_contribution(
        grade_row, position, role, peers_all
    )
    # Key-Strengths uplift — reward elite standout metrics; combined uplift capped at +8.
    _ks_bonus = _compute_key_strength_bonus(
        grade_row, peers_all[peers_all["posicion"] == position], position == "Goalkeeper"
    )
    _total_bonus = min(8.0, _exc_bonus_pts + _ks_bonus)
    _display_pct = round(min(99.9, _overall_pct + _total_bonus), 1) if _total_bonus > 0 else _overall_pct
    _display_grade = _percentile_to_grade(_display_pct) if _total_bonus > 0 else _overall_grade

    # ── Squad Role (share of the season's minutes) ────────────────────────
    _player_mins = row.get("Time Played", 0) or 0
    _season_mins = _league_season_minutes(df_total).get(league, 34 * 90)
    _squad_role = _squad_role_label(_player_mins, _season_mins)

    # Preferred foot (Transfermarkt, by id) — shown in the header if available.
    _foot = _load_footedness_csv(_bust=_csv_mtime(_FOOT_CSV)).get(row.get("id"))
    _foot_lbl = {"left": "🦶 Left-footed", "right": "🦶 Right-footed",
                 "both": "🦶 Two-footed"}.get(_foot)
    _foot_html = f" &middot; {_foot_lbl}" if _foot_lbl else ""

    # ── Build tooltip with sub-grade breakdown ────────────────────────
    _sub_lines = "&#10;".join(
        f"{attr}: {g} ({_ordinal(pct)})" for attr, (g, pct) in attr_grades.items() if pct is not None
    )
    _ov_color = _GRADE_COLORS.get(_display_grade, "#888")
    _stat_ctx = stat_mode if stat_mode != "Total" else "Season Totals"
    _scope_ctx = league if scope_mode == "League" else "All Europe"
    _basis_ctx = f"{role}s" if basis_mode == "Role" else f"{position}s"

    # ── Header Card: [Photo | Name + Overall Grade] ────────────────────
    _photo_amb = row.get("nombre") in _ambiguous_names(_bust=_data_fingerprint())
    player_photo = _fetch_player_photo(row.get("nombre", "?"), team=row.get("equipo"),
                                       ambiguous=_photo_amb)
    _grade_title = "Overall Grade ⭐" if _total_bonus > 0 else "Overall Grade"
    _pctl_suffix = f" (+{_total_bonus:.1f} bonus)" if _total_bonus > 0 else ""
    _pos_was = f" <em style='color:#aaa;'>(was {_orig_position})</em>" if _position_changed else ""
    _photo_html = (
        f"<img src='{player_photo}' width='130' style='border-radius:8px;flex-shrink:0;object-fit:cover;'/>"
        if player_photo else
        f"<div style='width:130px;height:130px;border-radius:50%;background:#2d6a4f;"
        f"display:flex;align-items:center;justify-content:center;font-size:48px;"
        f"color:white;flex-shrink:0;'>{row.get('nombre', '?')[0]}</div>"
    )
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;gap:16px;margin-bottom:4px;'>"
        f"{_photo_html}"
        f"<div style='display:flex;flex-direction:column;justify-content:flex-start;'>"
        f"<div style='display:flex;align-items:flex-start;gap:24px;'>"
        f"<div>"
        f"<div style='font-size:1.8rem;font-weight:700;'>{row.get('nombre', '?')}</div>"
        f"<div style='margin-top:2px;'><strong>{league}</strong></div>"
        f"<div style='margin-top:6px;font-size:0.92rem;'>"
        f"<strong>Position:</strong> {pos_detail} ({position}){_pos_was}"
        f" &middot; <strong>Role:</strong> {role}</div>"
        f"<div style='margin-top:3px;font-size:0.92rem;color:#aaa;'>"
        f"<strong style='color:#ccc;'>Squad Role:</strong> {_squad_role}"
        f" &middot; <strong style='color:#ccc;'>Minutes:</strong> {int(_player_mins):,}{_foot_html}</div>"
        f"</div>"
        f"<span style='display:inline-flex;flex-direction:column;align-items:center;'>"
        f"<span style='font-size:12px;color:#aaa;display:flex;align-items:center;gap:4px;'>"
        f"{_grade_title}"
        f"<span title='{_sub_lines}' style='cursor:help;display:inline-flex;align-items:center;"
        f"justify-content:center;width:15px;height:15px;border-radius:50%;"
        f"background:#444;color:#ccc;font-size:10px;font-weight:bold;line-height:1;'>"
        f"?</span></span>"
        f"<span style='font-size:72px;font-weight:bold;color:{_ov_color};line-height:1;'>{_display_grade}</span>"
        f"<span style='font-size:11px;color:#777;'>{_ordinal(_display_pct)} pctl{_pctl_suffix}</span>"
        f"</span></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if _exc_bonus_pts > 0:
        st.markdown(
            f"<div style='margin-top:10px;background:rgba(255,215,64,0.10);"
            f"border-left:3px solid #ffd740;border-radius:6px;padding:8px 14px;"
            f"font-size:13px;color:#eee;'>"
            f"&#11088; <strong>Exceptional Contribution</strong> &mdash; "
            f"<strong>{_exc_label}</strong>"
            f"&nbsp;&middot;&nbsp;<span style='color:#ffd740;font-weight:bold;'>"
            f"{_exc_tier} (+{_exc_bonus_pts} pts)</span>"
            f"&nbsp;&middot;&nbsp;{_ordinal(_exc_avg_pct)} pctl vs {position}s"
            f"<br><span style='font-size:11px;color:#888;'>"
            f"Base grade: {_overall_grade} ({_ordinal(_overall_pct)} pctl) "
            f"&#8594; boosted to {_display_grade} ({_ordinal(_display_pct)} pctl)"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Grade: {_stat_ctx} · vs {_basis_ctx} in {_scope_ctx}")
    if _ks_bonus > 0:
        st.caption(
            f"⭐ Key-Strengths uplift: +{_ks_bonus:.1f} pctl for elite standout metrics "
            f"(see Standout Strengths below)."
        )

    # Contextual nudge: defenders/DMs on dominant teams are under-credited on
    # raw defensive volume — point the user to the Possession-adjust toggle.
    _DEF_NUDGE_POSITIONS = {"Centre-Back", "Full-Back", "Central Midfield"}
    if position in _DEF_NUDGE_POSITIONS and not st.session_state.get("padj_on", False):
        st.caption(
            "💡 For a defender/midfielder, try **⚖️ Possession-adjust stats** in the "
            "sidebar — a fairer cross-team comparison of defensive work (players on "
            "dominant sides face fewer defensive actions)."
        )

    # ── Career trajectory sparkline (overall %ile across seasons) ─────────
    _pid = row.get("id")
    _traj = _build_trajectory(_pid, _bust=_data_fingerprint()) if (_pid is not None and not pd.isna(_pid)) else []
    _rel_traj = [t for t in _traj if t["reliable"] and t["pct"] is not None]
    if len(_rel_traj) >= 2:
        _spx = [f"{t['season'][2:4]}-{t['season'][7:9]}" for t in _rel_traj]
        _spy = [t["pct"] for t in _rel_traj]
        _cur_season = st.session_state.get("sel_season", CURRENT_SEASON)
        _cur_lbl = f"{_cur_season[2:4]}-{_cur_season[7:9]}"
        _spark = go.Figure(go.Scatter(
            x=_spx, y=_spy, mode="lines+markers+text",
            text=[f"{v:.0f}" for v in _spy], textposition="top center",
            textfont=dict(size=10, color="#9aa7b8"),
            line=dict(color="#52b788", width=2),
            marker=dict(size=[11 if x == _cur_lbl else 7 for x in _spx],
                        color=["#f4a261" if x == _cur_lbl else "#52b788" for x in _spx]),
            hovertemplate="<b>%{x}</b><br>Overall %ile: %{y:.0f}<extra></extra>",
        ))
        _spark.update_layout(
            height=140, margin=dict(t=28, b=24, l=34, r=20),
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font=dict(color="#ccc"),
            title=dict(text="📈 Career trajectory — overall %ile by season",
                       font=dict(size=12, color="#aaa")),
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(_spark, use_container_width=True)

    # ── Market Value & Salary (current season only) ──────────────────────
    if is_current:
        _player_team = row.get("equipo")
        _fin = _fin_rec(_load_financials_csv(_bust=_csv_mtime(_FINANCIALS_CSV)),
                        row["nombre"], _player_team)
        market_val = _fin.get("market_value")
        salary_val = _fin.get("salary")
        # Fall back to a live (cached) fetch for any value missing from the CSV —
        # ~72% of market values were never captured when the CSV was built, so this
        # fills the gaps for players present-but-empty, not just absent ones.
        if market_val is None or salary_val is None:
            _full_name = _resolve_full_name(row["nombre"], team=_player_team)
            if market_val is None:
                market_val = _fetch_transfermarkt_value(_full_name, team=_player_team)
            if salary_val is None:
                salary_val = _fetch_capology_salary(_full_name, team=_player_team)
        mv_col, sal_col = st.columns(2)
        with mv_col:
            st.metric("💰 Transfermarkt Market Value", market_val or "N/A")
        with sal_col:
            st.metric("💶 Gross Annual Salary (Capology)", salary_val or "N/A")

        # ── Value vs model (over/undervalued) — needs age, so only shows once
        # the financials have been refreshed with ages. ──
        _vr = _compute_value_ratings(_data_fingerprint()).get((row["nombre"], _player_team))
        if _vr and _vr.get("rating"):
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.metric("Model Expected Value", f"€{_vr['predicted']:.0f}m",
                          help="Predicted from quality, age, minutes, position and "
                               "league (log-linear fit on Transfermarkt values).")
            with vc2:
                _r = _vr["ratio"]
                st.metric("Value Rating", _vr["rating"],
                          delta=f"{_r:.2f}× model", delta_color="off",
                          help="Actual ÷ model. <1 = market prices him below his "
                               "on-pitch profile (potentially undervalued).")
            with vc3:
                st.metric("Actual ÷ Expected", f"{_vr['ratio']:.2f}×")
            st.caption("⚖️ Value rating is the market's premium/discount vs a model of "
                       "players with a similar profile — a scouting signal, not a verdict. "
                       "Driven heavily by age & league; contract length isn't modelled.")

    # Scope: league-only or all leagues
    if scope_mode == "League":
        peers = peers_all[peers_all["league_display"] == league]
    else:
        peers = peers_all.copy()

    # When position is overridden to one with no players in the data (e.g.
    # "Wingers"), use a related position for peer lookups.
    _PEER_POS_FALLBACK = {"Wingers": "Striker"}
    _peer_pos = position
    if len(peers[peers["posicion"] == position]) < 5 and position in _PEER_POS_FALLBACK:
        _peer_pos = _PEER_POS_FALLBACK[position]

    # Basis: position-generic categories or role-filtered categories
    _is_gk = position == "Goalkeeper"
    _POS_CATS_RENDER = {
        "Goalkeeper": GK_PROFILE_CATEGORIES,
        "Striker": STRIKER_PROFILE_CATEGORIES,
        "Wingers": WINGER_PROFILE_CATEGORIES,
        "Attacking Midfield": AM_PROFILE_CATEGORIES,
    }
    base_cats = _POS_CATS_RENDER.get(position, PROFILE_CATEGORIES)
    if basis_mode == "Role":
        role_profiles = POSITION_ROLE_PROFILES.get(position, {})
        role_metrics = role_profiles.get(role)
        if role_metrics:
            role_set = set(role_metrics)
            _prof_cats_sel = {}
            for cat, cat_metrics in base_cats.items():
                filtered = [m for m in cat_metrics if m in role_set]
                if filtered:
                    _prof_cats_sel[cat] = filtered
            if not _prof_cats_sel:
                _prof_cats_sel = base_cats
        else:
            _prof_cats_sel = base_cats
    else:
        _prof_cats_sel = base_cats

    scope_label = f" ({league})" if scope_mode == "League" else " (Europe)"
    basis_label = f" [{role}]" if basis_mode == "Role" else ""
    mode_label = (f" ({stat_mode})" if stat_mode != "Total" else "") + scope_label + basis_label

    # ── Standout Strengths ───────────────────────────────────────────────
    # Lead with the player's elite traits — surfaced at the metric level so a
    # standout (e.g. a GK's Command / Big Chances Saved) shows even when the
    # parent category grades as average.
    _strength_peers = peers[peers["posicion"] == _peer_pos]
    _strengths = _compute_standout_strengths(row_data, _strength_peers, _is_gk)
    if _strengths:
        st.markdown("---")
        st.markdown(f"### 🌟 Standout Strengths{scope_label}")
        st.caption(f"Where **{row['nombre']}** ranks among {_peer_pos}s — percentile vs peers.")
        _scols = st.columns(len(_strengths))
        for _sc, _s in zip(_scols, _strengths):
            _v = _s["value"]
            _vstr = f"{_v:.1f}" if isinstance(_v, (int, float)) else str(_v)
            with _sc:
                st.markdown(
                    f"<div style='background:#1a1a2e;border-left:4px solid {_s['color']};"
                    f"border-radius:8px;padding:12px 14px;'>"
                    f"<div style='font-size:12px;color:#aaa;height:32px;'>{_s['label']}</div>"
                    f"<div style='font-size:30px;font-weight:bold;color:{_s['color']};line-height:1.1;'>"
                    f"{_s['pct']:.0f}<span style='font-size:13px;'>{_ord_suffix(_s['pct'])}</span></div>"
                    f"<div style='font-size:11px;color:#ccc;'>{_s['tier']} · {_vstr}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Pizza Chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🍕 Percentile Pizza Chart{mode_label}")
    fig_pizza = _build_pizza_chart(row_data, peers, row["nombre"], position, is_gk=_is_gk)
    if fig_pizza:
        st.plotly_chart(fig_pizza, use_container_width=True)
    else:
        st.info("Not enough data for the pizza chart.")

    # ── GK PSxG Statline ─────────────────────────────────────────────────
    if _is_gk and row_data.get("PSxG") is not None and not pd.isna(row_data.get("PSxG", float("nan"))):
        st.markdown("---")
        st.markdown(f"### 🎯 Post-Shot Expected Goals (PSxG){scope_label}")
        st.caption(
            "PSxG is approximated from on-target shots faced, weighted by location "
            "(inside box × 0.34, big chances × 0.55, outside box × 0.12, penalties × 0.79). "
            "Percentile ranks are vs same-scope Goalkeepers."
        )

        gk_peers_psxg = peers[peers["posicion"] == "Goalkeeper"]

        def _gk_pct_rank(val, col):
            if val is None or pd.isna(val) or col not in gk_peers_psxg.columns:
                return None
            peer_vals = gk_peers_psxg[col].fillna(0)
            return round((peer_vals < val).sum() / max(len(peer_vals), 1) * 100, 1)

        psxg_val       = row_data.get("PSxG") or 0.0
        psxg_pm_val    = row_data.get("PSxG+/-")
        psxg_shot_val  = row_data.get("PSxG/Shot")
        soft_ga_val    = row_data.get("Saveable Goals Conceded")
        _has_measured_psxg = soft_ga_val is not None and not pd.isna(soft_ga_val)
        ga_val         = row_data.get("Goals Conceded") or 0
        saves_val      = row_data.get("Saves Made") or 0
        shots_faced    = saves_val + ga_val

        ib_shots  = ((row_data.get("Saves Made from Inside Box") or 0)
                     + (row_data.get("Goals Conceded Inside Box") or 0))
        ob_shots  = ((row_data.get("Saves Made from Outside Box") or 0)
                     + (row_data.get("Goals Conceded Outside Box") or 0))
        ib_saves  = row_data.get("Saves Made from Inside Box") or 0
        ob_saves  = row_data.get("Saves Made from Outside Box") or 0
        ib_ga     = row_data.get("Goals Conceded Inside Box") or 0
        ob_ga     = row_data.get("Goals Conceded Outside Box") or 0
        ib_sv_pct = row_data.get("Inside Box Save %")
        ob_sv_pct = row_data.get("Outside Box Save %")
        sv_pct    = row_data.get("Save %")
        _bc_raw    = row_data.get("Total Big Chances Saved")
        bc_missing = _bc_raw is None or pd.isna(_bc_raw)   # not tracked (e.g. older seasons)
        bc_saved   = 0 if bc_missing else (_bc_raw or 0)
        pens_faced = 0 if pd.isna(_v := row_data.get("Penalties Faced"))         else (_v or 0)
        pens_saved = 0 if pd.isna(_v := row_data.get("Penalties Saved"))         else (_v or 0)
        pen_sv_pct = round(pens_saved / pens_faced * 100, 1) if pens_faced > 0 else None

        _pm_rank   = _gk_pct_rank(psxg_pm_val, "PSxG+/-")
        _sv_rank   = _gk_pct_rank(sv_pct, "Save %")
        _shot_rank = _gk_pct_rank(psxg_shot_val, "PSxG/Shot")

        def _pct_delta(pct):
            return f"{_ordinal(pct)} %ile" if pct is not None else None

        # ── Top-row summary metrics ────────────────────────────────────
        _mcols = st.columns(6) if _has_measured_psxg else st.columns(5)
        with _mcols[0]:
            st.metric("Shots Faced", int(shots_faced))
        with _mcols[1]:
            _psxg_label = "PSxG" if _has_measured_psxg else "PSxG (approx.)"
            _psxg_help = ("Post-Shot xG from a shot-level model (goalmouth placement, "
                          "distance, angle, body part)." if _has_measured_psxg
                          else "Approximated Post-Shot xG using shot-location weights.")
            st.metric(_psxg_label, f"{psxg_val:.1f}", help=_psxg_help)
        with _mcols[2]:
            st.metric("Goals Against", int(ga_val))
        with _mcols[3]:
            pm_str = f"+{psxg_pm_val:.1f}" if (psxg_pm_val is not None and psxg_pm_val >= 0) else (f"{psxg_pm_val:.1f}" if psxg_pm_val is not None else "—")
            st.metric("PSxG+/- (Goals Prevented)", pm_str,
                      delta=_pct_delta(_pm_rank),
                      help="Positive = conceded fewer goals than shot-quality suggests.")
        # Saveable Goals sits right next to PSxG+/- as its counterpoint: soft
        # goals the +/- nets away.  Only shown when the measured model covers
        # this keeper-season.
        if _has_measured_psxg:
            with _mcols[4]:
                _soft_rank = _gk_pct_rank(soft_ga_val, "Saveable Goals Conceded")
                _soft_delta = (f"{_ordinal(100 - _soft_rank)} %ile" if _soft_rank is not None else None)
                st.metric("Saveable GA", int(soft_ga_val), delta=_soft_delta,
                          delta_color="off",
                          help="Goals conceded from low-difficulty shots (post-shot xG "
                               "< 0.20) — soft goals PSxG+/- nets away. Fewer is better; "
                               "percentile shown is rank among GKs (higher = fewer).")
        with _mcols[5 if _has_measured_psxg else 4]:
            sv_str = f"{sv_pct:.1f}%" if sv_pct is not None else "—"
            st.metric("Save %", sv_str, delta=_pct_delta(_sv_rank))

        # ── Shot-location breakdown table ─────────────────────────────
        st.markdown("**📐 Shot Location Breakdown**")
        loc_rows = [
            {
                "Zone": "Inside Box",
                "Shots Faced": int(ib_shots),
                "Saves": int(ib_saves),
                "Goals Against": int(ib_ga),
                "Save %": f"{ib_sv_pct:.1f}%" if ib_sv_pct is not None else "—",
            },
            {
                "Zone": "Outside Box",
                "Shots Faced": int(ob_shots),
                "Saves": int(ob_saves),
                "Goals Against": int(ob_ga),
                "Save %": f"{ob_sv_pct:.1f}%" if ob_sv_pct is not None else "—",
            },
            {
                "Zone": "Total",
                "Shots Faced": int(shots_faced),
                "Saves": int(saves_val),
                "Goals Against": int(ga_val),
                "Save %": f"{sv_pct:.1f}%" if sv_pct is not None else "—",
            },
        ]
        st.dataframe(pd.DataFrame(loc_rows).set_index("Zone"), use_container_width=True)

        # ── Penalty & big-chance rows ──────────────────────────────────
        pen_col, bc_col, psxg_shot_col = st.columns(3)
        with pen_col:
            st.markdown("**🥅 Penalties**")
            pen_df = pd.DataFrame([{
                "Faced": int(pens_faced),
                "Saved": int(pens_saved),
                "Save %": f"{pen_sv_pct:.1f}%" if pen_sv_pct is not None else "—",
            }])
            st.dataframe(pen_df, use_container_width=True, hide_index=True)
        with bc_col:
            st.markdown("**⚡ Big Chances**")
            # Show "—" when the season's feed doesn't track this stat (older
            # seasons) rather than a misleading "0".
            bc_df = pd.DataFrame([{"Big Chances Saved": "—" if bc_missing else int(bc_saved)}])
            st.dataframe(bc_df, use_container_width=True, hide_index=True)
            if bc_missing:
                st.caption("Not tracked this season — PSxG is unaffected (it's measured "
                           "from shot locations in the event data, not this stat).")
        with psxg_shot_col:
            st.markdown("**📊 Shot Quality**")
            sq_str = f"{psxg_shot_val:.3f}" if psxg_shot_val is not None else "—"
            sq_rank = _pct_delta(_shot_rank)
            sq_df = pd.DataFrame([{
                "PSxG/Shot": sq_str,
                "vs GK Peers": sq_rank or "—",
            }])
            st.dataframe(sq_df, use_container_width=True, hide_index=True)
        if _has_measured_psxg:
            st.caption("🧤 **Saveable GA** (top row) = goals conceded from low-difficulty "
                       "shots (post-shot xG < 0.20) — soft goals that PSxG+/- nets away.")

    # ── Outfield xG / xA statline ────────────────────────────────────────
    _npxg_val = row_data.get("npxG")
    if not _is_gk and _npxg_val is not None and not pd.isna(_npxg_val):
        st.markdown("---")
        st.markdown(f"### ⚽ Goals, Assists & Expected (xG / xA){mode_label}")
        st.caption("Top row = actual output. Below: **npxG** = chance quality (shooting). "
                   "**Finishing (npG−xG)** = "
                   "non-penalty goals minus xG (clinical vs wasteful). **xA** = expected "
                   "assists, the xG of chances created. **npxG+xA** = total expected "
                   "involvement. Percentiles are vs same-position peers.")
        _pos_peers_xg = peers[peers["posicion"] == position]

        def _xg_pct(val, col):
            if val is None or pd.isna(val) or col not in _pos_peers_xg.columns:
                return None
            pv = _pos_peers_xg[col].fillna(0)
            return round((pv < val).sum() / max(len(pv), 1) * 100, 1)

        def _pctd(val, col):
            p = _xg_pct(val, col)
            return f"{_ordinal(p)} %ile" if p is not None else None

        def _num(v):
            """Format a count: integer in Total mode, decimals in Per-90."""
            if v is None or pd.isna(v):
                return "—"
            return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"

        # ── Actual output: goals & assists ───────────────────────────────
        _goals_val = row_data.get("Goals")
        _ast_val = row_data.get("Goal Assists")
        _npg_val = row_data.get("Non-Penalty Goals")
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            st.metric("Goals", _num(_goals_val), delta=_pctd(_goals_val, "Goals"),
                      help="Total goals scored (includes penalties).")
        with oc2:
            st.metric("Assists", _num(_ast_val), delta=_pctd(_ast_val, "Goal Assists"),
                      help="Goals assisted (the pass that directly led to a goal).")
        with oc3:
            st.metric("Non-Pen Goals", _num(_npg_val))
        with oc4:
            _ga_val = (None if (_goals_val is None or pd.isna(_goals_val))
                       else (_goals_val or 0) + (_ast_val or 0))
            st.metric("Goals + Assists", _num(_ga_val),
                      help="Combined goal involvement (actual output).")

        # ── Expected: xG / xA ────────────────────────────────────────────
        _fin_val = row_data.get("npG-xG")
        _xgs_val = row_data.get("xG/Shot")
        _xa_val = row_data.get("xA")
        _inv_val = row_data.get("npxG+xA")
        xc1, xc2, xc3, xc4, xc5 = st.columns(5)
        with xc1:
            st.metric("npxG", f"{_npxg_val:.1f}", delta=_pctd(_npxg_val, "npxG"),
                      help="Non-penalty expected goals — quality of chances taken.")
        with xc2:
            fin_str = ("—" if _fin_val is None or pd.isna(_fin_val)
                       else (f"+{_fin_val:.1f}" if _fin_val >= 0 else f"{_fin_val:.1f}"))
            st.metric("Finishing", fin_str,
                      delta=_pctd(row_data.get("npG-xG (shrunk)"), "npG-xG (shrunk)"),
                      help="npG − xG (percentile from the sample-shrunk value).")
        with xc3:
            xa_str = (f"{_xa_val:.1f}" if _xa_val is not None and not pd.isna(_xa_val) else "—")
            st.metric("xA", xa_str, delta=_pctd(_xa_val, "xA"),
                      help="Expected assists — summed xG of the chances this player created.")
        with xc4:
            inv_str = (f"{_inv_val:.1f}" if _inv_val is not None and not pd.isna(_inv_val) else "—")
            st.metric("npxG + xA", inv_str, delta=_pctd(_inv_val, "npxG+xA"),
                      help="Total expected goal involvement (non-penalty xG + xA).")
        with xc5:
            xgs_str = (f"{_xgs_val:.3f}" if _xgs_val is not None and not pd.isna(_xgs_val) else "—")
            st.metric("xG / Shot", xgs_str, delta=_pctd(_xgs_val, "xG/Shot"),
                      help="Average chance quality per shot taken.")

    # ── Expected Threat (xT) statline ────────────────────────────────────
    _xtg_val = row_data.get("xT Generated")
    _xtp_val = row_data.get("xT Prevented")
    if ((_xtg_val is not None and not pd.isna(_xtg_val))
            or (_xtp_val is not None and not pd.isna(_xtp_val))):
        st.markdown("---")
        st.markdown(f"### 🧠 Expected Threat (xT){mode_label}")
        st.caption(
            "**xT Generated** — threat this player adds through open-play passes and "
            "carries (build-up value, not shots). **xT Prevented** — threat denied by "
            "ball-winning defensive actions, each valued by how dangerous the spot was. "
            "Percentiles are vs same-position peers.")
        _xt_peers = peers[peers["posicion"] == position]

        def _xt_delta(val, col):
            if val is None or pd.isna(val) or col not in _xt_peers.columns:
                return None
            pv = _xt_peers[col].fillna(0)
            return f"{_ordinal(round((pv < val).sum() / max(len(pv), 1) * 100, 1))} %ile"

        _tc1, _tc2 = st.columns(2)
        with _tc1:
            _s = f"{_xtg_val:.2f}" if _xtg_val is not None and not pd.isna(_xtg_val) else "—"
            st.metric("⚡ xT Generated", _s, delta=_xt_delta(_xtg_val, "xT Generated"),
                      help="Open-play threat created through passing and carrying "
                           "(net V(end)−V(start) over the player's moves).")
        with _tc2:
            _s = f"{_xtp_val:.2f}" if _xtp_val is not None and not pd.isna(_xtp_val) else "—"
            st.metric("🛡️ xT Prevented", _s, delta=_xt_delta(_xtp_val, "xT Prevented"),
                      help="Threat denied by ball-winning defensive actions, valued by "
                           "the danger of the spot.")
        st.caption(
            "xT Generated is a *net* build-up value; xT Prevented is *absolute* threat "
            "denied — different scales, so judge each by its percentile, not against "
            "each other. Shown "
            + ("per 90." if "Per 90" in (mode_label or "") else "as a season total.")
        )

        # ── Where he creates & prevents threat (maps, current season) ────
        _xtg = _load_player_xt_grid_csv(_csv_mtime(_PLAYER_XT_GRID_CSV))
        _pid_xt = str(row.get("id", "")) if "id" in row.index else ""
        _mine = _xtg[_xtg["id"] == _pid_xt] if (_pid_xt and not _xtg.empty) else pd.DataFrame()
        if not _mine.empty:
            _gen = _mine[_mine["kind"] == "gen"]
            _prv = _mine[_mine["kind"] == "prev"]
            if not is_current and (not _gen.empty or not _prv.empty):
                st.caption("xT maps are built for the current season (2025-26).")
            if not _gen.empty:
                _f = _xt_zone_heatmap(_gen, player_sel, CURRENT_SEASON)
                if _f is not None:
                    st.plotly_chart(_f, use_container_width=False)
                    st.caption("**xT Generated** — red zones are where his passes and carries "
                               "raise threat; blue where he moves it back to safety. Attacks "
                               "left → right.")
            if not _prv.empty:
                _f = _xt_prevented_heatmap(_prv.rename(columns={"xt": "xtp"}),
                                           player_sel, CURRENT_SEASON)
                if _f is not None:
                    st.plotly_chart(_f, use_container_width=False)
                    st.caption("**xT Prevented** — where his ball-winning actions deny the most "
                               "threat (his own goal is on the left).")

    # ── Heat map & pass sonar (current season only) ──────────────────────
    _pid = str(row.get("id", "")) if "id" in row.index else ""
    if _pid:
        _heat = _load_player_heatmap_csv(_csv_mtime(_PLAYER_HEATMAP_CSV))
        _sonar = _load_player_sonar_csv(_csv_mtime(_PLAYER_SONAR_CSV))
        _hg = _heat[_heat["id"] == _pid] if not _heat.empty else pd.DataFrame()
        _sg = _sonar[_sonar["id"] == _pid] if not _sonar.empty else pd.DataFrame()
        if not _hg.empty or not _sg.empty:
            st.markdown("---")
            st.markdown("### 🔥 Activity Heat Map & Pass Sonar (2025-26)")
            if not is_current:
                st.caption("Heat maps and pass sonars are built for the current season "
                           "only — showing 2025-26 for this player.")
            _hcol, _scol = st.columns([3, 2])
            with _hcol:
                _f = _touch_heatmap(_hg, f"{player_sel} — where he plays") if not _hg.empty else None
                if _f is not None:
                    st.plotly_chart(_f, use_container_width=True)
                    st.caption("Density of his on-ball actions, attacking left → right.")
            with _scol:
                _f = _pass_sonar(_sg, f"{player_sel} — pass sonar") if not _sg.empty else None
                if _f is not None:
                    st.plotly_chart(_f, use_container_width=True)
                    st.caption("Passing shape: wedge = direction (forward = up), length = "
                               "avg distance, colour = completion %.")

    # ── FBref Scouting Report ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 Scouting Report{mode_label}")
    _scout_template = FBREF_TEMPLATES.get(position, FBREF_TEMPLATES.get(_peer_pos, FBREF_TEMPLATES["Central Midfield"]))
    all_scout_metrics = list(_scout_template.keys())
    selected_scout = st.multiselect("Select metrics to include", all_scout_metrics,
                                    default=all_scout_metrics, key="scout_metrics")
    pos_peers_fbref = peers[peers["posicion"] == _peer_pos]
    fig_fbref = _build_fbref_bar_chart(row_data, pos_peers_fbref, row["nombre"], position,
                                       selected_metrics=selected_scout if selected_scout else None)
    if fig_fbref:
        st.plotly_chart(fig_fbref, use_container_width=True)
    else:
        st.info("Select at least one metric above.")

    # ── Scatter Plot ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Peer Scatter Plot")
    _scatter_opts = GK_SCATTER_METRIC_OPTIONS if _is_gk else SCATTER_METRIC_OPTIONS
    avail_metrics = [m for m in _scatter_opts if m in peers.columns]
    if avail_metrics:
        defaults = SCATTER_DEFAULTS.get(position, SCATTER_DEFAULTS.get(_peer_pos, ("Att. Passes", "Goals")))
        default_x = defaults[0] if defaults[0] in avail_metrics else avail_metrics[0]
        default_y = defaults[1] if defaults[1] in avail_metrics else avail_metrics[min(1, len(avail_metrics) - 1)]
        sc1, sc2 = st.columns(2)
        with sc1:
            x_metric = st.selectbox("X-Axis", avail_metrics,
                                    index=avail_metrics.index(default_x), key="scatter_x")
        with sc2:
            y_metric = st.selectbox("Y-Axis", avail_metrics,
                                    index=avail_metrics.index(default_y), key="scatter_y")
        fig_scatter = _build_scatter_plot(row_data, peers, row["nombre"], position, x_metric, y_metric,
                                           scope_label=f"in {league}" if scope_mode == "League" else "Across Europe")
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot.")

    # ── Percentile Summary ───────────────────────────────────────────────
    st.markdown("---")
    summary_title = role if basis_mode == "Role" else f"{position}s"
    pos_peers = peers[peers["posicion"] == _peer_pos]
    st.markdown(f"### 📋 Percentile Summary — {summary_title}{mode_label}")
    pcts = _compute_percentiles(row_data, pos_peers, _prof_cats_sel)

    # Compute peer-average percentile per category
    peer_avg_pcts = {}
    for cat, metrics in _prof_cats_sel.items():
        avail = [m for m in metrics if m in pos_peers.columns]
        if not avail:
            peer_avg_pcts[cat] = 50.0
            continue
        peer_sums = pos_peers[avail].fillna(0).sum(axis=1)
        avg_sum = peer_sums.mean()
        p_pct = (peer_sums < avg_sum).sum() / max(len(peer_sums), 1) * 100
        if cat == "Discipline":
            p_pct = 100 - p_pct
        peer_avg_pcts[cat] = round(p_pct, 1)

    avg_pct = round(sum(pcts.values()) / max(len(pcts), 1), 1)
    peer_avg_pct = round(sum(peer_avg_pcts.values()) / max(len(peer_avg_pcts), 1), 1)

    v1, v2 = st.columns(2)
    with v1:
        st.metric("Player Average Percentile", f"{avg_pct:.0f}")
        st.progress(min(int(avg_pct), 100))
    with v2:
        st.metric("Peer Benchmark", f"{peer_avg_pct:.0f}")
        st.progress(min(int(peer_avg_pct), 100))

    if pcts:
        bar_cols = st.columns(len(pcts))
        for col_ui, (cat, pct) in zip(bar_cols, pcts.items()):
            with col_ui:
                st.markdown(f"**{cat}**")
                st.progress(min(int(pct), 100))
                p_avg = peer_avg_pcts.get(cat, 50)
                st.caption(f"{_ordinal(pct)} %ile · Benchmark: {_ordinal(p_avg)}")

    # ── Detailed Stats ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Detailed Statistics{mode_label}")
    if _is_gk:
        t1, t2, t3, t4 = st.tabs(["🧤 Shot-Stopping", "🏟️ Command", "📊 Distribution", "🧹 Sweeping"])
        with t1:
            ss = {m: round(row_data.get(m, 0) or 0, 2) for m in
                  ["Saves Made", "Save %",
                   "Goals Prevented", "PSxG+/-", "PSxG/Shot",
                   "Total Big Chances Saved",
                   "Penalties Saved", "Penalty Save %"] if m in peers.columns}
            if ss:
                st.dataframe(pd.DataFrame([ss]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t2:
            cmd = {m: round(row_data.get(m, 0) or 0, 2) for m in
                   ["Catches", "Punches", "Claim %", "Caught %",
                    "Aerial Duels won", "Aerial Duels", "Aerial Win %"] if m in peers.columns}
            if cmd:
                st.dataframe(pd.DataFrame([cmd]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t3:
            dist = {m: round(row_data.get(m, 0) or 0, 2) for m in
                    ["GK Successful Distribution", "GK Unsuccessful Distribution",
                     "Successful Launches", "Unsuccessful Launches",
                     "Launch %"] if m in peers.columns}
            if dist:
                st.dataframe(pd.DataFrame([dist]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t4:
            sweep = {m: round(row_data.get(m, 0) or 0, 2) for m in
                     ["Recoveries", "Total Clearances", "Interceptions"] if m in peers.columns}
            if sweep:
                st.dataframe(pd.DataFrame([sweep]).T.rename(columns={0: "Value"}), use_container_width=True)
    else:
        t1, t2, t3, t4 = st.tabs(["⚔️ Attacking", "🛡️ Defending", "📊 Passing", "🏃 Dribbling"])
        with t1:
            off = {m: round(row_data.get(m, 0) or 0, 2) for m in OFFENSIVE_METRICS if m in peers.columns}
            if off:
                st.dataframe(pd.DataFrame([off]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t2:
            defn = {m: round(row_data.get(m, 0) or 0, 2) for m in DEFENSIVE_METRICS if m in peers.columns}
            if defn:
                st.dataframe(pd.DataFrame([defn]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t3:
            pas = {m: round(row_data.get(m, 0) or 0, 2) for m in PASSING_METRICS if m in peers.columns}
            if pas:
                st.dataframe(pd.DataFrame([pas]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t4:
            dri = {m: round(row_data.get(m, 0) or 0, 2) for m in DRIBBLING_METRICS if m in peers.columns}
            if dri:
                st.dataframe(pd.DataFrame([dri]).T.rename(columns={0: "Value"}), use_container_width=True)


# ── UI: Data Explorer ────────────────────────────────────────────────────────

def render_explorer(data):
    df_total = data["total"]
    st.subheader("🔍 Data Explorer")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        leagues = st.multiselect("Filter league", sorted(df_total["league_display"].unique()), key="exp_league")
    with c2:
        exp_team_pool = df_total[df_total["league_display"].isin(leagues)] if leagues else df_total
        exp_team_opts = sorted([t for t in exp_team_pool["equipo"].dropna().unique() if t != "Unknown"])
        teams_exp = st.multiselect("Filter team", exp_team_opts, key="exp_team")
    with c3:
        pos_options = sorted(df_total["posicion_detail"].dropna().unique())
        positions = st.multiselect("Filter position", pos_options, key="exp_pos")
    with c4:
        search = st.text_input("Search player name", key="exp_search")

    # Stat mode toggle
    exp_stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="exp_stat_mode")
    df = _select_df(data, exp_stat_mode)

    filtered = df.copy()
    if leagues:
        filtered = filtered[filtered["league_display"].isin(leagues)]
    if teams_exp:
        filtered = filtered[filtered["equipo"].isin(teams_exp)]
    if positions:
        filtered = filtered[filtered["posicion_detail"].isin(positions)]
    if search:
        # Accent-insensitive: "Sesko" matches "Šeško", "Odegaard" matches "Ødegaard".
        _ns = _normalize_name(search)
        filtered = filtered[filtered["nombre"].map(_normalize_name).str.contains(_ns, regex=False, na=False)]

    # Column selector
    default_cols = ["nombre", "equipo", "league_display", "posicion_detail", "Goals", "Goal Assists",
                    "Total Shots", "Key Passes (Attempt Assists)", "Total Tackles", "Interceptions"]
    avail_cols = [c for c in default_cols if c in filtered.columns]
    all_cols = list(filtered.columns)
    show_cols = st.multiselect("Columns to display", all_cols, default=avail_cols, key="exp_cols")

    if show_cols:
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True, height=500)
    st.caption(f"Showing {len(filtered):,} players")

    # CSV download
    if show_cols:
        csv_data = filtered[show_cols].to_csv(index=False)
        st.download_button("📥 Download CSV", csv_data, file_name="forensics_xg_export.csv", mime="text/csv")


# ── UI: Team Profile ─────────────────────────────────────────────────────────

TEAM_STAT_CATEGORIES = {
    "⚔️ Attacking": [
        "Goals", "Non-Penalty Goals", "Total Shots", "Shots On Target ( inc goals )",
        "Goals from Inside Box", "Goals from Outside Box", "Headed Goals",
        "Total Touches In Opposition Box",
        "Total Big Chances Scored", "Total Big Chances Missed",
        "Shots Created", "Total Fouls Won",
    ],
    "🛡️ Defending": [
        "Total Tackles", "Tackles Won", "Tackle Win %",
        "Interceptions", "Total Clearances", "Recoveries",
        "Blocked Shots", "Blocks",
        "Aerial Duels won", "Aerial Win %",
        "Ground Duels won", "Ground Duel %",
        "Duels won", "Duel %",
        "Tackles Lost",
    ],
    "📊 Passing": [
        "Goal Assists", "Total Passes",
        "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Key Passes (Attempt Assists)",
        "Successful Long Passes", "Long Pass %",
        "Forward Passes",
        "Successful Open Play Passes", "Through balls",
        "Successful Crosses & Corners", "Cross %",
        "Short Pass %",
    ],
    "🏃 Possession": [
        "Successful Dribbles", "Dribble %",
        "Unsuccessful Dribbles",
        "Progressive Carries", "Carries",
        "Overruns",
    ],
    "🧤 Goalkeeping": [
        "Saves Made", "Goals Prevented", "Clean Sheets",
        "Penalties Saved", "Catches", "Punches",
        "Total Big Chances Saved", "Save %",
        "Launch %",
    ],
}

GK_CAT = "🧤 Goalkeeping"

# Flat list for backward compat
TEAM_AGG_METRICS = [m for metrics in TEAM_STAT_CATEGORIES.values() for m in metrics]


def _team_vs_league_chart(squad, league_players, metrics, team_sel, league_sel, title, mode_label):
    """Build a grouped bar chart comparing team avg vs league avg for a list of metrics."""
    avail = [m for m in metrics if m in squad.columns and m in league_players.columns]
    if not avail:
        return None
    team_avgs = squad[avail].mean()
    league_avgs = league_players[avail].mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=team_sel, x=avail,
        y=[round(team_avgs.get(m, 0), 2) for m in avail],
        marker_color="#2d6a4f",
    ))
    fig.add_trace(go.Bar(
        name=f"{league_sel} Avg", x=avail,
        y=[round(league_avgs.get(m, 0), 2) for m in avail],
        marker_color="#e9c46a",
    ))
    fig.update_layout(
        barmode="group", title=f"{title}{mode_label}",
        template="plotly_white", height=420, xaxis_tickangle=-45,
        margin=dict(b=120),
    )
    return fig


# ── Expected Threat (xT) ─────────────────────────────────────────────────────

# Grid dimensions, mirroring build_team_xt.py.
_XT_NX, _XT_NY = 12, 8

_XT_HIGHLIGHT = "#2d6a4f"   # the selected team
_XT_RECESSIVE = "#c9d6cf"   # every other team in the league

# ── Pitch geometry ───────────────────────────────────────────────────────────
# Opta events are 0-100 x 0-100; a real pitch is 105m x 68m, so the two axes
# carry different metres-per-unit.  Everything below converts through _mx/_my,
# which is also why the centre circle is drawn as an ellipse — a true circle on
# the grass is an ellipse in these coordinates.
_PITCH_M_X, _PITCH_M_Y = 105.0, 68.0
_PITCH_FILL = "#12302a"                    # dark tactical turf
_PITCH_LINE = "rgba(255,255,255,0.55)"


def _mx(m):
    """Metres along the pitch length → Opta x units."""
    return m / _PITCH_M_X * 100.0


def _my(m):
    """Metres across the pitch width → Opta y units."""
    return m / _PITCH_M_Y * 100.0


# Diverging red ↔ blue, with ZERO pinned to the turf colour so neutral zones
# sink into the pitch and only real gains/losses glow.  A near-white midpoint
# (the usual choice on a light background) would make every empty zone the
# brightest thing on a dark pitch — exactly backwards.
_XT_DIVERGING = [
    [0.00, "#4aa3df"], [0.18, "#2f7bb0"], [0.36, "#1d4f5e"],
    [0.50, _PITCH_FILL],
    [0.64, "#7a2b20"], [0.82, "#cf3b22"], [1.00, "#ff7a52"],
]

# xT prevented is all-positive (threat denied), so a sequential scale from the
# turf up to a bright defensive green — zero sinks into the pitch, hotspots glow.
_XTP_SEQUENTIAL = [
    [0.00, _PITCH_FILL], [0.22, "#1d4f3a"], [0.5, "#2d8a5f"],
    [0.78, "#57c98a"], [1.00, "#b6f2cf"],
]

# Classic "heat" ramp for touch density — turf → green → amber → red.
_HEAT_SEQUENTIAL = [
    [0.00, _PITCH_FILL], [0.20, "#2e5d33"], [0.45, "#b9a02a"],
    [0.70, "#e8801f"], [1.00, "#ff3b1e"],
]


def _arc_path(cx, cy, rx, ry, t0, t1, n=48):
    """SVG path string for an elliptical arc between two angles (degrees)."""
    t = np.radians(np.linspace(t0, t1, n + 1))
    pts = zip(cx + rx * np.cos(t), cy + ry * np.sin(t))
    return "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts)


def _pitch_shapes():
    """Pitch markings in Opta 0-100 coordinates, attacking left → right."""
    line = dict(color=_PITCH_LINE, width=1.6)
    box_x, six_x = _mx(16.5), _mx(5.5)
    box_y0, box_y1 = 50 - _my(40.32 / 2), 50 + _my(40.32 / 2)
    six_y0, six_y1 = 50 - _my(18.32 / 2), 50 + _my(18.32 / 2)
    goal_y0, goal_y1 = 50 - _my(7.32 / 2), 50 + _my(7.32 / 2)
    spot_x, r_x, r_y = _mx(11), _mx(9.15), _my(9.15)

    shapes = [
        # touchlines / goal lines
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=line, layer="above"),
        # halfway line
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=line, layer="above"),
        # centre circle + spot
        dict(type="circle", x0=50 - r_x, y0=50 - r_y, x1=50 + r_x, y1=50 + r_y,
             line=line, layer="above"),
        dict(type="circle", x0=50 - 0.4, y0=50 - 0.6, x1=50 + 0.4, y1=50 + 0.6,
             fillcolor=_PITCH_LINE, line=dict(width=0), layer="above"),
    ]

    for side in (0, 1):
        sx = 1 if side == 0 else -1              # mirror for the right-hand end
        base = 0 if side == 0 else 100
        # penalty area, six-yard box, goal
        shapes += [
            dict(type="rect", x0=base, y0=box_y0, x1=base + sx * box_x, y1=box_y1,
                 line=line, layer="above"),
            dict(type="rect", x0=base, y0=six_y0, x1=base + sx * six_x, y1=six_y1,
                 line=line, layer="above"),
            dict(type="rect", x0=base, y0=goal_y0, x1=base - sx * _mx(2.0), y1=goal_y1,
                 line=dict(color=_PITCH_LINE, width=2.2), layer="above"),
            # penalty spot
            dict(type="circle",
                 x0=base + sx * spot_x - 0.4, y0=50 - 0.6,
                 x1=base + sx * spot_x + 0.4, y1=50 + 0.6,
                 fillcolor=_PITCH_LINE, line=dict(width=0), layer="above"),
        ]
        # penalty arc — only the part standing outside the box
        cx = base + sx * spot_x
        cut = np.degrees(np.arccos(min(1.0, (box_x - spot_x) / r_x)))
        t0, t1 = (-cut, cut) if side == 0 else (180 - cut, 180 + cut)
        shapes.append(dict(type="path", path=_arc_path(cx, 50, r_x, r_y, t0, t1),
                           line=line, layer="above"))
        # corner arcs
        for cy in (0, 100):
            q0 = 0 if cy == 0 else -90
            shapes.append(dict(
                type="path",
                path=_arc_path(base, cy, sx * _mx(1.0), _my(1.0), q0, q0 + 90),
                line=dict(color=_PITCH_LINE, width=1.2), layer="above"))
    return shapes


def _xt_zone_heatmap(grid, team_sel, temporada, lim=None):
    """Heatmap of xT generated per match, by the zone each action started in,
    drawn on a pitch so the zones read positionally.

    *lim* fixes the colour range across the whole league-season.  Scaling each
    team to its own maximum instead would make an equally-red cell mean a
    different number on every team's map — useless for the comparison this
    section exists to support.
    """
    z = np.full((_XT_NY, _XT_NX), np.nan)
    for _, r in grid.iterrows():
        zx, zy = int(r["zx"]), int(r["zy"])
        if 0 <= zx < _XT_NX and 0 <= zy < _XT_NY:
            z[zy, zx] = float(r["xt"])
    if np.all(np.isnan(z)):
        return None

    # Symmetric range so the neutral midpoint really sits at zero.
    if not lim or not np.isfinite(lim):
        lim = float(np.nanmax(np.abs(z)))
    lim = float(lim) or 1.0
    dx, dy = 100.0 / _XT_NX, 100.0 / _XT_NY

    fig = go.Figure(go.Heatmap(
        z=z, zmid=0, zmin=-lim, zmax=lim,
        x0=dx / 2, dx=dx, y0=dy / 2, dy=dy,
        colorscale=_XT_DIVERGING, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="xT / match", side="right"), thickness=14,
                      outlinewidth=0),
        hovertemplate="Threat created: %{z:+.4f} per match<extra></extra>",
    ))
    fig.update_layout(
        title=f"{team_sel} — where the threat is created ({temporada})",
        template="plotly_white",
        width=900, height=600,
        shapes=_pitch_shapes(),
        plot_bgcolor=_PITCH_FILL, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Own goal  →  opponent goal", range=[-3, 103],
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Pitch width", range=[-2, 102],
                   showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=_PITCH_M_Y / _PITCH_M_X),
        margin=dict(l=50, r=20, t=70, b=55),
    )
    return fig


def _xt_prevented_heatmap(grid, team_sel, temporada, lim=None):
    """Heatmap of xT PREVENTED per match by the zone the defensive action
    happened in.  Drawn in the team's own attacking orientation, so its
    defensive third (where most threat is snuffed out) sits on the left.
    All-positive → a sequential (turf → green) scale, range fixed by *lim*."""
    z = np.full((_XT_NY, _XT_NX), np.nan)
    for _, r in grid.iterrows():
        zx, zy = int(r["zx"]), int(r["zy"])
        if 0 <= zx < _XT_NX and 0 <= zy < _XT_NY:
            z[zy, zx] = float(r["xtp"])
    if np.all(np.isnan(z)):
        return None
    if not lim or not np.isfinite(lim):
        lim = float(np.nanmax(z))
    lim = float(lim) or 1.0
    dx, dy = 100.0 / _XT_NX, 100.0 / _XT_NY
    fig = go.Figure(go.Heatmap(
        z=z, zmin=0, zmax=lim,
        x0=dx / 2, dx=dx, y0=dy / 2, dy=dy,
        colorscale=_XTP_SEQUENTIAL, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="xT prevented / match", side="right"),
                      thickness=14, outlinewidth=0),
        hovertemplate="Threat prevented: %{z:.4f} per match<extra></extra>",
    ))
    fig.update_layout(
        title=f"{team_sel} — where the threat is prevented ({temporada})",
        template="plotly_white", width=900, height=600, shapes=_pitch_shapes(),
        plot_bgcolor=_PITCH_FILL, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Own goal  →  opponent goal", range=[-3, 103],
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Pitch width", range=[-2, 102],
                   showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=_PITCH_M_Y / _PITCH_M_X),
        margin=dict(l=50, r=20, t=70, b=55),
    )
    return fig


def _xt_conceded_heatmap(grid, team_sel, temporada, lim=None):
    """Heatmap of xT CONCEDED per match, by the zone the opponent's action
    started in — mirrored into this team's own attacking orientation, so its
    own goal is on the left exactly like the xT-prevented map.

    Net values, so the diverging scale is the right one: red means opponents
    gain threat building from there, blue means they tend to lose it — a zone
    where this team pushes them backwards."""
    z = np.full((_XT_NY, _XT_NX), np.nan)
    for _, r in grid.iterrows():
        zx, zy = int(r["zx"]), int(r["zy"])
        if 0 <= zx < _XT_NX and 0 <= zy < _XT_NY:
            z[zy, zx] = float(r["xtc"])
    if np.all(np.isnan(z)):
        return None
    if not lim or not np.isfinite(lim):
        lim = float(np.nanmax(np.abs(z)))
    lim = float(lim) or 1.0
    dx, dy = 100.0 / _XT_NX, 100.0 / _XT_NY
    fig = go.Figure(go.Heatmap(
        z=z, zmid=0, zmin=-lim, zmax=lim,
        x0=dx / 2, dx=dx, y0=dy / 2, dy=dy,
        colorscale=_XT_DIVERGING, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="xT conceded / match", side="right"),
                      thickness=14, outlinewidth=0),
        hovertemplate="Threat conceded: %{z:+.4f} per match<extra></extra>",
    ))
    fig.update_layout(
        title=f"{team_sel} — where the threat is conceded ({temporada})",
        template="plotly_white", width=900, height=600, shapes=_pitch_shapes(),
        plot_bgcolor=_PITCH_FILL, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Own goal  →  opponent goal", range=[-3, 103],
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Pitch width", range=[-2, 102],
                   showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=_PITCH_M_Y / _PITCH_M_X),
        margin=dict(l=50, r=20, t=70, b=55),
    )
    return fig


def _touch_heatmap(grid, title, count_col="touches"):
    """Touch-density heat map on the pitch (attacking left → right).  Each cell
    is normalised to the map's own peak, then smoothed, so the shape reads as a
    classic heat map regardless of the absolute counts."""
    z = np.zeros((_XT_NY, _XT_NX))
    seen = False
    for _, r in grid.iterrows():
        zx, zy = int(r["zx"]), int(r["zy"])
        if 0 <= zx < _XT_NX and 0 <= zy < _XT_NY:
            z[zy, zx] = float(r[count_col]); seen = True
    if not seen or z.max() <= 0:
        return None
    dx, dy = 100.0 / _XT_NX, 100.0 / _XT_NY
    fig = go.Figure(go.Heatmap(
        z=z, zmin=0, zmax=float(z.max()),
        x0=dx / 2, dx=dx, y0=dy / 2, dy=dy,
        colorscale=_HEAT_SEQUENTIAL, zsmooth="best", showscale=False,
        hovertemplate="Activity here: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_white", width=760, height=500,
        shapes=_pitch_shapes(), plot_bgcolor=_PITCH_FILL, paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Own goal  →  opponent goal", range=[-3, 103],
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Pitch width", range=[-2, 102],
                   showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=_PITCH_M_Y / _PITCH_M_X),
        margin=dict(l=40, r=20, t=60, b=45),
    )
    return fig


_SONAR_N = 16   # compass sectors, mirroring build_maps.py


def _pass_sonar(sonar, title):
    """Pass sonar: a polar chart of passing shape.  Each wedge is a 22.5°
    direction; its length is the average pass distance that way, its colour the
    completion %, and hover shows the volume.  Forward (up the pitch) is up."""
    if sonar is None or sonar.empty:
        return None
    by_sec = {int(r["sector"]): r for _, r in sonar.iterrows()}
    r_vals, theta, comp, cnt = [], [], [], []
    step = 360.0 / _SONAR_N
    for s in range(_SONAR_N):
        row = by_sec.get(s)
        theta.append(s * step + step / 2.0)      # wedge centre, math angle (0=forward)
        if row is not None:
            r_vals.append(float(row["avg_dist"]))
            comp.append(float(row["completion"]))
            cnt.append(int(row["passes"]))
        else:
            r_vals.append(0.0); comp.append(0.0); cnt.append(0)
    fig = go.Figure(go.Barpolar(
        r=r_vals, theta=theta, width=[step * 0.92] * _SONAR_N,
        marker=dict(color=comp, colorscale="RdYlGn", cmin=50, cmax=95,
                    colorbar=dict(title=dict(text="Completion %", side="right"),
                                  thickness=14, outlinewidth=0),
                    line=dict(color="rgba(255,255,255,0.25)", width=1)),
        customdata=np.stack([cnt, comp], axis=-1),
        hovertemplate="Avg length: %{r:.1f} m<br>Passes: %{customdata[0]}<br>"
                      "Completion: %{customdata[1]:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_white", height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor=_PITCH_FILL,
            radialaxis=dict(showticklabels=True, ticksuffix=" m", angle=90,
                            gridcolor="rgba(255,255,255,0.15)", tickfont=dict(size=9)),
            angularaxis=dict(rotation=90, direction="counterclockwise",
                             showticklabels=False,
                             gridcolor="rgba(255,255,255,0.15)"),
        ),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def _render_team_xt_prevented(squad, team_sel, league_sel):
    """Open-play xT prevented: where a team's ball-winning defensive actions
    deny the most threat.  Reads the CSVs built by build_spatial.py."""
    st.markdown("### 🛡️ Expected Threat (xT) Prevented")
    st.caption(
        "The defensive mirror of xT: every ball-winning action (tackle won, "
        "interception, recovery, blocked pass, clearance) is valued by the "
        "threat the opponent held at that spot — snuffing out an attack near "
        "your own box prevents far more than winning the ball high up. Per "
        "match, on the same fitted surface; higher = more threat denied."
    )
    xtp = _load_team_xtp_csv(_csv_mtime(_TEAM_XTP_CSV))
    if xtp.empty:
        st.info("xT prevented hasn't been built yet — run `python build_spatial.py build`.")
        return
    liga = str(squad["liga"].iloc[0]) if "liga" in squad.columns else ""
    temporada = str(squad["temporada"].iloc[0]) if "temporada" in squad.columns else ""
    peers = xtp[(xtp["liga"] == liga) & (xtp["temporada"] == temporada)].copy()
    if peers.empty:
        st.info(f"No xT-prevented data for {league_sel} in {temporada or 'this season'}.")
        return
    peers = peers.sort_values("xtp_per_match", ascending=False).reset_index(drop=True)
    peers["Rank"] = peers.index + 1
    me = peers[peers["equipo"] == team_sel]
    league_avg = float(peers["xtp_per_match"].mean())
    if me.empty:
        st.info(f"No xT-prevented data for {team_sel} in {temporada}.")
        return
    my_xtp = float(me["xtp_per_match"].iloc[0])
    my_rank = int(me["Rank"].iloc[0])

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"{team_sel} — xT prevented / match", f"{my_xtp:.3f}",
                  delta=f"{my_xtp - league_avg:+.3f} vs league")
    with k2:
        st.metric(f"{league_sel} average", f"{league_avg:.3f}")
    with k3:
        st.metric("League rank", f"{_ordinal(my_rank)} of {len(peers)}")
    st.caption("Higher xT prevented usually means a team that *defends more* — "
               "it reflects defensive workload/threat faced, so read the map for "
               "*where* they defend rather than as a pure quality ranking.")

    grid_all = _load_team_xtp_grid_csv(_csv_mtime(_TEAM_XTP_GRID_CSV))
    if not grid_all.empty:
        league_grid = grid_all[(grid_all["liga"] == liga) & (grid_all["temporada"] == temporada)]
        grid = league_grid[league_grid["equipo"] == team_sel]
        if not grid.empty:
            lim = float(league_grid["xtp"].max())
            hm = _xt_prevented_heatmap(grid, team_sel, temporada, lim=lim)
            if hm is not None:
                st.plotly_chart(hm, use_container_width=False)
                st.caption(
                    "The team attacks left → right, so its **own goal is on the "
                    "left** — that's where most threat is prevented. Each cell is "
                    "the xT it denies per match from ball-winning actions in that "
                    f"zone; the colour scale is fixed across {league_sel} this "
                    "season so the maps compare directly."
                )


def _render_team_xt_conceded(squad, team_sel, league_sel):
    """Open-play xT conceded: how much threat opponents build against this team,
    where they build it, and the net of created minus conceded.  Reads the CSVs
    built by build_xt_conceded.py."""
    st.markdown("### 🚨 Expected Threat (xT) Conceded")
    st.caption(
        "The threat opponents create **against** this team — every opposition "
        "open-play pass and carry valued on the same fitted surface and charged "
        "to the side it was played against. Unlike xT prevented, which partly "
        "measures how *often* a team has to defend, this has no ambiguity: "
        "lower is better. Always per match."
    )

    xtc = _load_team_xtc_csv(_csv_mtime(_TEAM_XTC_CSV))
    if xtc.empty:
        st.info(
            "xT conceded hasn't been built yet — run "
            "`python build_xt_conceded.py build` to generate "
            "`team_xt_conceded.csv` from the Opta event JSONs."
        )
        return

    liga = str(squad["liga"].iloc[0]) if "liga" in squad.columns else ""
    temporada = str(squad["temporada"].iloc[0]) if "temporada" in squad.columns else ""
    peers = xtc[(xtc["liga"] == liga) & (xtc["temporada"] == temporada)].copy()
    if peers.empty:
        st.info(f"No xT-conceded data for {league_sel} in {temporada or 'this season'}.")
        return

    # Ascending: conceding least is rank 1.
    peers = peers.sort_values("xtc_per_match").reset_index(drop=True)
    peers["Rank"] = peers.index + 1
    me = peers[peers["equipo"] == team_sel]
    if me.empty:
        st.info(f"No xT-conceded data for {team_sel} in {temporada}.")
        return
    league_avg = float(peers["xtc_per_match"].mean())
    my_xtc = float(me["xtc_per_match"].iloc[0])
    my_rank = int(me["Rank"].iloc[0])

    # Net xT — the expected-threat equivalent of goal difference.
    gen = _load_team_xt_csv(_csv_mtime(_TEAM_XT_CSV))
    my_gen = None
    if not gen.empty:
        g = gen[(gen["liga"] == liga) & (gen["temporada"] == temporada)
                & (gen["equipo"] == team_sel)]
        if not g.empty:
            my_gen = float(g["xt_per_match"].iloc[0])

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"{team_sel} — xT conceded / match", f"{my_xtc:.3f}",
                  delta=f"{my_xtc - league_avg:+.3f} vs league",
                  delta_color="inverse")   # conceding less than average is good
    with k2:
        st.metric("League rank", f"{_ordinal(my_rank)} of {len(peers)}",
                  help="1st = concedes the least open-play threat per match.")
    with k3:
        if my_gen is not None:
            st.metric("Net xT / match", f"{my_gen - my_xtc:+.3f}",
                      help="xT generated minus xT conceded — the expected-threat "
                           "equivalent of goal difference.")
        else:
            st.metric(f"{league_sel} average", f"{league_avg:.3f}")

    # ── League ranking ───────────────────────────────────────────────────
    order = peers.sort_values("xtc_per_match", ascending=False)   # best on top
    colors = [_XT_HIGHLIGHT if t == team_sel else _XT_RECESSIVE
              for t in order["equipo"]]
    fig = go.Figure(go.Bar(
        x=order["xtc_per_match"], y=order["equipo"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in order["xtc_per_match"]],
        textposition="outside", cliponaxis=False,
        customdata=np.stack([order["Rank"], order["matches"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>xT conceded per match: %{x:.3f}<br>"
                      "Rank: %{customdata[0]}<br>Matches: %{customdata[1]}"
                      "<extra></extra>",
    ))
    fig.add_vline(x=league_avg, line=dict(color="#e9c46a", width=2, dash="dash"),
                  annotation_text="League avg", annotation_position="top")
    fig.update_layout(
        title=f"Open-play xT conceded per match — {league_sel}, {temporada}",
        template="plotly_white",
        height=max(360, 26 * len(order) + 130),
        xaxis_title="xT conceded per match (lower is better)", yaxis_title=None,
        margin=dict(l=10, r=60, t=60, b=40), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Best defences sit at the **bottom** — the axis is threat conceded, "
               f"so shorter is better. {team_sel} is highlighted in dark green.")

    # ── Where that threat is conceded ────────────────────────────────────
    grid_all = _load_team_xtc_grid_csv(_csv_mtime(_TEAM_XTC_GRID_CSV))
    if not grid_all.empty:
        league_grid = grid_all[(grid_all["liga"] == liga)
                               & (grid_all["temporada"] == temporada)]
        grid = league_grid[league_grid["equipo"] == team_sel]
        if not grid.empty:
            lim = float(league_grid["xtc"].abs().max())
            hm = _xt_conceded_heatmap(grid, team_sel, temporada, lim=lim)
            if hm is not None:
                st.plotly_chart(hm, use_container_width=False)
                st.caption(
                    "Read in **this team's** orientation — its own goal is on the "
                    "left, like the xT-prevented map. Each cell is the threat "
                    "opponents generate per match from moves **starting** in that "
                    "zone: red means they build danger from there, blue means they "
                    "tend to lose ground there. A red band on the left is a side "
                    "being played through inside its own half; red on the right "
                    "means opponents are already arriving in the final third. The "
                    f"colour scale is fixed across {league_sel} this season."
                )

    # ── Net xT: the two halves against each other ────────────────────────
    net = None
    if not gen.empty:
        g_league = gen[(gen["liga"] == liga) & (gen["temporada"] == temporada)]
        net = peers.merge(g_league[["equipo", "xt_per_match"]], on="equipo", how="inner")
        net["net_xt"] = net["xt_per_match"] - net["xtc_per_match"]
        net = net.sort_values("net_xt", ascending=False).reset_index(drop=True)
        net["NetRank"] = net.index + 1

    if net is not None and len(net) > 2:
        st.markdown("#### ⚖️ Net xT — creation against concession")
        st.caption(
            "Both halves on one plot. Right = creates more threat; **up = concedes "
            "less** (the axis is reversed so better is always higher). The diagonal "
            "bands are net xT, so the top-right corner is genuine two-way dominance "
            "rather than a good attack papering over a bad defence."
        )
        x_mid, y_mid = float(net["xt_per_match"].median()), float(net["xtc_per_match"].median())
        colors = [_XT_HIGHLIGHT if t == team_sel else _XT_RECESSIVE for t in net["equipo"]]
        qfig = go.Figure(go.Scatter(
            x=net["xt_per_match"], y=net["xtc_per_match"], mode="markers+text",
            text=net["equipo"], textposition="top center", textfont=dict(size=9),
            marker=dict(size=13, color=colors, line=dict(width=1, color="#2d3436")),
            customdata=np.stack([net["net_xt"], net["NetRank"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>Created: %{x:.3f}<br>"
                          "Conceded: %{y:.3f}<br>Net: %{customdata[0]:+.3f} "
                          "(%{customdata[1]:.0f} in league)<extra></extra>",
        ))
        qfig.add_vline(x=x_mid, line=dict(color="#b2bec3", width=1, dash="dot"))
        qfig.add_hline(y=y_mid, line=dict(color="#b2bec3", width=1, dash="dot"))
        _xr = [float(net["xt_per_match"].min()), float(net["xt_per_match"].max())]
        _yr = [float(net["xtc_per_match"].min()), float(net["xtc_per_match"].max())]
        _px = (_xr[1] - _xr[0]) * 0.06 or 0.01
        for xa, ya, txt, ax_, ay_ in (
            (_xr[1], _yr[0], "DOMINANT", "right", "bottom"),
            (_xr[0], _yr[0], "SOLID BUT BLUNT", "left", "bottom"),
            (_xr[1], _yr[1], "ATTACKING BUT LEAKY", "right", "top"),
            (_xr[0], _yr[1], "STRUGGLING", "left", "top"),
        ):
            qfig.add_annotation(x=xa, y=ya, text=txt, showarrow=False,
                                xanchor=ax_, yanchor=ay_,
                                font=dict(size=10, color="#95a5a6"))
        qfig.update_layout(
            title=f"Creation vs concession — {league_sel}, {temporada}",
            template="plotly_white", height=560, showlegend=False,
            xaxis=dict(title="xT generated per match  →",
                       range=[_xr[0] - _px, _xr[1] + _px]),
            # Reversed: less conceded (better) sits at the top.
            yaxis=dict(title="←  xT conceded per match (less is higher)",
                       autorange="reversed"),
            margin=dict(l=10, r=10, t=60, b=50),
        )
        st.plotly_chart(qfig, use_container_width=True)
        my_net = net[net["equipo"] == team_sel]
        if not my_net.empty:
            r = my_net.iloc[0]
            st.caption(
                f"**{team_sel}** creates {r['xt_per_match']:.3f} and concedes "
                f"{r['xtc_per_match']:.3f} per match — net **{r['net_xt']:+.3f}**, "
                f"{_ordinal(int(r['NetRank']))} of {len(net)} in {league_sel}."
            )

    with st.expander("📋 Full league table & how xT conceded is built"):
        if net is not None:
            table = net[["NetRank", "equipo", "xt_per_match", "xtc_per_match",
                         "net_xt", "matches"]].rename(columns={
                "NetRank": "Net rank", "equipo": "Team",
                "xt_per_match": "xT created / match",
                "xtc_per_match": "xT conceded / match",
                "net_xt": "Net xT / match", "matches": "Matches"})
        else:
            table = peers[["Rank", "equipo", "xtc_per_match", "xtc_total", "matches"]].rename(
                columns={"equipo": "Team", "xtc_per_match": "xT conceded / match",
                         "xtc_total": "xT conceded total", "matches": "Matches"})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown(
            "**Method.** Every opponent open-play move is valued `V(end) − V(start)` "
            "on the same surface as xT generated, then charged to the team it was "
            "played against, with the starting zone mirrored into that team's own "
            "orientation. Because every move created by one side is conceded by the "
            "other, the league totals of generated and conceded agree exactly — "
            "that identity is the builder's correctness check.\n\n"
            "Values are net, so an opponent playing backwards carries negative xT "
            "and is netted off. Set pieces, penalties and direct free-kick shots are "
            "excluded, so this is strictly open-play. Built by `build_xt_conceded.py`."
        )


def _render_team_maps(squad, team_sel, league_sel):
    """Touch heat map + pass sonar for the team (current season only)."""
    st.markdown("### 🔥 Activity Heat Map & Pass Sonar")
    temporada = str(squad["temporada"].iloc[0]) if "temporada" in squad.columns else ""
    liga = str(squad["liga"].iloc[0]) if "liga" in squad.columns else ""
    if temporada != CURRENT_SEASON:
        st.info("Heat maps and pass sonars are built for the current season only.")
        return
    heat = _load_team_heatmap_csv(_csv_mtime(_TEAM_HEATMAP_CSV))
    sonar = _load_team_sonar_csv(_csv_mtime(_TEAM_SONAR_CSV))
    if heat.empty and sonar.empty:
        st.info("Not built yet — run `python build_maps.py`.")
        return
    hc, sc = st.columns([3, 2])
    with hc:
        g = heat[(heat["liga"] == liga) & (heat["temporada"] == temporada)
                 & (heat["equipo"] == team_sel)]
        fig = _touch_heatmap(g, f"{team_sel} — where it plays ({temporada})") if not g.empty else None
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Density of the team's on-ball actions per match, attacking "
                       "left → right. Brighter = more of the game happens there.")
    with sc:
        s = sonar[(sonar["liga"] == liga) & (sonar["temporada"] == temporada)
                  & (sonar["equipo"] == team_sel)]
        fig = _pass_sonar(s, f"{team_sel} — pass sonar") if not s.empty else None
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each wedge is a passing direction (forward = up). Length = "
                       "average pass distance that way; colour = completion %.")


def _render_team_xt(squad, team_sel, league_sel):
    """Open-play xT generated: where the team ranks in its league, and which
    pitch zones its threat comes from.  Reads the CSVs built by
    build_team_xt.py, and degrades to a notice if they aren't built yet."""
    st.markdown("### ⚡ Expected Threat (xT) Generated")
    st.caption(
        "xT values the build-up rather than the shot: every successful open-play "
        "pass and carry is scored by how much it raised the probability that the "
        "possession ends in a goal. Fit in-house on a 12×8 grid from the raw Opta "
        "event data, using a single surface across all leagues and seasons so the "
        "numbers compare directly. Always per match — the Total / Per 90 toggle "
        "does not apply here."
    )

    xt = _load_team_xt_csv(_csv_mtime(_TEAM_XT_CSV))
    if xt.empty:
        st.info(
            "xT hasn't been built yet — run `python build_team_xt.py build` to "
            "generate `team_xt.csv` from the Opta event JSONs."
        )
        return

    liga = str(squad["liga"].iloc[0]) if "liga" in squad.columns else ""
    temporada = str(squad["temporada"].iloc[0]) if "temporada" in squad.columns else ""
    peers = xt[(xt["liga"] == liga) & (xt["temporada"] == temporada)].copy()
    if peers.empty:
        st.info(f"No xT data for {league_sel} in {temporada or 'this season'}.")
        return

    peers = peers.sort_values("xt_per_match", ascending=False).reset_index(drop=True)
    peers["Rank"] = peers.index + 1
    me = peers[peers["equipo"] == team_sel]
    league_avg = float(peers["xt_per_match"].mean())

    if me.empty:
        st.info(f"No xT data for {team_sel} in {temporada}.")
        return
    my_xt = float(me["xt_per_match"].iloc[0])
    my_rank = int(me["Rank"].iloc[0])

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"{team_sel} — xT per match", f"{my_xt:.3f}",
                  delta=f"{my_xt - league_avg:+.3f} vs league")
    with k2:
        st.metric(f"{league_sel} average", f"{league_avg:.3f}")
    with k3:
        st.metric("League rank", f"{_ordinal(my_rank)} of {len(peers)}")

    # ── League ranking ───────────────────────────────────────────────────
    order = peers.sort_values("xt_per_match")          # ascending: best on top
    colors = [_XT_HIGHLIGHT if t == team_sel else _XT_RECESSIVE
              for t in order["equipo"]]
    fig = go.Figure(go.Bar(
        x=order["xt_per_match"], y=order["equipo"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in order["xt_per_match"]],
        textposition="outside", cliponaxis=False,
        customdata=np.stack([order["Rank"], order["matches"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>xT per match: %{x:.3f}<br>"
                      "Rank: %{customdata[0]}<br>Matches: %{customdata[1]}"
                      "<extra></extra>",
    ))
    fig.add_vline(x=league_avg, line=dict(color="#e9c46a", width=2, dash="dash"),
                  annotation_text="League avg", annotation_position="top")
    fig.update_layout(
        title=f"Average xT generated per match — {league_sel}, {temporada}",
        template="plotly_white",
        height=max(360, 26 * len(order) + 130),
        xaxis_title="xT per match", yaxis_title=None,
        margin=dict(l=10, r=60, t=60, b=40), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{team_sel} is highlighted in dark green; the dashed line is the league average.")

    # ── Where that threat comes from ─────────────────────────────────────
    grid_all = _load_team_xt_grid_csv(_csv_mtime(_TEAM_XT_GRID_CSV))
    if not grid_all.empty:
        league_grid = grid_all[(grid_all["liga"] == liga)
                               & (grid_all["temporada"] == temporada)]
        grid = league_grid[league_grid["equipo"] == team_sel]
        if not grid.empty:
            # One colour range for the whole league-season, so switching teams
            # compares like with like.
            lim = float(league_grid["xt"].abs().max())
            hm = _xt_zone_heatmap(grid, team_sel, temporada, lim=lim)
            if hm is not None:
                st.plotly_chart(hm, use_container_width=False)
                st.caption(
                    "The team attacks left → right. Each cell is the xT it creates "
                    "per match from actions **starting** in that zone: red means it "
                    "gains threat from there, blue means possession there tends to "
                    "move the ball away from danger, and zones level with the turf "
                    "are neutral. The colour scale is fixed across every team in "
                    f"{league_sel} this season, so these maps compare directly."
                )

    with st.expander("📋 Full league table & how xT is built"):
        table = peers[["Rank", "equipo", "xt_per_match", "xt_total", "matches", "moves"]].rename(
            columns={"equipo": "Team", "xt_per_match": "xT / match",
                     "xt_total": "xT total", "matches": "Matches",
                     "moves": "Open-play moves"})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown(
            "**Method.** Every action from a pitch zone is one of three things: a "
            "shot, a successful move (pass or carry), or a loss of possession. "
            "Value iteration solves for each zone's goal probability, "
            "`V(z) = P(shot)·P(goal|shot) + P(move)·Σ T(z→z')·V(z')`. An action is "
            "worth `V(end) − V(start)`, and a team's xT is the sum over its "
            "successful open-play moves.\n\n"
            "Set pieces (corners, free kicks, throw-ins, goal kicks), penalties and "
            "direct free-kick shots are excluded throughout, so this is strictly "
            "open-play xT. Built by `build_team_xt.py`."
        )


def _pivot_verdict(rank, name, score):
    """Turn the squad's best Pivot Index into a plain-language verdict.  The
    rank is Europe-wide for that season (all seven leagues), so it already
    folds in control, progression and how deep the player operates."""
    if rank <= 20:
        return ("✅", f"**{name}** is a genuine deep-lying playmaker — "
                      f"{_ordinal(rank)} in Europe this season.")
    if rank <= 60:
        return ("🟢", f"**{name}** is a credible pivot ({_ordinal(rank)} in Europe), "
                      f"without being an elite one.")
    if rank <= 150:
        return ("🟠", f"No established regista. The best profile, **{name}**, ranks "
                      f"{_ordinal(rank)} in Europe — the squad's control and its "
                      f"progression sit in different players.")
    return ("🔴", f"No deep playmaking presence at all — the best midfield profile, "
                  f"**{name}**, is only {_ordinal(rank)} in Europe.")


def _render_team_pivot(squad, team_sel, league_sel):
    """Midfield archetypes: does this squad contain a true deep-lying playmaker?

    Plots the Pivot Index's two on-ball axes against each other — CONTROL (does
    the ball go through him and survive) vs PROGRESSION (does it gain value when
    it does) — with colour carrying the third axis, ANCHOR (how deep he plays).
    A regista sits top-right in a dark dot; an empty top-right corner IS the
    answer for a club that hasn't got one.  Built by build_pivot_index.py."""
    st.markdown("### 🎛️ Midfield Archetypes — is there a true pivot?")
    st.caption(
        "A deep-lying playmaker is an *intersection*, not a single number: high "
        "passing volume alone is a recycler, high progression alone is an advanced "
        "creator. Every midfielder in the seven leagues is percentile-ranked on "
        "three axes — **CONTROL** (own-half and total pass volume, completion, ball "
        "security), **PROGRESSION** (xT generated per 90 and per move, long passes, "
        "through balls, forward-pass share) and **ANCHOR** (share of passes played "
        "in his own half). Minimum 600 minutes."
    )

    piv = _load_player_pivot_csv(_csv_mtime(_PLAYER_PIVOT_CSV))
    if piv.empty:
        st.info(
            "The Pivot Index hasn't been built yet — run "
            "`python build_pivot_index.py build` to generate `player_pivot.csv`."
        )
        return

    liga = str(squad["liga"].iloc[0]) if "liga" in squad.columns else ""
    temporada = str(squad["temporada"].iloc[0]) if "temporada" in squad.columns else ""
    pool = piv[(piv["liga"] == liga) & (piv["temporada"] == temporada)].copy()
    mine = pool[pool["equipo"] == team_sel].copy()
    if mine.empty:
        if pool.empty:
            # Bundesliga's historical file was rebuilt from event JSONs and has
            # no own-half passing split, so past Bundesliga seasons can't be
            # scored at all — say so rather than implying nobody played enough.
            st.info(
                f"The Pivot Index isn't available for {league_sel} in {temporada} — "
                "the source season-stat file for those seasons has no own-half "
                "passing split, which the Control and Anchor axes both need."
            )
        else:
            st.info(f"No {team_sel} midfielder reached 600 minutes in {temporada}.")
        return
    mine = mine.sort_values("PIVOT", ascending=False)

    best = mine.iloc[0]
    icon, verdict = _pivot_verdict(int(best["pivot_rank"]), best["nombre"], best["PIVOT"])

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"Best pivot profile — {best['nombre']}", f"{best['PIVOT']:.1f}",
                  delta=f"{_ordinal(int(best['pivot_rank']))} in Europe",
                  delta_color="off")
    with k2:
        # Clubs ordered by their single best midfielder: sort desc, then the
        # first appearance of each club is that club's best profile.
        club_order = pool.sort_values("PIVOT", ascending=False)["equipo"].drop_duplicates().tolist()
        lg_rank = club_order.index(team_sel) + 1
        st.metric(f"Rank within {league_sel}",
                  f"{_ordinal(lg_rank)} of {len(club_order)}",
                  help="Clubs ranked by their single best midfielder's Pivot Index.")
    with k3:
        st.metric("Squad midfielders scored", f"{len(mine)}",
                  help="Midfielders with at least 600 minutes this season.")
    st.markdown(f"{icon} {verdict}")

    # ── The quadrant ─────────────────────────────────────────────────────
    x_mid = float(pool["CONTROL"].median())
    y_mid = float(pool["PROGRESSION"].median())
    fig = go.Figure()

    # League peers as context, so the team's cluster is read against something.
    others = pool[pool["equipo"] != team_sel]
    if not others.empty:
        fig.add_trace(go.Scatter(
            x=others["CONTROL"], y=others["PROGRESSION"], mode="markers",
            marker=dict(size=7, color=_XT_RECESSIVE, line=dict(width=0)),
            name=f"Other {league_sel} midfielders",
            customdata=np.stack([others["nombre"], others["equipo"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                          "Control: %{x:.0f}<br>Progression: %{y:.0f}<extra></extra>",
        ))

    sizes = 12 + 20 * (mine["minutes"] / max(float(mine["minutes"].max()), 1.0))
    fig.add_trace(go.Scatter(
        x=mine["CONTROL"], y=mine["PROGRESSION"], mode="markers+text",
        text=mine["nombre"], textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=sizes, color=mine["ANCHOR"], colorscale="Teal",
                    cmin=0, cmax=100, line=dict(width=1, color="#2d3436"),
                    colorbar=dict(title="Anchor<br>(deep ↑)", thickness=14, len=0.7)),
        name=team_sel,
        customdata=np.stack([mine["ANCHOR"], mine["minutes"], mine["long90"],
                             mine["xt90"], mine["pivot_rank"]], axis=-1),
        hovertemplate="<b>%{text}</b><br>Control: %{x:.0f}<br>"
                      "Progression: %{y:.0f}<br>Anchor: %{customdata[0]:.0f}<br>"
                      "Minutes: %{customdata[1]:.0f}<br>"
                      "Long passes/90: %{customdata[2]:.1f}<br>"
                      "xT/90: %{customdata[3]:.3f}<br>"
                      "Europe rank: %{customdata[4]:.0f}<extra></extra>",
    ))

    fig.add_vline(x=x_mid, line=dict(color="#b2bec3", width=1, dash="dot"))
    fig.add_hline(y=y_mid, line=dict(color="#b2bec3", width=1, dash="dot"))
    for xa, ya, txt, anc in (
        (99, 99, "PIVOT PLAYMAKER", "right"),
        (1, 99, "ADVANCED CREATOR", "left"),
        (99, 1, "RECYCLER / HOLDER", "right"),
        (1, 1, "LIMITED ON THE BALL", "left"),
    ):
        fig.add_annotation(x=xa, y=ya, text=txt, showarrow=False,
                           xanchor=anc, yanchor="top" if ya > 50 else "bottom",
                           font=dict(size=10, color="#95a5a6"))
    fig.update_layout(
        title=f"Midfield archetypes — {team_sel}, {temporada}",
        template="plotly_white", height=560,
        xaxis=dict(title="CONTROL — does the ball go through him?", range=[0, 102]),
        yaxis=dict(title="PROGRESSION — does it gain value?", range=[0, 102]),
        margin=dict(l=10, r=10, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Dot size is minutes played; colour is **ANCHOR** — dark means he builds "
        "from his own half, pale means he operates high up the pitch. The dotted "
        f"lines are the {league_sel} midfield medians. A true pivot is a *dark* dot "
        "in the top-right: a light dot there is an advanced creator who happens to "
        "see a lot of the ball."
    )

    with st.expander("📋 Squad detail & how the Pivot Index is built"):
        tbl = mine[["nombre", "minutes", "PIVOT", "pivot_rank", "CONTROL",
                    "PROGRESSION", "ANCHOR", "deep_pass90", "pass90", "long90",
                    "xt90", "cmp_pct"]].rename(columns={
            "nombre": "Player", "minutes": "Minutes", "pivot_rank": "Europe rank",
            "deep_pass90": "Own-half passes/90", "pass90": "Passes/90",
            "long90": "Long passes/90", "xt90": "xT/90", "cmp_pct": "Pass %"})
        st.dataframe(tbl.round(2), use_container_width=True, hide_index=True)
        st.markdown(
            "**Method.** `PIVOT = √(CONTROL × PROGRESSION) × anchor_gate`. The "
            "geometric mean is the point: a weighted *sum* lets a high-volume "
            "destroyer with no progression score like a playmaker, and the product "
            "does not — a midfielder has to be good at both. ANCHOR is only a gate "
            "(0.70 → 1.00), never an additive term, because a holding midfielder "
            "who never leaves his own half tops the anchor metric without being a "
            "playmaker: it can pull a score down for playing too high, but it can "
            "never lift one.\n\n"
            "Components are percentile-ranked within each season's midfielder pool "
            "across all seven leagues, so scores compare across leagues and across "
            "seasons. Built by `build_pivot_index.py` from the season-stat CSVs "
            "plus the xT model in `player_xt.csv`.\n\n"
            "*Coverage note.* The Bundesliga can only be scored for the current "
            "season — its historical file was rebuilt from event JSONs and carries "
            "no own-half passing split — so the pools for 2020-21 → 2024-25 are "
            "six leagues rather than seven."
        )


def render_team_profile(data):
    df_total = data["total"]
    st.subheader("🏟️ Team Profile")

    # Selectors
    c1, c2 = st.columns(2)
    with c1:
        league_sel = st.selectbox("League", sorted(df_total["league_display"].unique()), key="tp_lg")
    with c2:
        teams_in_league = sorted([t for t in df_total[df_total["league_display"] == league_sel]["equipo"].dropna().unique()
                                  if t != "Unknown"])
        if not teams_in_league:
            st.warning("No team data available for this league.")
            return
        team_sel = st.selectbox("Team", teams_in_league, key="tp_team")

    if not team_sel:
        return

    stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="tp_stat_mode")
    df = _select_df(data, stat_mode)
    mode_label = f" ({stat_mode})" if stat_mode != "Total" else ""

    squad = df[(df["equipo"] == team_sel) & (df["league_display"] == league_sel)].copy()
    if squad.empty:
        st.warning("No players found for this team.")
        return

    # ── Team Overview ────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Squad Size", len(squad))
    with m2:
        st.metric("Total Goals", _safe_int(squad["Goals"].sum()) if "Goals" in squad.columns else 0)
    with m3:
        st.metric("Total Assists", _safe_int(squad["Goal Assists"].sum()) if "Goal Assists" in squad.columns else 0)
    with m4:
        st.metric("Total Shots", _safe_int(squad["Total Shots"].sum()) if "Total Shots" in squad.columns else 0)
    with m5:
        st.metric("Big Chances", _safe_int(squad["Total Big Chances Created"].sum()) if "Total Big Chances Created" in squad.columns else 0)

    # ── Team Radar — Category Overview vs League ──────────────────────────
    st.markdown(f"### 🕸️ Team Radar — vs League Average{mode_label}")
    league_players = df[df["league_display"] == league_sel]

    # Compute a composite score per category: mean of (team_avg / league_avg) across available metrics
    radar_cats = []
    team_scores = []
    league_scores = []   # always 100 (baseline)
    team_avgs_per_cat = []   # actual team averages per category
    league_avgs_per_cat = [] # actual league averages per category
    for cat_name, cat_metrics in TEAM_STAT_CATEGORIES.items():
        sq = squad[squad["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else squad
        lp = league_players[league_players["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else league_players
        avail = [m for m in cat_metrics if m in sq.columns and m in lp.columns]
        if not avail or sq.empty:
            continue
        ratios = []
        t_avg_sum, l_avg_sum, n_metrics = 0, 0, 0
        for m in avail:
            l_avg = lp[m].mean()
            t_avg = sq[m].mean()
            if pd.notna(l_avg) and l_avg > 0 and pd.notna(t_avg):
                ratios.append(t_avg / l_avg)
                t_avg_sum += t_avg
                l_avg_sum += l_avg
                n_metrics += 1
        if ratios:
            radar_cats.append(cat_name)
            team_scores.append(round(np.mean(ratios) * 100, 1))   # 100 = league-level
            league_scores.append(100)
            team_avgs_per_cat.append(round(t_avg_sum / n_metrics, 2))
            league_avgs_per_cat.append(round(l_avg_sum / n_metrics, 2))

    if len(radar_cats) >= 3:
        fig_radar = go.Figure()
        # Build hover data: team avg, league avg, score
        team_hover = list(zip(team_avgs_per_cat + [team_avgs_per_cat[0]],
                              league_avgs_per_cat + [league_avgs_per_cat[0]],
                              team_scores + [team_scores[0]]))
        league_hover = list(zip(league_avgs_per_cat + [league_avgs_per_cat[0]],
                                team_avgs_per_cat + [team_avgs_per_cat[0]]))
        fig_radar.add_trace(go.Scatterpolar(
            r=team_scores + [team_scores[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself", name=team_sel,
            fillcolor="rgba(45,106,79,0.25)", line=dict(color="#2d6a4f", width=2),
            customdata=team_hover,
            hovertemplate="<b>%{theta}</b><br>"
                          f"{team_sel} Avg: " + "%{customdata[0]}<br>"
                          "League Avg: %{customdata[1]}<br>"
                          "Score: %{customdata[2]} / 100"
                          "<extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=league_scores + [league_scores[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself", name=f"{league_sel} Avg",
            fillcolor="rgba(233,196,106,0.15)", line=dict(color="#e9c46a", width=2, dash="dash"),
            customdata=league_hover,
            hovertemplate="<b>%{theta}</b><br>"
                          "League Avg: %{customdata[0]}<br>"
                          f"{team_sel} Avg: " + "%{customdata[1]}<br>"
                          "Baseline: 100"
                          "<extra></extra>",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(max(team_scores), 100) + 20]),
            ),
            title=f"{team_sel} — Category Profile vs {league_sel} Average{mode_label}",
            template="plotly_white", height=520, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Score of 100 = league average. Above 100 means the team outperforms the league in that category.")
    else:
        st.info("Not enough category data to build the radar chart.")

    # ── Squad Table (Categorised) ───────────────────────────────────────
    st.markdown(f"### 📋 Squad — Detailed Stats{mode_label}")
    cat_tabs = st.tabs(list(TEAM_STAT_CATEGORIES.keys()))
    id_cols = ["nombre", "posicion_detail"]
    for tab, (cat_name, cat_metrics) in zip(cat_tabs, TEAM_STAT_CATEGORIES.items()):
        with tab:
            sq = squad[squad["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else squad
            cols_avail = [m for m in cat_metrics if m in sq.columns]
            if not cols_avail or sq.empty:
                st.info(f"No {cat_name} data available for this squad.")
                continue
            show = [c for c in id_cols if c in sq.columns] + cols_avail
            sort_col = cols_avail[0]
            st.dataframe(
                sq[show].sort_values(sort_col, ascending=False).reset_index(drop=True),
                use_container_width=True, height=420,
            )

    # ── Top Contributors ─────────────────────────────────────────────────
    st.markdown(f"### ⭐ Top Contributors{mode_label}")
    contrib_metric = st.selectbox("Rank players by",
                                  [m for m in TEAM_AGG_METRICS if m in squad.columns],
                                  key="tp_metric")
    if contrib_metric:
        top = squad[squad[contrib_metric].notna()].nlargest(min(10, len(squad)), contrib_metric)
        fig = chart_bar(top, "nombre", contrib_metric,
                        f"{team_sel} — Top Players by {contrib_metric}{mode_label}",
                        color="posicion_detail")
        st.plotly_chart(fig, use_container_width=True)

    # ── Team vs League Average (Categorised) ─────────────────────────────
    st.markdown(f"### 🌍 Team vs League Average{mode_label}")
    compare_tabs = st.tabs(list(TEAM_STAT_CATEGORIES.keys()))
    for tab, (cat_name, cat_metrics) in zip(compare_tabs, TEAM_STAT_CATEGORIES.items()):
        with tab:
            sq = squad[squad["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else squad
            lp = league_players[league_players["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else league_players
            if sq.empty:
                st.info(f"No goalkeepers found for {cat_name} comparison.")
                continue
            fig = _team_vs_league_chart(
                sq, lp, cat_metrics, team_sel, league_sel,
                f"{team_sel} vs {league_sel} — {cat_name} ", mode_label,
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {cat_name} metrics available for comparison.")

    # ── Expected Threat (xT) Generated ────────────────────────────────────
    _render_team_xt(squad, team_sel, league_sel)

    # ── Expected Threat (xT) Conceded ─────────────────────────────────────
    _render_team_xt_conceded(squad, team_sel, league_sel)

    # ── Expected Threat (xT) Prevented ────────────────────────────────────
    _render_team_xt_prevented(squad, team_sel, league_sel)

    # ── Midfield archetypes / Pivot Index ─────────────────────────────────
    _render_team_pivot(squad, team_sel, league_sel)

    # ── Heat map & pass sonar ─────────────────────────────────────────────
    _render_team_maps(squad, team_sel, league_sel)

    # ── Player Scorecard & Grading ──────────────────────────────────────
    st.markdown("### 🃏 Player Scorecard & Grading")
    st.caption(
        "Each player is graded on four categories — Attacking, Defending, Passing, and Possession — "
        "using league-wide positional percentiles. GKs are graded on Shot-Stopping, Command, "
        "Distribution, and Sweeping. Grades range from S+ (97th+) to F (below 45th)."
    )

    def _sc_grade_color(grade):
        return _GRADE_COLORS.get(grade, "#555")

    # ── Scorecard metric categories ─────────────────────────────────
    _SC_CATEGORIES = {
        "⚔️ Attacking": ATTRIBUTE_GRADE_CATEGORIES["Attacking"],
        "🛡️ Defending": ATTRIBUTE_GRADE_CATEGORIES["Defending"],
        "📊 Passing": ATTRIBUTE_GRADE_CATEGORIES["Passing"],
        "🎯 Possession": (
            ATTRIBUTE_GRADE_CATEGORIES.get("Dribbling & Carrying", [])
            + ATTRIBUTE_GRADE_CATEGORIES.get("Ball Progression", [])
            + ATTRIBUTE_GRADE_CATEGORIES.get("Passing Safety", [])
        ),
    }
    # Deduplicate Possession while preserving order
    _seen = set()
    _SC_CATEGORIES["🎯 Possession"] = [
        m for m in _SC_CATEGORIES["🎯 Possession"]
        if not (m in _seen or _seen.add(m))
    ]
    _GK_SC_CATEGORIES = {
        "🧤 Shot-Stopping": GK_ATTRIBUTE_GRADE_CATEGORIES["Shot-Stopping"],
        "📢 Command": GK_ATTRIBUTE_GRADE_CATEGORIES["Command"],
        "📊 Distribution": GK_ATTRIBUTE_GRADE_CATEGORIES["Distribution"],
        "🧹 Sweeping": GK_ATTRIBUTE_GRADE_CATEGORIES["Sweeping"],
    }

    _LOWER_SC = {
        "Unsuccessful Dribbles", "Overruns",
        "Total Losses Of Possession",
    }

    # ── Separate outfield vs GK ─────────────────────────────────────
    _outfield_squad = squad[squad["posicion"] != "Goalkeeper"].copy()
    _gk_squad = squad[squad["posicion"] == "Goalkeeper"].copy()

    # --- Build league reference matching the active stat mode ---
    _league_raw = df[df["league_display"] == league_sel]
    if stat_mode == "Per 90" and "estimated_90s" in _league_raw.columns:
        _MIN_90S_SC = 5
        _is_team = _league_raw["equipo"] == team_sel
        _has_mins = _league_raw["estimated_90s"].fillna(0) >= _MIN_90S_SC
        _league_df = _league_raw[_is_team | _has_mins].copy()
    else:
        _league_df = _league_raw.copy()

    _outfield_peers = _league_df[_league_df["posicion"] != "Goalkeeper"].copy()
    _gk_peers = _league_df[_league_df["posicion"] == "Goalkeeper"].copy()

    # ── Compute outfield percentiles ─────────────────────────────────
    _outfield_cards = {}  # name -> {cat_name: {metrics, pctiles, composite, grade}}
    for cat_name, cat_metrics in _SC_CATEGORIES.items():
        avail = [m for m in cat_metrics if m in _outfield_peers.columns]
        if not avail:
            continue
        pctile_df = _outfield_peers[avail].rank(pct=True) * 100
        for m in avail:
            if m in _LOWER_SC or m in _INVERTED_GRADE_CATS:
                pctile_df[m] = 100 - pctile_df[m]
        pctile_df["nombre"] = _outfield_peers["nombre"].values
        pctile_df["equipo"] = _outfield_peers["equipo"].values
        team_pct = pctile_df[pctile_df["equipo"] == team_sel]
        for _, row in team_pct.iterrows():
            name = row["nombre"]
            comp = round(pd.to_numeric(row[avail], errors="coerce").mean(), 1)
            _outfield_cards.setdefault(name, {})[cat_name] = {
                "metrics": avail,
                "pctiles": {m: round(row[m], 1) for m in avail},
                "composite": comp,
                "grade": _percentile_to_grade(comp),
            }

    # ── Compute GK percentiles ───────────────────────────────────────
    _gk_cards = {}
    for cat_name, cat_metrics in _GK_SC_CATEGORIES.items():
        avail = [m for m in cat_metrics if m in _gk_peers.columns]
        if not avail or _gk_peers.empty:
            continue
        pctile_df = _gk_peers[avail].rank(pct=True) * 100
        pctile_df["nombre"] = _gk_peers["nombre"].values
        pctile_df["equipo"] = _gk_peers["equipo"].values
        team_pct = pctile_df[pctile_df["equipo"] == team_sel]
        for _, row in team_pct.iterrows():
            name = row["nombre"]
            comp = round(pd.to_numeric(row[avail], errors="coerce").mean(), 1)
            _gk_cards.setdefault(name, {})[cat_name] = {
                "metrics": avail,
                "pctiles": {m: round(row[m], 1) for m in avail},
                "composite": comp,
                "grade": _percentile_to_grade(comp),
            }

    # ── Build dropdown ──────────────────────────────────────────────
    is_gk = False
    _dropdown_options = []  # (label, name, is_gk)
    for name, cats in sorted(_outfield_cards.items(),
                              key=lambda x: sum(c["composite"] for c in x[1].values()) / max(len(x[1]), 1),
                              reverse=True):
        cat_summary = " | ".join(f"{cn.split(' ', 1)[-1]}: {cd['grade']}" for cn, cd in cats.items())
        _dropdown_options.append((f"{name}  —  {cat_summary}", name, False))
    for name, cats in sorted(_gk_cards.items(),
                              key=lambda x: sum(c["composite"] for c in x[1].values()) / max(len(x[1]), 1),
                              reverse=True):
        cat_summary = " | ".join(f"{cn.split(' ', 1)[-1]}: {cd['grade']}" for cn, cd in cats.items())
        _dropdown_options.append((f"🧤 {name}  —  {cat_summary}", name, True))

    if not _dropdown_options:
        st.info(f"No graded players found for {team_sel}.")
    else:
        _labels = [opt[0] for opt in _dropdown_options]
        _sel_label = st.selectbox("Select a player", _labels, key="sc_player_sel")
        _sel_idx = _labels.index(_sel_label)
        _, player_name, is_gk = _dropdown_options[_sel_idx]

        cats = _gk_cards.get(player_name, {}) if is_gk else _outfield_cards.get(player_name, {})
        raw_squad = _gk_squad if is_gk else _outfield_squad

        # Header: just the 4 category grades
        prefix = "🧤 " if is_gk else ""
        cat_header = "  |  ".join(f"{cn.split(' ', 1)[-1]}: **{cd['grade']}**" for cn, cd in cats.items())
        st.markdown(f"#### {prefix}{player_name}  —  {cat_header}")

        # Tabs for each category
        cat_tabs = st.tabs(list(cats.keys()))
        for tab, (cat_name, cat_data) in zip(cat_tabs, cats.items()):
            with tab:
                raw_row = raw_squad[raw_squad["nombre"] == player_name]
                detail_rows = []
                for m in cat_data["metrics"]:
                    raw_val = (round(raw_row[m].values[0], 2)
                               if len(raw_row) and m in raw_row.columns and pd.notna(raw_row[m].values[0])
                               else "-")
                    pct = cat_data["pctiles"][m]
                    g = _percentile_to_grade(pct)
                    detail_rows.append({
                        "Metric": m,
                        f"Value{mode_label}": raw_val,
                        "Pctile": f"{pct:.0f}",
                        "Grade": g,
                    })

                _sc_pcts = [cat_data["pctiles"][r["Metric"]] for r in detail_rows]
                _sc_grades = [_percentile_to_grade(p) for p in _sc_pcts]
                fig_sc = go.Figure(go.Bar(
                    x=[r["Metric"] for r in detail_rows],
                    y=_sc_pcts,
                    marker_color=[_sc_grade_color(g) for g in _sc_grades],
                    text=_sc_grades,
                    textposition="outside",
                    customdata=list(zip(
                        [r[f"Value{mode_label}"] for r in detail_rows],
                        [f"{p:.0f}" for p in _sc_pcts],
                        _sc_grades,
                    )),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Value: %{customdata[0]}<br>"
                        "Percentile: %{customdata[1]}<br>"
                        "Grade: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ))
                fig_sc.add_hline(y=50, line_dash="dash", line_color="grey",
                                 annotation_text="League Median", annotation_position="top left")
                fig_sc.update_layout(
                    title=f"{player_name} — {cat_name}",
                    yaxis_title="Percentile", yaxis_range=[0, 105],
                    template="plotly_white", height=380,
                    xaxis_tickangle=-45, margin=dict(b=100),
                )
                st.plotly_chart(fig_sc, use_container_width=True,
                                key=f"sc_{player_name}_{cat_name}")
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # ── GK Value Behind Defence Quality ──────────────────────────────────
    if is_gk:
        st.markdown("### 🧱 GK Value Add — Defence Quality Context")
        st.caption(
            "How much does each goalkeeper overperform (or underperform) given "
            "the quality of the defence in front of them? GKs paired with weak "
            "defences face more — and harder — shots; this chart measures who "
            "rises above (or sinks below) expectations."
        )

        # 1. Compute team defensive composite for every team in this league
        _def_peers = df_total[
            (df_total["league_display"] == league_sel) &
            (df_total["posicion"].isin(_DEFENDER_POSITIONS))
        ].copy()
        _def_avail = [m for m in ATTRIBUTE_GRADE_CATEGORIES.get("Defending", []) if m in _def_peers.columns]
        _def_pct = _def_peers[_def_avail].rank(pct=True) * 100
        _def_pct["equipo"] = _def_peers["equipo"].values
        _team_def_comp = _def_pct.groupby("equipo")[_def_avail].mean().mean(axis=1)

        # 2. For each GK with enough data, compute their key performance metrics
        _min_saves = 15
        _all_gks = df_total[
            (df_total["league_display"] == league_sel) & (df_total["posicion"] == "Goalkeeper")
        ].copy()
        _gk_rows = []
        for _, _r in _all_gks.iterrows():
            _t = _r["equipo"]
            if _t == "Unknown":
                continue
            _gp = _r.get("Goals Prevented", np.nan)
            _sv = _r.get("Save %", np.nan)
            _saves = _r.get("Saves Made", np.nan)
            if pd.isna(_gp) and pd.isna(_sv):
                continue
            if pd.notna(_saves) and _saves < _min_saves:
                continue  # skip GKs without enough playing time
            _gk_rows.append({
                "GK": _r["nombre"],
                "Team": _t,
                "Defence Quality": round(_team_def_comp.get(_t, 50), 1),
                "Goals Prevented": round(_gp, 2) if pd.notna(_gp) else np.nan,
                "Save %": round(_sv, 1) if pd.notna(_sv) else np.nan,
                "Saves Made": _saves,
            })

        if _gk_rows:
            gk_ctx = pd.DataFrame(_gk_rows)

            # 3. Compute "Value Add" = residual from linear fit of GP ~ Defence Quality
            _valid = gk_ctx.dropna(subset=["Goals Prevented", "Defence Quality"])
            if len(_valid) >= 4:
                _x = _valid["Defence Quality"].values
                _y = _valid["Goals Prevented"].values
                _coeffs = np.polyfit(_x, _y, 1)
                _expected = np.polyval(_coeffs, gk_ctx["Defence Quality"].values)
                gk_ctx["Expected GP"] = np.round(_expected, 2)
                gk_ctx["GK Value Add"] = np.round(
                    gk_ctx["Goals Prevented"].values - _expected, 2
                )
            else:
                gk_ctx["Expected GP"] = np.nan
                gk_ctx["GK Value Add"] = np.nan

            # 4. Scatter: Defence Quality (x) vs Goals Prevented (y)
            _highlight = gk_ctx["Team"] == team_sel
            gk_ctx["Selected"] = _highlight.map({True: team_sel, False: "Other"})
            fig_gk = px.scatter(
                gk_ctx, x="Defence Quality", y="Goals Prevented",
                color="Selected",
                color_discrete_map={team_sel: "#e63946", "Other": "#457b9d"},
                hover_data=["GK", "Team", "Save %", "GK Value Add"],
                text="GK",
                title="Goals Prevented vs Defence Quality in Front of GK",
            )
            fig_gk.update_traces(textposition="top center", textfont_size=9, marker_size=10)

            # Trend line
            if len(_valid) >= 4:
                _xline = np.linspace(gk_ctx["Defence Quality"].min(), gk_ctx["Defence Quality"].max(), 50)
                _yline = np.polyval(_coeffs, _xline)
                fig_gk.add_scatter(
                    x=_xline, y=_yline, mode="lines",
                    line=dict(color="grey", dash="dash", width=1.5),
                    name="Expected GP (trend)", showlegend=True,
                )

            fig_gk.update_layout(
                template="plotly_white", height=500,
                xaxis_title="Team Defence Quality (Defender Composite Percentile)",
                yaxis_title="Goals Prevented (Saves Made − Goals Conceded)",
                legend_title="",
            )
            fig_gk.add_hline(y=0, line_color="lightgrey", line_width=1)
            st.plotly_chart(fig_gk, use_container_width=True)
            st.caption(
                "**Goals Prevented** = Saves Made − Goals Conceded. "
                "Positive = GK saves more than expected. Points above the trend line are "
                "GKs who **overperform given their defence quality** (high GK Value Add)."
            )

            # 5. Leaderboard table
            _show = gk_ctx[["GK", "Team", "Defence Quality", "Goals Prevented",
                            "Save %", "GK Value Add"]].copy()
            _show = _show.sort_values("GK Value Add", ascending=False).reset_index(drop=True)
            _show.index = _show.index + 1
            st.dataframe(
                _show.style.format({
                    "Defence Quality": "{:.1f}",
                    "Goals Prevented": "{:+.2f}",
                    "Save %": "{:.1f}%",
                    "GK Value Add": "{:+.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("Not enough GK data in this league for the Defence Quality analysis.")


# ── UI: Player Comparison ────────────────────────────────────────────────────

COMPARE_METRIC_GROUPS = {
    "⚔️ Attacking": OFFENSIVE_METRICS,
    "🛡️ Defending": DEFENSIVE_METRICS,
    "📊 Passing": PASSING_METRICS,
    "🏃 Dribbling & Carrying": DRIBBLING_METRICS,
    "🚀 Ball Progression": BALL_PROGRESSION_METRICS,
    "🧤 Goalkeeping": GK_METRICS,
    "🟨 Discipline": DISCIPLINE_METRICS,
}


def _find_similar_players(df, player_row, position, metrics, n=10, detail_positions=None):
    """Find the N most similar players by percentile-rank cosine similarity."""
    avail = [m for m in metrics if m in df.columns]
    if not avail:
        return pd.DataFrame()

    # Use granular posicion_detail filtering when a sub-group is specified
    if detail_positions and "posicion_detail" in df.columns:
        peers = df[df["posicion_detail"].isin(detail_positions)].copy()
    else:
        peers = df[df["posicion"] == position].copy()
    if len(peers) < 3:
        return pd.DataFrame()

    # Build percentile matrix for available metrics
    pct_df = peers[avail].rank(pct=True).fillna(0.5)

    # Player's own percentile vector — match by exact row index (not name) so
    # same-named players (e.g. André vs Amadou Onana) resolve correctly.
    _pid = player_row.name
    if _pid not in pct_df.index:
        _byname = peers.index[peers["nombre"] == player_row["nombre"]]
        if _byname.empty:
            return pd.DataFrame()
        _pid = _byname[0]
    player_vec = pct_df.loc[_pid].values.astype(float)

    # Euclidean similarity on the percentile vectors.  Cosine (angle-only) treats
    # an all-90th-pct player as similar to an all-50th-pct one and inflates
    # everything for these all-positive vectors; Euclidean measures closeness in
    # actual levels, so "similar" means similar output across the metrics.
    dist = np.linalg.norm(pct_df.values - player_vec, axis=1)
    max_dist = np.sqrt(len(avail)) or 1.0
    sim = (1.0 - dist / max_dist).clip(0, 1)

    peers = peers.copy()
    peers["Similarity %"] = (sim * 100).round(1)
    # Exclude only the selected player (keep any same-named different player)
    peers = peers[peers.index != _pid]
    return peers.nlargest(n, "Similarity %")


# ── Similarity profile definitions by position ──────────────────────────────

_SIMILARITY_PROFILES = {
    "Wingers": {
        "All-round": ["Goals", "Goal Assists", "Successful Dribbles", "Key Passes (Attempt Assists)",
                      "Successful Crosses & Corners", "Progressive Carries"],
        "Goal Scorer": ["Goals", "Total Shots", "Shots On Target ( inc goals )",
                        "Total Touches In Opposition Box"],
        "Creator": ["Goal Assists", "Key Passes (Attempt Assists)",
                    "Successful Crosses & Corners", "Total Big Chances Created"],
        "Dribbler": ["Successful Dribbles", "Progressive Carries", "Total Fouls Won",
                     "Goals", "Goal Assists"],
        "Goal Threat": ["Goals", "Total Shots", "Total Touches In Opposition Box",
                        "Total Big Chances Scored"],
    },
    "Attacking Midfield": {
        "All-round": ["Goals", "Goal Assists", "Key Passes (Attempt Assists)",
                      "Successful Dribbles", "Progressive Carries", "Through balls"],
        "Goal Scorer": ["Goals", "Total Shots", "Shots On Target ( inc goals )",
                        "Total Touches In Opposition Box"],
        "Creator": ["Goal Assists", "Key Passes (Attempt Assists)",
                    "Through balls", "Total Big Chances Created"],
        "Dribbler": ["Successful Dribbles", "Progressive Carries", "Total Fouls Won",
                     "Goals", "Goal Assists"],
    },
    "Striker": {
        "All-round": ["Goals", "Goal Assists", "Total Shots", "Key Passes (Attempt Assists)",
                      "Aerial Duels won"],
        "Goal Scorer": ["Goals", "Total Shots", "Shots On Target ( inc goals )",
                        "Total Big Chances Scored"],
        "Creator": ["Goal Assists", "Key Passes (Attempt Assists)",
                    "Total Big Chances Created", "Through balls"],
        "Target Man": ["Aerial Duels won", "Headed Goals", "Total Fouls Won",
                       "Goals", "Shots On Target ( inc goals )"],
    },
    "Central Midfield": {
        "All-round": ["Goals", "Goal Assists", "Key Passes (Attempt Assists)",
                      "Total Tackles", "Interceptions", "Total Passes"],
        "Box-to-Box": ["Goals", "Goal Assists", "Total Tackles", "Interceptions",
                       "Progressive Carries", "Recoveries"],
        "Deep Playmaker": ["Total Passes", "Forward Passes",
                           "Key Passes (Attempt Assists)", "Successful Long Passes",
                           "Interceptions"],
        "Creator": ["Goal Assists", "Key Passes (Attempt Assists)",
                    "Successful Crosses & Corners", "Through balls"],
        "Ball Winner": ["Total Tackles", "Interceptions", "Recoveries",
                        "Aerial Duels won"],
        "Shield": ["Interceptions", "Total Clearances", "Blocked Shots",
                   "Aerial Duels won", "Total Tackles"],
    },
    "Centre-Back": {
        "All-round": ["Total Tackles", "Interceptions", "Total Clearances",
                      "Aerial Duels won", "Total Passes"],
        "Ball-Playing CB": ["Total Passes", "Forward Passes",
                            "Successful Long Passes", "Interceptions", "Total Clearances"],
        "Duelist": ["Total Tackles", "Interceptions", "Total Clearances",
                    "Aerial Duels won", "Blocked Shots"],
    },
    "Full-Back": {
        "All-round": ["Total Tackles", "Goal Assists", "Successful Crosses & Corners",
                      "Progressive Carries", "Interceptions"],
        "Attacking Full-Back": ["Goal Assists", "Key Passes (Attempt Assists)",
                                "Successful Crosses & Corners", "Progressive Carries", "Goals"],
        "Defensive Full-Back": ["Total Tackles", "Interceptions", "Total Clearances",
                                "Aerial Duels won", "Blocked Shots"],
    },
    "Goalkeeper": {
        "All-round": ["Saves Made", "Goals Prevented", "Save %", "Launch %"],
    },
}

# Map profiles to specific posicion_detail sub-groups for more accurate matching
_PROFILE_POSITION_DETAIL = {}


# ── UI: GK Analysis (PSxG) ───────────────────────────────────────────────────

def render_gk_analysis(data):
    """PSxG-based goalkeeper performance analysis across all leagues."""
    df_total = data["total"]

    st.subheader("🧤 Goalkeeper Analysis")
    st.caption(
        "PSxG-based performance rankings for goalkeepers across Europe's top 7 leagues. "
        "PSxG is approximated from on-target shots faced, weighted by location "
        "(inside box × 0.34, big chances × 0.55, outside box × 0.12, penalties × 0.79)."
    )

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        gka_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="gka_stat_mode")
    with ctrl2:
        league_options = sorted(df_total["league_display"].dropna().unique())
        gka_leagues = st.multiselect("League", league_options, key="gka_leagues")
    with ctrl3:
        min_apps = st.slider("Min appearances", 1, 38, 10, key="gka_min_apps")

    src = _select_df(data, gka_mode)
    gks = src[src["posicion"] == "Goalkeeper"].copy()
    if gka_leagues:
        gks = gks[gks["league_display"].isin(gka_leagues)]
    gks = gks[gks["Appearances"].fillna(0) >= min_apps]

    if gks.empty or "PSxG" not in gks.columns:
        st.info("No PSxG data available. Ensure 2025-2026 season data is loaded.")
        return

    gks = gks[gks["PSxG"].notna() & (gks["PSxG"] > 0)].copy()
    if gks.empty:
        st.info("No PSxG data for the selected filters.")
        return

    scope_lbl = gka_mode

    # ── Standout Strengths spotlight ─────────────────────────────────────
    # Lead with a keeper's elite traits — surfaced at the metric level so
    # strong command / big-chance saving shows even when overall shot-stopping
    # rates are average.
    st.markdown(f"### 🌟 Goalkeeper Spotlight ({scope_lbl})")
    _gk_names = sorted(gks["nombre"].unique())
    _spot_gk = st.selectbox("Highlight a goalkeeper", _gk_names, key="gka_spotlight")
    if _spot_gk:
        _spot_row = dict(gks[gks["nombre"] == _spot_gk].iloc[0])
        _spot_strengths = _compute_standout_strengths(_spot_row, gks, True)
        if _spot_strengths:
            st.caption(f"Where **{_spot_gk}** ranks among the filtered goalkeepers — percentile vs peers.")
            _spot_cols = st.columns(len(_spot_strengths))
            for _c, _s in zip(_spot_cols, _spot_strengths):
                _v = _s["value"]
                _vstr = f"{_v:.1f}" if isinstance(_v, (int, float)) else str(_v)
                with _c:
                    st.markdown(
                        f"<div style='background:#1a1a2e;border-left:4px solid {_s['color']};"
                        f"border-radius:8px;padding:12px 14px;'>"
                        f"<div style='font-size:12px;color:#aaa;height:32px;'>{_s['label']}</div>"
                        f"<div style='font-size:30px;font-weight:bold;color:{_s['color']};line-height:1.1;'>"
                        f"{_s['pct']:.0f}<span style='font-size:13px;'>{_ord_suffix(_s['pct'])}</span></div>"
                        f"<div style='font-size:11px;color:#ccc;'>{_s['tier']} · {_vstr}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(f"{_spot_gk} has no metric ranking in the top 30% of the filtered pool.")

    # ── Scatter: PSxG/Shot vs PSxG+/- ────────────────────────────────────
    st.markdown(f"### 📊 Shot Difficulty vs Goals Prevented ({scope_lbl})")
    st.caption(
        "**X-axis:** average xG per shot faced (higher = harder shots faced). "
        "**Y-axis:** PSxG+/- — goals prevented above expectation (positive = elite)."
    )

    if "PSxG+/-" in gks.columns and "PSxG/Shot" in gks.columns:
        sc_df = gks[["nombre", "equipo", "league_display", "PSxG", "PSxG+/-",
                     "PSxG/Shot", "Goals Conceded", "Save %", "Appearances"]].dropna(
            subset=["PSxG+/-", "PSxG/Shot"])

        fig_scatter = px.scatter(
            sc_df,
            x="PSxG/Shot",
            y="PSxG+/-",
            text="nombre",
            color="league_display",
            color_discrete_sequence=CHART_COLORS,
            hover_data={
                "nombre": True, "equipo": True, "Appearances": True,
                "PSxG": ":.1f", "Goals Conceded": True, "Save %": ":.1f",
                "PSxG/Shot": ":.3f", "PSxG+/-": ":.1f",
            },
            labels={
                "PSxG/Shot": "PSxG/Shot (Shot Difficulty ↑)",
                "PSxG+/-": "PSxG+/- — Goals Prevented (↑ Better)",
                "league_display": "League",
            },
        )
        fig_scatter.update_traces(
            textposition="top center",
            textfont_size=9,
            marker=dict(size=9, opacity=0.85),
        )
        fig_scatter.add_hline(
            y=0, line_dash="dash",
            line_color="rgba(255,255,255,0.35)",
            annotation_text="Breakeven", annotation_position="bottom right",
        )
        x_mid = sc_df["PSxG/Shot"].median()
        fig_scatter.add_vline(x=x_mid, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_scatter.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#0d1b2a",
            font=dict(color="#eee"), height=580,
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title_font_size=12),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title_font_size=12),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Bar chart: PSxG+/- ranking ────────────────────────────────────────
    st.markdown(f"### 🏆 PSxG+/- Ranking — Goals Prevented ({scope_lbl})")

    if "PSxG+/-" in gks.columns:
        top_n = st.slider("Show top / bottom N goalkeepers", 5, 30, 15, key="gka_top_n")
        rank_df = gks[["nombre", "equipo", "league_display", "PSxG+/-", "PSxG",
                        "Goals Conceded", "Save %", "Appearances"]].dropna(subset=["PSxG+/-"])
        rank_df = rank_df.sort_values("PSxG+/-", ascending=False)
        top_df = pd.concat([rank_df.head(top_n), rank_df.tail(top_n)]).drop_duplicates()
        top_df = top_df.sort_values("PSxG+/-", ascending=True).copy()

        top_df["bar_color"] = top_df["PSxG+/-"].apply(
            lambda v: "#00e676" if v >= 0 else "#e63946"
        )
        top_df["label"] = (top_df["nombre"] + "  ·  "
                           + top_df["equipo"].str[:18]
                           + "  [" + top_df["league_display"].str[:3].str.upper() + "]")

        fig_bar = go.Figure(go.Bar(
            x=top_df["PSxG+/-"],
            y=top_df["label"],
            orientation="h",
            marker_color=top_df["bar_color"].tolist(),
            text=top_df["PSxG+/-"].apply(lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "PSxG+/-: %{x:.1f}<br>"
                "<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#0d1b2a",
            font=dict(color="#eee"),
            height=max(420, len(top_df) * 25 + 80),
            xaxis=dict(
                title="PSxG+/- (Goals Prevented)",
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=True, zerolinecolor="rgba(255,255,255,0.4)",
            ),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=260, r=70, t=30, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Full stats table ──────────────────────────────────────────────────
    st.markdown("### 📋 Full GK PSxG Table")
    _tbl_cols = [
        "nombre", "equipo", "league_display", "Appearances",
        "PSxG", "Goals Conceded", "PSxG+/-", "PSxG/Shot",
        "Save %", "Inside Box Save %", "Outside Box Save %",
        "Total Big Chances Saved", "Penalties Faced", "Penalties Saved",
    ]
    _tbl_cols = [c for c in _tbl_cols if c in gks.columns]
    tbl_df = (
        gks[_tbl_cols]
        .sort_values("PSxG+/-", ascending=False)
        .rename(columns={"nombre": "Player", "equipo": "Team", "league_display": "League"})
        .reset_index(drop=True)
    )
    tbl_df.index += 1
    st.dataframe(tbl_df.round(2), use_container_width=True)


def render_player_comparison(data):
    st.subheader("⚔️ Player Comparison")

    compare_mode = st.radio(
        "Mode",
        ["🔍 Find Similar Players", "⚔️ Compare Players", "📅 Cross-Season Compare"],
        horizontal=True, key="cmp_mode_sel",
    )

    # ── Stat mode ────────────────────────────────────────────────────────
    stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="cmp_stat_mode")
    df = _select_df(data, stat_mode)
    mode_label = f" ({stat_mode})" if stat_mode != "Total" else ""

    if compare_mode == "🔍 Find Similar Players":
        _render_find_similar(df, mode_label)
    elif compare_mode == "📅 Cross-Season Compare":
        _render_cross_season(stat_mode, mode_label)
    else:
        _render_head_to_head(df, mode_label)


def _chart_radar_rows(rows, metrics, title="Comparison"):
    """Radar chart from explicit player rows (each a Series with a '_label'),
    normalized 0-100 vs the max among the selected players. Used for cross-season
    comparison where players come from different season DataFrames."""
    sel_max = {}
    for m in metrics:
        sel_max[m] = max((0 if pd.isna(r.get(m, 0)) else (r.get(m, 0) or 0)) for r in rows) or 1
    fig = go.Figure()
    for row in rows:
        vals, raw_vals = [], []
        for m in metrics:
            v = row.get(m, 0)
            v = 0 if pd.isna(v) else v
            raw_vals.append(round(v, 2))
            vals.append(round(v / sel_max[m] * 100, 1) if sel_max[m] > 0 else 0)
        vals.append(vals[0])
        raw_vals.append(raw_vals[0])
        label = row.get("_label", row.get("nombre", "?"))
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]], fill="toself", name=label, opacity=0.6,
            customdata=raw_vals, hoveron="points", marker=dict(size=6),
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra>" + label + "</extra>",
        ))
    _h = max(560, 420 + len(metrics) * 20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 115], dtick=20)),
        title=title, height=_h, template="plotly_white", showlegend=True, hovermode="closest",
    )
    return fig


def _player_options(df_total):
    """[(label, nombre, equipo)] for a player picker. Names shared by more than
    one player (e.g. 'A. Onana' = André GK and Amadou MF) are disambiguated with
    team + position so each is individually selectable."""
    cols = [c for c in ("nombre", "equipo", "posicion") if c in df_total.columns]
    sub = df_total[cols].dropna(subset=["nombre"]).drop_duplicates(subset=["nombre", "equipo"])
    counts = sub["nombre"].value_counts()
    opts = []
    for _, r in sub.sort_values(["nombre", "equipo"]).iterrows():
        nm, tm, pos = r["nombre"], r.get("equipo", ""), r.get("posicion", "")
        label = f"{nm} — {tm} ({pos})" if counts.get(nm, 0) > 1 else nm
        # Append a typeable ASCII alias for accented names so e.g. "Sesko"
        # finds "B. Šeško" in the selectbox's type-to-search.
        ascii_nm = _ascii_name(nm)
        if ascii_nm and ascii_nm.lower() != str(nm).lower():
            label = f"{label}  ·  {ascii_nm}"
        opts.append((label, nm, tm))
    return opts


def _render_cross_season(stat_mode, mode_label):
    """Compare players from different seasons side by side."""
    st.markdown("---")
    st.markdown("#### Compare players across different seasons")
    st.caption("Pick each player and the season they played — values are normalized "
               "vs the selected players, so it's a fair like-for-like stat comparison.")

    seasons = get_available_seasons()
    n = st.slider("Number of players", 2, 4, 2, key="xs_n")

    picks = []  # (season, nombre, equipo)
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            sea = st.selectbox(f"Season {i + 1}", seasons, key=f"xs_season_{i}")
            opts = _player_options(load_data(sea)["total"])
            labels = [o[0] for o in opts]
            lbl = st.selectbox(f"Player {i + 1}", labels, index=None,
                               placeholder="Select a player…", key=f"xs_player_{i}")
            if lbl:
                _, nm, tm = opts[labels.index(lbl)]
                picks.append((sea, nm, tm))

    if len(picks) < 2:
        st.info("Select at least **2 players** (each with a season) to compare.")
        return

    # Build one row per pick from that season's selected stat frame
    rows = []
    for sea, nm, tm in picks:
        sdf = _select_df(load_data(sea), stat_mode)
        match = sdf[(sdf["nombre"] == nm) & (sdf["equipo"] == tm)]
        if match.empty:
            match = sdf[sdf["nombre"] == nm]
        if not match.empty:
            if "Time Played" in match.columns:
                match = match.sort_values("Time Played", ascending=False)
            row = match.iloc[0].copy()
            row["_label"] = f"{nm} · {sea}"
            rows.append(row)
    if len(rows) < 2:
        st.warning("Couldn't load the selected players.")
        return

    # Metric group, restricted to metrics present across all selected seasons
    group_sel = st.selectbox("Metric group", list(COMPARE_METRIC_GROUPS.keys()), key="xs_group")
    common = set.intersection(*[set(r.index) for r in rows])
    sel_metrics = [m for m in COMPARE_METRIC_GROUPS[group_sel] if m in common]
    if len(sel_metrics) < 3:
        st.warning("Not enough metrics common to all selected seasons for this group "
                   "(older seasons track fewer metrics). Try another group.")
        return

    # ── Radar ──
    st.markdown(f"### 🕸️ Radar — Cross-Season{mode_label}")
    st.plotly_chart(_chart_radar_rows(rows, sel_metrics, f"Cross-Season Comparison{mode_label}"),
                    use_container_width=True)

    # ── Side-by-side table ──
    st.markdown("### 📋 Side-by-Side Stats")
    tbl = pd.DataFrame({r["_label"]: {m: round(r.get(m, 0) or 0, 2) for m in sel_metrics} for r in rows})
    st.dataframe(tbl, use_container_width=True)

    # ── Head-to-head difference (exactly 2 players) ──
    if len(rows) == 2:
        st.markdown("### 📊 Head-to-Head Difference")
        diffs = [round((rows[0].get(m, 0) or 0) - (rows[1].get(m, 0) or 0), 2) for m in sel_metrics]
        fig_diff = go.Figure(go.Bar(
            x=sel_metrics, y=diffs,
            marker_color=["#2d6a4f" if d >= 0 else "#e63946" for d in diffs],
            text=[f"+{d}" if d > 0 else str(d) for d in diffs], textposition="outside",
        ))
        fig_diff.update_layout(
            title=f"{rows[0]['_label']} minus {rows[1]['_label']}",
            template="plotly_white", height=420, xaxis_tickangle=-45,
            yaxis_title="Difference", margin=dict(b=120),
        )
        st.plotly_chart(fig_diff, use_container_width=True)


def _render_find_similar(df, mode_label):
    """Find players with a similar statistical profile."""
    st.markdown("---")
    st.markdown("#### Select a player to find similar profiles")

    # ── Player search ────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        _opts = _player_options(df)
        _labels = [o[0] for o in _opts]
        _sel_label = st.selectbox("Type or select a player", _labels, index=None,
                                  placeholder="Start typing a name...", key="sim_player")
    with c2:
        n_results = st.slider("Number of similar players", 5, 20, 10, key="sim_n")

    if not _sel_label:
        st.info("Select a player above to find players with a similar statistical profile.")
        return

    player_sel, _sel_team = _opts[_labels.index(_sel_label)][1:]
    player_rows = df[(df["nombre"] == player_sel) & (df["equipo"] == _sel_team)]
    if player_rows.empty:
        player_rows = df[df["nombre"] == player_sel]
    if player_rows.empty:
        st.warning("Player not found.")
        return
    player_row = player_rows.iloc[0]
    position = player_row.get("posicion", "Striker")
    detail = player_row.get("posicion_detail", "")

    # ── Profile / playstyle selector ─────────────────────────────────────
    # Use role-based profiles for outfield players, legacy for GK
    if position in POSITION_ROLE_PROFILES:
        role_profiles = POSITION_ROLE_PROFILES[position]
        legacy = _SIMILARITY_PROFILES.get(position, {})
        profiles = {}
        if "All-round" in legacy:
            profiles["All-round"] = legacy["All-round"]
        profiles.update(role_profiles)
    else:
        profiles = _SIMILARITY_PROFILES.get(position, _SIMILARITY_PROFILES["Striker"])

    # Auto-detect the player's role and default-select it
    detected_role = _classify_role(player_row, position, df)
    profile_keys = list(profiles.keys())
    default_idx = profile_keys.index(detected_role) if detected_role in profile_keys else 0

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        profile_sel = st.selectbox("Playstyle profile", profile_keys,
                                   index=default_idx, key="sim_profile")
    with c2:
        scope = st.radio("Search scope", ["Same position", "All positions"],
                         horizontal=True, key="sim_scope")
    with c3:
        all_leagues = sorted(df["league_display"].dropna().unique())
        league_filter = st.multiselect("League(s)", all_leagues, default=all_leagues,
                                       key="sim_league")

    base_metrics = profiles[profile_sel]
    avail_metrics = [m for m in base_metrics if m in df.columns]

    if not avail_metrics:
        st.warning("Not enough data for this profile.")
        return

    # ── Find similar ─────────────────────────────────────────────────────
    # Check if this profile maps to specific sub-positions (e.g. Full-Back -> LB, RB)
    detail_positions = _PROFILE_POSITION_DETAIL.get(profile_sel)

    # Apply league filter
    if league_filter:
        df = df[df["league_display"].isin(league_filter)]

    if scope == "All positions":
        # Search all positions, no position filter at all
        peers = df.copy()
        avail = avail_metrics
        pct_df = peers[avail].rank(pct=True).fillna(0.5)
        _pid = player_row.name if player_row.name in pct_df.index else None
        if _pid is None:
            _byname = peers.index[peers["nombre"] == player_row["nombre"]]
            _pid = _byname[0] if not _byname.empty else None
        similar = pd.DataFrame()
        if _pid is not None:
            player_vec = pct_df.loc[_pid].values.astype(float)
            # Euclidean similarity (see _find_similar_players) — closeness in
            # actual percentile levels, not just direction.
            dist = np.linalg.norm(pct_df.values - player_vec, axis=1)
            max_dist = np.sqrt(len(avail)) or 1.0
            sim = (1.0 - dist / max_dist).clip(0, 1)
            peers = peers.copy()
            peers["Similarity %"] = (sim * 100).round(1)
            peers = peers[peers.index != _pid]
            similar = peers.nlargest(n_results, "Similarity %")
    else:
        similar = _find_similar_players(df, player_row, position,
                                        avail_metrics, n=n_results,
                                        detail_positions=detail_positions)

    if similar.empty:
        st.warning("Not enough peers to compute similarity.")
        return

    # ── Selected player card ─────────────────────────────────────────────
    st.markdown("---")
    _season = st.session_state.get("sel_season", CURRENT_SEASON)
    st.markdown(f"### Players similar to **{player_sel}**  ·  📅 {_season}")
    st.caption(f"Season: **{_season}** | Profile: **{profile_sel}** | Metrics: {', '.join(avail_metrics)} | "
               f"Position: {position} {'(all positions searched)' if scope == 'All positions' else ''}")

    # ── Results table ────────────────────────────────────────────────────
    similar = similar.copy()
    similar["Season"] = _season
    show_cols = ["nombre", "Season", "equipo", "league_display", "posicion_detail", "Similarity %"] + avail_metrics
    show_cols = [c for c in show_cols if c in similar.columns]
    show = similar[show_cols].reset_index(drop=True).copy()
    # Missing values (metric not recorded / not tracked for that player or season,
    # e.g. Progressive Carries in older seasons) → show "—" instead of a blank/NaN.
    _missing_any = False
    for m in avail_metrics:
        if m in show.columns:
            if show[m].isna().any():
                _missing_any = True
            show[m] = show[m].map(lambda v: "—" if pd.isna(v) else (round(v, 2) if isinstance(v, float) else v))
    st.dataframe(show, use_container_width=True)
    if _missing_any:
        st.caption("“—” = metric not recorded for that player/season (e.g. some advanced "
                   "metrics aren't tracked in older seasons). Similarity treats these as average.")

    # ── Radar: target player vs top 3 similar (explicit rows avoid name clashes)
    _tgt = player_row.copy()
    _tgt["_label"] = player_sel
    _radar_rows = [_tgt]
    for _, sr in similar.head(3).iterrows():
        rr = sr.copy()
        rr["_label"] = f"{sr['nombre']} ({sr.get('equipo', '')})"
        _radar_rows.append(rr)
    st.markdown(f"### 🕸️ Radar: {player_sel} vs Top Matches{mode_label}")
    st.plotly_chart(
        _chart_radar_rows(_radar_rows, avail_metrics, f"{player_sel} vs Similar Players{mode_label}"),
        use_container_width=True,
    )

    # ── Similarity bar chart ─────────────────────────────────────────────
    st.markdown("### 📊 Similarity Ranking")
    fig_sim = go.Figure(go.Bar(
        x=similar["nombre"].head(n_results),
        y=similar["Similarity %"].head(n_results),
        marker_color=["#2d6a4f" if s >= 90 else "#e9c46a" if s >= 75 else "#e76f51"
                       for s in similar["Similarity %"].head(n_results)],
        text=[f"{s:.1f}%" for s in similar["Similarity %"].head(n_results)],
        textposition="outside",
    ))
    fig_sim.update_layout(
        title=f"Most Similar to {player_sel}",
        yaxis_title="Similarity %", yaxis_range=[0, 105],
        template="plotly_white", height=420, xaxis_tickangle=-45,
        margin=dict(b=120),
    )
    st.plotly_chart(fig_sim, use_container_width=True)


def _render_head_to_head(df, mode_label):
    """Original head-to-head comparison."""

    # ── Filters ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_leagues = st.multiselect("Filter by league", sorted(df["league_display"].unique()),
                                     default=[], key="cmp_lg")
    with c2:
        team_pool = df[df["league_display"].isin(sel_leagues)] if sel_leagues else df
        team_opts = sorted([t for t in team_pool["equipo"].dropna().unique() if t != "Unknown"])
        sel_teams = st.multiselect("Filter by team", team_opts, default=[], key="cmp_tm")
    with c3:
        pos_opts = sorted(df["posicion"].dropna().unique())
        sel_pos = st.multiselect("Filter by position", pos_opts, default=[], key="cmp_pos")

    filt = filter_df(df, sel_leagues or None, sel_pos or None, teams=sel_teams or None)

    # ── Player selection (team-disambiguated for same-named players) ──────
    _opts = _player_options(filt)
    _labels = [o[0] for o in _opts]
    selected = st.multiselect("Select players to compare (2–6)", _labels,
                              max_selections=6, key="cmp_players")

    if len(selected) < 2:
        st.info("Pick at least **2 players** to start the comparison.")
        return

    _lab2pl = {o[0]: (o[1], o[2]) for o in _opts}
    sel_rows = []
    for lbl in selected:
        nm, tm = _lab2pl[lbl]
        m = filt[(filt["nombre"] == nm) & (filt["equipo"] == tm)]
        if m.empty:
            m = filt[filt["nombre"] == nm]
        if not m.empty:
            r = m.iloc[0].copy()
            r["_label"] = lbl
            sel_rows.append(r)
    if len(sel_rows) < 2:
        st.warning("Couldn't load the selected players.")
        return

    # ── Metric selection ─────────────────────────────────────────────────
    st.markdown("#### Choose metrics")
    metric_mode = st.radio("Metric selection", ["Pick a preset group", "Build your own"],
                           horizontal=True, key="cmp_metric_mode")

    if metric_mode == "Pick a preset group":
        group_names = list(COMPARE_METRIC_GROUPS.keys())
        group_sel = st.selectbox("Metric group", group_names, key="cmp_group")
        raw_metrics = COMPARE_METRIC_GROUPS[group_sel]
        sel_metrics = [m for m in raw_metrics if m in filt.columns]
    else:
        all_metrics = []
        for g, mlist in COMPARE_METRIC_GROUPS.items():
            all_metrics.extend(m for m in mlist if m in filt.columns and m not in all_metrics)
        sel_metrics = st.multiselect("Select metrics (3–10 recommended)", all_metrics,
                                     default=all_metrics[:6] if len(all_metrics) >= 6 else all_metrics,
                                     key="cmp_custom_metrics")

    if len(sel_metrics) < 3:
        st.warning("Please select at least 3 metrics for a meaningful radar chart.")
        return

    # ── Radar chart ──────────────────────────────────────────────────────
    st.markdown(f"### 🕸️ Radar Comparison{mode_label}")
    st.plotly_chart(_chart_radar_rows(sel_rows, sel_metrics, f"Player Comparison{mode_label}"),
                    use_container_width=True)

    # ── Side-by-side stats table ─────────────────────────────────────────
    st.markdown("### 📋 Side-by-Side Stats")
    tbl = pd.DataFrame([
        {"Player": r["_label"], "Team": r.get("equipo"), "League": r.get("league_display"),
         "Pos": r.get("posicion_detail"),
         **{m: round(r.get(m, 0) or 0, 2) for m in sel_metrics}}
        for r in sel_rows
    ])
    st.dataframe(tbl, use_container_width=True)

    # ── Difference bar chart ─────────────────────────────────────────────
    if len(sel_rows) == 2:
        st.markdown("### 📊 Head-to-Head Difference")
        diffs = [round((sel_rows[0].get(m, 0) or 0) - (sel_rows[1].get(m, 0) or 0), 2) for m in sel_metrics]
        fig_diff = go.Figure(go.Bar(
            x=sel_metrics, y=diffs,
            marker_color=["#2d6a4f" if d >= 0 else "#e63946" for d in diffs],
            text=[f"+{d}" if d > 0 else str(d) for d in diffs],
            textposition="outside",
        ))
        fig_diff.update_layout(
            title=f"{sel_rows[0]['_label']} minus {sel_rows[1]['_label']}",
            template="plotly_white", height=420, xaxis_tickangle=-45,
            yaxis_title="Difference", margin=dict(b=120),
        )
        st.plotly_chart(fig_diff, use_container_width=True)


# ── UI: Team Comparison ──────────────────────────────────────────────────────

def render_team_comparison(data):
    st.subheader("🏟️ Team Comparison")

    # ── Stat mode ────────────────────────────────────────────────────────
    stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="tcmp_stat_mode")
    df = _select_df(data, stat_mode)
    mode_label = f" ({stat_mode})" if stat_mode != "Total" else ""

    # ── League filter ────────────────────────────────────────────────────
    sel_leagues = st.multiselect("Filter by league", sorted(df["league_display"].unique()),
                                 default=[], key="tcmp_lg")
    pool = df[df["league_display"].isin(sel_leagues)] if sel_leagues else df
    pool = pool[pool["equipo"] != "Unknown"]

    # ── Team selection ───────────────────────────────────────────────────
    all_teams = sorted(pool["equipo"].dropna().unique())
    selected_teams = st.multiselect("Select teams to compare (2–6)", all_teams,
                                    max_selections=6, key="tcmp_teams")

    if len(selected_teams) < 2:
        st.info("Pick at least **2 teams** to start the comparison.")
        return

    # ── Aggregation method ───────────────────────────────────────────────
    agg_method = st.radio("Aggregation", ["Team Total (sum)", "Per-Player Average (mean)"],
                          horizontal=True, key="tcmp_agg")
    agg_func = "sum" if "sum" in agg_method.lower() else "mean"

    # ── Metric selection ─────────────────────────────────────────────────
    st.markdown("#### Choose metrics")
    metric_mode = st.radio("Metric selection", ["Pick a preset group", "Build your own"],
                           horizontal=True, key="tcmp_metric_mode")

    if metric_mode == "Pick a preset group":
        group_names = list(TEAM_STAT_CATEGORIES.keys())
        group_sel = st.selectbox("Metric group", group_names, key="tcmp_group")
        raw_metrics = TEAM_STAT_CATEGORIES[group_sel]
        sel_metrics = [m for m in raw_metrics if m in pool.columns]
    else:
        all_metrics = []
        for g, mlist in TEAM_STAT_CATEGORIES.items():
            all_metrics.extend(m for m in mlist if m in pool.columns and m not in all_metrics)
        sel_metrics = st.multiselect("Select metrics (3–10 recommended)", all_metrics,
                                     default=all_metrics[:6] if len(all_metrics) >= 6 else all_metrics,
                                     key="tcmp_custom_metrics")

    if len(sel_metrics) < 3:
        st.warning("Please select at least 3 metrics for a meaningful radar chart.")
        return

    # ── Aggregate team data ──────────────────────────────────────────────
    team_data = pool[pool["equipo"].isin(selected_teams)]
    if agg_func == "sum":
        team_agg = team_data.groupby("equipo")[sel_metrics].sum()
    else:
        team_agg = team_data.groupby("equipo")[sel_metrics].mean()
    team_agg = team_agg.round(2)

    # ── Radar chart ──────────────────────────────────────────────────────
    st.markdown(f"### 🕸️ Team Radar Comparison{mode_label}")
    fig = go.Figure()
    for team in selected_teams:
        if team not in team_agg.index:
            continue
        row = team_agg.loc[team]
        vals = []
        raw_vals = []
        for m in sel_metrics:
            v = row.get(m, 0) or 0
            raw_vals.append(round(v, 2))
            mx = team_agg[m].max() if m in team_agg.columns else 1
            vals.append(round(v / mx * 100, 1) if mx > 0 else 0)
        vals.append(vals[0])
        raw_vals.append(raw_vals[0])
        league_label = pool[pool["equipo"] == team]["league_display"].iloc[0] if not pool[pool["equipo"] == team].empty else ""
        trace_name = f"{team} ({league_label})"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=sel_metrics + [sel_metrics[0]],
            fill="toself", name=trace_name, opacity=0.6,
            customdata=raw_vals,
            hoveron="points", marker=dict(size=6),
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra>" + trace_name + "</extra>",
        ))
    _team_radar_h = max(580, 420 + len(sel_metrics) * 20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 115], dtick=20)),
        title=f"Team Comparison — {agg_method}{mode_label}",
        height=_team_radar_h, template="plotly_white", showlegend=True,
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Stats table ──────────────────────────────────────────────────────
    st.markdown("### 📋 Team Stats Table")
    display_df = team_agg.loc[[t for t in selected_teams if t in team_agg.index]].copy()
    display_df.index.name = "Team"
    st.dataframe(display_df.reset_index(), use_container_width=True)

    # ── Grouped bar chart ────────────────────────────────────────────────
    if len(selected_teams) <= 4:
        st.markdown("### 📊 Metric Breakdown")
        fig_bar = go.Figure()
        colors = ["#2d6a4f", "#e9c46a", "#e63946", "#457b9d", "#264653", "#f4a261"]
        for i, team in enumerate(selected_teams):
            if team not in team_agg.index:
                continue
            row = team_agg.loc[team]
            fig_bar.add_trace(go.Bar(
                name=team, x=sel_metrics,
                y=[round(row.get(m, 0), 2) for m in sel_metrics],
                marker_color=colors[i % len(colors)],
            ))
        fig_bar.update_layout(
            barmode="group",
            title=f"Team Comparison — {agg_method}{mode_label}",
            template="plotly_white", height=450, xaxis_tickangle=-45,
            margin=dict(b=120),
        )
        st.plotly_chart(fig_bar, use_container_width=True)



# ── Potential Grading Helpers ─────────────────────────────────────────────────

def _get_age_growth(age):
    """Return (annual_percentile_growth_pts, career_phase_label) for a player of *age*.

    Growth is measured in raw percentile points added (or subtracted) per year.
    Very young players gain the most; players past 29 begin to regress.
    """
    if age <= 17:  return  9.0, "🚀 Prodigy"
    if age <= 19:  return  7.0, "🚀 High Potential"
    if age <= 21:  return  5.5, "📈 Rising Star"
    if age <= 23:  return  3.5, "📈 Developing"
    if age <= 25:  return  1.5, "📈 Growing"
    if age <= 27:  return  0.5, "⭐ Prime"
    if age <= 29:  return -1.5, "⭐ Late Prime"
    if age <= 31:  return -3.0, "📉 Declining"
    if age <= 33:  return -4.5, "📉 Fading"
    return -6.0, "📉 Late Career"


def _overall_pct_for_row(row_data, position, season_df, role):
    """Europe-wide overall percentile for one player in one season's data —
    the same KPI/position-weighted calculation the Potential tab uses for the
    current season, reused to build the multi-season trajectory."""
    g = _compute_attribute_grades(row_data, position, season_df, league=None, kpi_role=role)
    if position == "Goalkeeper":
        weights = _ROLE_GRADE_WEIGHTS.get(role, {})
    else:
        kpi = _ROLE_KPI_PROFILES.get(role)
        weights = {n: w for n, (w, _) in kpi.items()} if kpi else _POSITION_GRADE_WEIGHTS.get(position, {})
    pcts = {k: pct for k, (_, pct) in g.items() if pct is not None}
    if not pcts:
        return 0.0
    if weights:
        ws = sum(weights.get(k, 0) for k in pcts)
        base = (sum(pcts[k] * weights.get(k, 0) for k in pcts) / ws
                if ws > 0 else sum(pcts.values()) / len(pcts))
    else:
        base = sum(pcts.values()) / len(pcts)
    # Apply the same Key-Strengths uplift the Lab/Profile grades use, so the
    # trajectory and Potential grade line up with the headline grade elsewhere.
    pos_peers = season_df[season_df["posicion"] == position]
    ks = _compute_key_strength_bonus(row_data, pos_peers, position == "Goalkeeper")
    return min(99.9, base + ks)


@st.cache_data(show_spinner="Building career trajectory…")
def _build_trajectory(player_id, min_minutes=600, stat_mode="Total", padj=False,
                      _schema=_ROLE_SCHEMA_VERSION, _bust=0):
    """Per-season overall percentile for a player, tracked by stable Opta id.
    Returns chronological [{season, pct, mins, team, role, reliable}].
    Seasons below *min_minutes* are kept (for context) but flagged unreliable.

    *stat_mode*: "Total" grades on season totals; "Per 90" grades the player's
    per-90 row against the season's ≥900-minute starters (the fair per-90 pool).
    *padj*: grade on the possession-adjusted frame (helps defensive output).
    Role is always classified from raw totals (volume-based) for consistency."""
    is_p90 = stat_mode == "Per 90"
    g_key = (("padj_per90" if padj else "per90") if is_p90
             else ("padj" if padj else "total"))
    traj = []
    for sea in sorted(get_available_seasons()):  # chronological ascending
        full = load_data(sea)
        dt = full["total"]
        if dt.empty or "id" not in dt.columns:
            continue
        mt = dt[dt["id"] == player_id]
        if mt.empty:
            continue
        rt = mt.iloc[0]
        mins = int(rt.get("Time Played", 0) or 0)
        pos = rt.get("posicion", "Unknown")
        role = _classify_role(rt, pos, dt)            # role from raw totals
        reliable = mins >= min_minutes
        pct = None
        if reliable:
            dg = full.get(g_key, dt)
            mg = dg[dg["id"] == player_id] if dg is not None else dt.iloc[0:0]
            if not mg.empty:
                pool = (dg[(dg["estimated_90s"].fillna(0) >= 10) | (dg["id"] == player_id)]
                        if is_p90 else dg)
                pct = round(_overall_pct_for_row(dict(mg.iloc[0]), pos, pool, role), 1)
        traj.append({"season": sea, "pct": pct, "mins": mins,
                     "team": rt.get("equipo", ""), "role": role, "reliable": reliable})
    return traj


def _trajectory_momentum(reliable_pcts):
    """Annual momentum (pctl pts/yr) + confidence (0-1) from a chronological list
    of reliable season percentiles. Slope via least-squares, clamped to ±8."""
    n = len(reliable_pcts)
    if n < 2:
        return 0.0, 0.0, n
    xs = list(range(n))
    slope = float(np.polyfit(xs, reliable_pcts, 1)[0])
    slope = max(-8.0, min(8.0, slope))
    conf = {2: 0.5, 3: 0.75}.get(n, 0.9)  # 4+ seasons → high confidence
    return round(slope, 2), conf, n


def _project_potential(current_pct, age, club_tier, years=3, momentum=0.0, momentum_conf=0.0):
    """Year-by-year potential projection.

    Blends the age-development curve with the player's observed form *momentum*
    (from their multi-season trajectory). Momentum dominates the near term in
    proportion to its confidence, then regresses toward the age curve over the
    horizon. With momentum_conf=0 this reduces to the pure age-curve model.

    Club boost is phased in: Yr1 30 % (minus the adaptation dip), Yr2 65 %, Yr3+ 100 %.
    """
    tier = _CLUB_TIERS.get(club_tier, _CLUB_TIERS["No major change"])
    ceiling_boost = tier["ceiling_boost"]
    year1_dip = tier["year1_dip"]
    _MOM_DECAY = 0.55  # momentum's weight halves-ish each projected year

    _phase_label = _get_age_growth(age)[1]
    projections = [
        {
            "year": "Now",
            "pct": current_pct,
            "grade": _percentile_to_grade(current_pct),
            "phase": _phase_label,
            "notes": "Current performance",
        }
    ]

    cumulative_growth = 0.0
    for yr in range(1, years + 1):
        yr_growth, yr_phase = _get_age_growth(age + yr)
        # Blend age growth with observed momentum (decaying weight by year).
        mw = momentum_conf * (_MOM_DECAY ** (yr - 1))
        eff_growth = (1 - mw) * yr_growth + mw * momentum
        cumulative_growth += eff_growth

        # Club environment effect — phased in over 3 seasons
        if ceiling_boost != 0:
            if yr == 1:
                club_effect = ceiling_boost * 0.30 - year1_dip
            elif yr == 2:
                club_effect = ceiling_boost * 0.65
            else:
                club_effect = ceiling_boost
        else:
            club_effect = 0.0

        projected = max(0.0, min(99.0, round(current_pct + cumulative_growth + club_effect, 1)))

        notes = []
        if club_effect < 0:
            notes.append(f"Adaptation dip ({club_effect:+.0f})")
        elif club_effect > 0:
            notes.append(f"Club boost (+{club_effect:.0f})")
        if mw >= 0.1 and abs(momentum) >= 0.5:
            notes.append(f"Form trend ({momentum:+.1f}/yr)")
        elif yr_growth > 0:
            notes.append(f"Age growth (+{yr_growth:.1f}/yr)")
        elif yr_growth < 0:
            notes.append(f"Age decline ({yr_growth:.1f}/yr)")

        projections.append({
            "year": f"+{yr} yr{'s' if yr > 1 else ''}",
            "pct": projected,
            "grade": _percentile_to_grade(projected),
            "phase": yr_phase,
            "notes": " · ".join(notes) if notes else "Stable",
        })

    return projections


# ── UI: Potential Grading ─────────────────────────────────────────────────────

def render_potential_grading(data, is_current=True):
    """Project a player's future grade from their age, current form, and club environment."""
    df_total = data["total"]
    st.subheader("🌟 Potential Grading")
    if not is_current:
        st.info(
            "🔒 Potential Grading is available for the current season only — it "
            "projects a player's trajectory from their **current** form. Switch the "
            "season selector back to the current season to use it."
        )
        return
    st.caption(
        "Forecasts a player's grade trajectory from their **multi-season form momentum**, "
        "an **age-based development curve**, and an optional **club/league environment boost**."
    )
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
**Data basis:** Grades are Europe-wide percentile rankings. The selected **role** sets which
KPIs and weights are graded (so a Defensive Midfielder and a Ball-Winning Midfielder are judged
on different metrics), while the **peer pool is the player's position** — role-curated metrics
give the role lens without the small-sample noise of filtering peers down to exact role-mates.
**Grade basis** can be **Total** or **Per 90** (per-90 judges per-minute output vs ≥900-minute
starters — better for prospects on limited minutes). The model tracks each player across seasons
(2020-21 → present) by stable Opta id to read their **actual form trajectory**, not a snapshot.

**Form momentum (new):** The season-over-season slope of the player's overall percentile is
blended into the forecast — a player genuinely improving gets a higher ceiling; a declining
one is tempered. Momentum is confidence-weighted (more reliable seasons = more weight) and
regresses toward the age curve over the horizon. Seasons under ~600 minutes are ignored.

**Age curve:** Each age bracket carries a growth rate (e.g. a 19-year-old gains ~+7 percentile
points per year in ideal conditions; a 31-year-old loses ~3/yr). It anchors the long-term
trend once near-term momentum fades.

**Club/League environment:** Joining an elite club (e.g. Belgian league → Manchester United)
triggers a ceiling boost because better coaching, training facilities and teammates accelerate
development — especially for young players.  A year-1 adaptation dip is applied before the
player fully benefits in year 2-3.

**Uncertainty band:** The shaded area on the chart widens over time — projections become
less certain the further out you look.
            """.strip()
        )

    st.markdown("---")
    st.markdown("### 1️⃣ Select Player")

    c1, c2 = st.columns(2)
    with c1:
        league_sel = st.selectbox(
            "League", ["All"] + sorted(df_total["league_display"].unique()), key="pot_lg"
        )
    with c2:
        pool = df_total if league_sel == "All" else df_total[df_total["league_display"] == league_sel]
        _opts = _player_options(pool)
        _labels = [o[0] for o in _opts]
        _sel_label = st.selectbox("Player", _labels, index=None,
                                  placeholder="Select a player…", key="pot_pl")

    if not _sel_label:
        return

    player_sel, _sel_team = _opts[_labels.index(_sel_label)][1:]
    _prow = pool[(pool["nombre"] == player_sel) & (pool["equipo"] == _sel_team)]
    if _prow.empty:
        _prow = pool[pool["nombre"] == player_sel]
    row = _prow.iloc[0]
    orig_position = row.get("posicion", "Unknown")
    league = row.get("league_display", "")
    team = row.get("equipo", "Unknown")

    # Position / Role override (same pattern as Player Profile)
    _all_positions = sorted(POSITION_ROLE_PROFILES.keys())
    _orig_idx = _all_positions.index(orig_position) if orig_position in _all_positions else 0
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        position = st.selectbox(
            "Position", _all_positions, index=_orig_idx, key="pot_pos",
            help="Override if Opta's position label doesn't reflect the player's actual role.",
        )
    role = _classify_role(row, position, df_total)
    _avail_roles = list(POSITION_ROLE_PROFILES.get(position, {}).keys())
    with ov2:
        if _avail_roles:
            _role_idx = _avail_roles.index(role) if role in _avail_roles else 0
            role = st.selectbox("Role", _avail_roles, index=_role_idx, key="pot_role")
        else:
            st.selectbox("Role", [role], key="pot_role")
    with ov3:
        pot_stat_mode = st.radio(
            "Grade basis", ["Total", "Per 90"], horizontal=True, key="pot_stat_mode",
            help="Per 90 grades per-minute output (better for prospects on limited "
                 "minutes); it ranks the player against ≥900-minute starters.",
        )
    if st.session_state.get("padj_on", False):
        st.caption("⚖️ **Possession-adjusted** grading is on (sidebar toggle) — defensive "
                   "& volume metrics are normalised to 50% possession, lifting defenders "
                   "on lower-possession sides. Applied to the grade *and* the trajectory.")

    # ── Compute current Europe-wide overall grade ────────────────────────
    # Respect the sidebar possession-adjust toggle, and (Per-90) grade the
    # player's per-90 row against ≥900-minute starters (+ self).
    _padj_on = st.session_state.get("padj_on", False)
    if pot_stat_mode == "Per 90":
        _frame = data["padj_per90" if _padj_on else "per90"]
        _is_self = (_frame["nombre"] == player_sel) & (_frame["equipo"] == _sel_team)
        grade_pool = _frame[(_frame["estimated_90s"].fillna(0) >= 10) | _is_self]
        _grow = _frame[_is_self]
        grade_row = dict(_grow.iloc[0]) if not _grow.empty else dict(row)
    else:
        grade_pool = data["padj"] if _padj_on else df_total
        _grow = grade_pool[(grade_pool["nombre"] == player_sel) & (grade_pool["equipo"] == _sel_team)]
        grade_row = dict(_grow.iloc[0]) if not _grow.empty else dict(row)
    attr_grades_ov = _compute_attribute_grades(
        grade_row, position, grade_pool, league=None, kpi_role=role
    )
    _kpi = _ROLE_KPI_PROFILES.get(role)
    _ov_weights = (
        {n: w for n, (w, _) in _kpi.items()} if _kpi
        else _POSITION_GRADE_WEIGHTS.get(position, {})
    )
    _ov_pcts = {k: pct for k, (_, pct) in attr_grades_ov.items() if pct is not None}
    if _ov_pcts and _ov_weights:
        _w_sum = sum(_ov_weights.get(k, 0) for k in _ov_pcts)
        _base_pct = (
            sum(_ov_pcts[k] * _ov_weights.get(k, 0) for k in _ov_pcts) / _w_sum
            if _w_sum > 0 else sum(_ov_pcts.values()) / len(_ov_pcts)
        )
    else:
        _base_pct = sum(_ov_pcts.values()) / len(_ov_pcts) if _ov_pcts else 0.0
    # Apply the same uplift the Player Profile headline grade uses (Key-Strengths
    # + exceptional contribution, capped +8) so the two never disagree.
    _pos_peers_pot = grade_pool[grade_pool["posicion"] == position]
    _ks_pot = _compute_key_strength_bonus(grade_row, _pos_peers_pot, position == "Goalkeeper")
    _exc_pot = _compute_exceptional_contribution(grade_row, position, role, grade_pool)[2]
    current_pct = round(min(99.9, _base_pct + min(8.0, _ks_pot + _exc_pot)), 1)
    current_grade = _percentile_to_grade(current_pct)

    # ── Multi-season trajectory & form momentum (tracked by stable id) ────
    _pid = row.get("id")
    _traj = (_build_trajectory(_pid, stat_mode=pot_stat_mode, padj=_padj_on, _bust=_data_fingerprint())
             if (_pid is not None and not pd.isna(_pid)) else [])
    # Anchor the current season's point to the displayed grade (user role/pos).
    for _t in _traj:
        if _t["season"] == CURRENT_SEASON:
            _t["pct"], _t["reliable"] = current_pct, True
    _reliable_traj = [t for t in _traj if t["reliable"] and t["pct"] is not None]
    _mom, _mom_conf, _n_seasons = _trajectory_momentum([t["pct"] for t in _reliable_traj])

    st.markdown("---")
    st.markdown("### 2️⃣ Age & Environment")

    # Try to auto-detect age from dataset columns
    _detected_age = None
    for _age_col in ("edad", "age", "Age"):
        if _age_col in row.index:
            try:
                _v = int(row[_age_col])
                if 15 <= _v <= 45:
                    _detected_age = _v
                    break
            except (ValueError, TypeError):
                pass

    age_c1, age_c2 = st.columns(2)
    with age_c1:
        _age_hint = f" (auto-detected: {_detected_age})" if _detected_age else " (enter manually)"
        age = st.number_input(
            f"Player Age{_age_hint}",
            min_value=15, max_value=45,
            value=_detected_age if _detected_age else 23,
            step=1, key="pot_age",
            help=(
                "Age is the biggest driver of the growth curve.  "
                "Players ≤21 have the highest ceiling; players ≥30 typically decline."
            ),
        )
    with age_c2:
        years_ahead = st.slider(
            "Project ahead (years)", min_value=1, max_value=5, value=3, key="pot_years"
        )

    club_move = st.selectbox(
        "Recent Club / League Upgrade",
        list(_CLUB_TIERS.keys()), index=0, key="pot_club",
        help=(
            "Did this player recently join a significantly better club or league?  "
            "Elite environments accelerate development, especially for young players.  "
            "A year-1 adaptation dip is modelled before the boost kicks in."
        ),
    )
    tier_data = _CLUB_TIERS[club_move]
    if tier_data["ceiling_boost"] != 0:
        _boost_sign = "+" if tier_data["ceiling_boost"] > 0 else ""
        _dip_txt = f", Yr1 adaptation dip −{tier_data['year1_dip']}" if tier_data["year1_dip"] else ""
        st.caption(
            f"**Effect:** {tier_data['desc']}  "
            f"*(Ceiling {_boost_sign}{tier_data['ceiling_boost']} pctl pts{_dip_txt})*"
        )

    # ── Run projection (age curve blended with multi-season momentum) ─────
    projections = _project_potential(current_pct, int(age), club_move, years=int(years_ahead),
                                     momentum=_mom, momentum_conf=_mom_conf)
    _curr_phase = _get_age_growth(int(age))[1]

    st.markdown("---")
    st.markdown("### 3️⃣ Grade Projection")

    # ── Form-trend banner (multi-season momentum) ────────────────────────
    if _n_seasons < 2:
        _trend_lbl, _trend_color = "Insufficient history — age curve only", "#888"
    elif _mom >= 2:
        _trend_lbl, _trend_color = f"Rising (+{_mom:.1f} pctl/yr)", "#00e676"
    elif _mom <= -2:
        _trend_lbl, _trend_color = f"Declining ({_mom:.1f} pctl/yr)", "#e63946"
    else:
        _trend_lbl, _trend_color = f"Stable ({_mom:+.1f} pctl/yr)", "#ffd740"
    _conf_lbl = {0.5: "low", 0.75: "medium", 0.9: "high"}.get(_mom_conf, "—")
    _hist_txt = " → ".join(f"{t['season'][2:4]}-{t['season'][7:9]}: {t['pct']:.0f}"
                           for t in _reliable_traj) or "—"
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04);border-left:4px solid {_trend_color};"
        f"border-radius:6px;padding:8px 14px;margin-bottom:6px;font-size:13px;color:#eee;'>"
        f"📈 <strong>Form trend:</strong> <span style='color:{_trend_color};font-weight:bold;'>"
        f"{_trend_lbl}</span> over {_n_seasons} season{'s' if _n_seasons != 1 else ''} "
        f"({_conf_lbl} confidence)"
        f"<br><span style='font-size:11px;color:#999;'>Overall %ile by season: {_hist_txt}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Header cards: Current + Projected ───────────────────────────────
    _curr_color = _GRADE_COLORS.get(current_grade, "#888")
    _phase_color = _POTENTIAL_PHASE_COLORS.get(_curr_phase, "#aaa")

    header_cols = st.columns([1, 1] + [1] * len(projections[1:]))
    with header_cols[0]:
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:12px;padding:16px 20px;"
            f"text-align:center;'>"
            f"<div style='font-size:11px;color:#aaa;'>Current Grade</div>"
            f"<div style='font-size:60px;font-weight:bold;color:{_curr_color};line-height:1.05;'>"
            f"{current_grade}</div>"
            f"<div style='font-size:12px;color:#777;'>{_ordinal(current_pct)} pctl · {league}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:12px;padding:16px 20px;"
            f"text-align:center;height:100%;'>"
            f"<div style='font-size:11px;color:#aaa;'>Career Phase</div>"
            f"<div style='font-size:26px;line-height:1.3;margin:10px 0;color:{_phase_color};'>"
            f"{_curr_phase}</div>"
            f"<div style='font-size:12px;color:#777;'>Age {int(age)} · {role}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    for col_ui, proj in zip(header_cols[2:], projections[1:]):
        _pg_color = _GRADE_COLORS.get(proj["grade"], "#888")
        _delta = proj["pct"] - current_pct
        _arrow = "↑" if _delta > 0.5 else ("↓" if _delta < -0.5 else "→")
        _arrow_color = "#00e676" if _delta > 0.5 else ("#e63946" if _delta < -0.5 else "#aaa")
        with col_ui:
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:12px;padding:16px 20px;"
                f"text-align:center;'>"
                f"<div style='font-size:11px;color:#aaa;'>{proj['year']}</div>"
                f"<div style='font-size:52px;font-weight:bold;color:{_pg_color};line-height:1.05;'>"
                f"{proj['grade']}</div>"
                f"<div style='font-size:12px;color:{_arrow_color};'>"
                f"{_arrow} {_ordinal(proj['pct'])} pctl</div>"
                f"<div style='font-size:10px;color:#555;margin-top:4px;'>{proj['notes']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Trajectory chart ─────────────────────────────────────────────────
    st.markdown("#### 📈 Grade Trajectory")

    # Actual past seasons (reliable, excluding the current one) → real history.
    _past = [t for t in _reliable_traj if t["season"] != CURRENT_SEASON]
    _past_labels = [f"{t['season'][2:4]}-{t['season'][7:9]}" for t in _past]
    _past_y = [t["pct"] for t in _past]

    proj_labels = [p["year"] for p in projections]      # ["Now","+1 yr",...]
    proj_y = [p["pct"] for p in projections]
    proj_grades = [p["grade"] for p in projections]

    # Actual segment = past seasons + "Now"; projected segment starts at "Now".
    actual_x = _past_labels + ["Now"]
    actual_y = _past_y + [current_pct]
    fut_x = proj_labels                                 # "Now" + future
    fut_y = proj_y

    fig_proj = go.Figure()

    # Uncertainty band over the projected (future) portion only — widens by year.
    _bw = [3 + i * 5 for i in range(len(fut_x))]
    _upper = [min(99, v + b) for v, b in zip(fut_y, _bw)]
    _lower = [max(0, v - b) for v, b in zip(fut_y, _bw)]
    fig_proj.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1], y=_upper + _lower[::-1],
        fill="toself", fillcolor="rgba(45,106,79,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Uncertainty range",
        showlegend=True, hoverinfo="skip",
    ))

    # Actual history (solid line, filled markers)
    if _past_labels:
        fig_proj.add_trace(go.Scatter(
            x=actual_x, y=actual_y, mode="lines+markers+text",
            marker=dict(size=13, color=[_GRADE_COLORS.get(_percentile_to_grade(v), "#888") for v in actual_y],
                        line=dict(width=2, color="#1a1a2e")),
            line=dict(color="#9aa7b8", width=2.5),
            text=[_percentile_to_grade(v) for v in actual_y], textposition="top center",
            textfont=dict(size=13, color="#ccd", family="Arial Black"),
            name="Actual (past seasons)",
            customdata=[[_percentile_to_grade(v), _ordinal(v)] for v in actual_y],
            hovertemplate="<b>%{x}</b><br>Grade: %{customdata[0]}<br>Percentile: %{customdata[1]}<extra>Actual</extra>",
        ))

    # Projection (dashed line, open markers)
    fig_proj.add_trace(go.Scatter(
        x=fut_x, y=fut_y, mode="lines+markers+text",
        marker=dict(size=14, symbol="circle-open",
                    color=[_GRADE_COLORS.get(g, "#888") for g in proj_grades],
                    line=dict(width=3, color="#52b788")),
        line=dict(color="#52b788", width=2.5, dash="dash"),
        text=proj_grades, textposition="bottom center",
        textfont=dict(size=14, color="#f4a261", family="Arial Black"),
        name="Projected",
        customdata=[[p["grade"], _ordinal(p["pct"]), p["notes"]] for p in projections],
        hovertemplate=(
            "<b>%{x}</b><br>Grade: %{customdata[0]}<br>Percentile: %{customdata[1]}<br>"
            "%{customdata[2]}<extra>Projected</extra>"
        ),
    ))

    # Subtle grade-tier shading
    for lo, hi, clr in [
        (97, 100, "#00c853"), (90, 97, "#2979ff"), (80, 90, "#aa00ff"),
        (70, 80, "#ff9100"), (55, 70, "#ff3d00"), (0, 55, "#d50000"),
    ]:
        fig_proj.add_hrect(y0=lo, y1=hi, fillcolor=clr, opacity=0.03, line_width=0)

    for threshold, lbl in [(97, "S"), (90, "A"), (80, "B"), (70, "C"), (55, "D")]:
        fig_proj.add_hline(
            y=threshold, line_dash="dot", line_color="rgba(255,255,255,0.15)",
            annotation_text=lbl, annotation_position="right",
            annotation_font_color="rgba(255,255,255,0.4)", annotation_font_size=10,
        )

    fig_proj.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        yaxis=dict(
            title="Overall Percentile (Europe-wide vs position peers)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                   categoryorder="array",
                   categoryarray=_past_labels + proj_labels),  # past → Now → future
        height=440,
        legend=dict(font=dict(color="#eee")),
        margin=dict(t=40, b=40, l=60, r=80),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ── Ceiling card + factor breakdown ──────────────────────────────────
    st.markdown("---")
    ceiling_col, factor_col = st.columns([1, 2])

    _ceiling_proj = projections[-1]
    _ceiling_color = _GRADE_COLORS.get(_ceiling_proj["grade"], "#888")

    with ceiling_col:
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:12px;padding:24px;"
            f"text-align:center;'>"
            f"<div style='font-size:13px;color:#aaa;margin-bottom:8px;'>"
            f"Projected Ceiling<br>"
            f"<span style='font-size:10px;'>in {int(years_ahead)} "
            f"year{'s' if int(years_ahead) > 1 else ''}</span></div>"
            f"<div style='font-size:76px;font-weight:bold;color:{_ceiling_color};"
            f"line-height:1;'>{_ceiling_proj['grade']}</div>"
            f"<div style='font-size:13px;color:#777;margin-top:8px;'>"
            f"{_ordinal(_ceiling_proj['pct'])} percentile</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with factor_col:
        st.markdown("#### 🔍 Projection Factors")
        _curr_growth, _ = _get_age_growth(int(age))
        _total_age = sum(_get_age_growth(int(age) + y)[0] for y in range(1, int(years_ahead) + 1))

        factors = [
            {
                "Factor": "Current Grade",
                "Value": f"{current_grade} ({_ordinal(current_pct)} pctl)",
                "Impact": "Baseline",
            },
            {
                "Factor": "Career Phase",
                "Value": _curr_phase,
                "Impact": (
                    f"{'+' if _curr_growth >= 0 else ''}{_curr_growth:.1f} pctl pts / yr"
                ),
            },
            {
                "Factor": "Form Trend (multi-season)",
                "Value": (f"{_n_seasons} reliable seasons · {_conf_lbl} conf"
                          if _n_seasons >= 2 else "Insufficient history"),
                "Impact": (f"{_mom:+.1f} pctl pts / yr" if _n_seasons >= 2 else "—"),
            },
        ]
        if tier_data["ceiling_boost"] != 0:
            _dip_part = f", Yr1 dip -{tier_data['year1_dip']}" if tier_data["year1_dip"] else ""
            factors.append({
                "Factor": "Club / League Upgrade",
                "Value": club_move.split("(")[0].strip(),
                "Impact": f"+{tier_data['ceiling_boost']} ceiling{_dip_part}",
            })
        factors.append({
            "Factor": f"Total Age Growth ({int(years_ahead)} yr)",
            "Value": f"Age {int(age)} → {int(age) + int(years_ahead)}",
            "Impact": f"{'+' if _total_age >= 0 else ''}{_total_age:.1f} pctl pts cumulative",
        })
        st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)

    # ── Per-attribute projections ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔬 Attribute-level Projections")
    st.caption(
        "Each attribute grade is projected by applying the same overall percentile delta to each "
        "individual attribute — showing where the player could end up across key skill categories."
    )

    if attr_grades_ov:
        _delta_total = _ceiling_proj["pct"] - current_pct
        proj_table = []
        for attr, (grade, pct) in attr_grades_ov.items():
            if pct is None:
                continue
            proj_pct = max(0, min(99, round(pct + _delta_total, 1)))
            proj_grade = _percentile_to_grade(proj_pct)
            _chg = proj_pct - pct
            proj_table.append({
                "Attribute": attr,
                "Current Grade": grade,
                "Current %ile": f"{pct:.0f}",
                f"Grade (+{int(years_ahead)}yr)": proj_grade,
                f"%ile (+{int(years_ahead)}yr)": f"{proj_pct:.0f}",
                "Change": f"{'+' if _chg >= 0 else ''}{_chg:.0f}",
            })
        if proj_table:
            st.dataframe(pd.DataFrame(proj_table), use_container_width=True, hide_index=True)

    # ── Disclaimer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **Statistical estimate only.** These projections are based on typical age-development "
        "curves in elite football and are NOT guarantees of future performance.  "
        "Injuries, coaching changes, form, and many other factors can materially alter a "
        "player's trajectory.  Use as a scouting aid alongside other qualitative evidence."
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🕵️ FORENSICS XG: INTELLIGENCE UNIT")
    st.markdown("#### WHERE THE BEAUTIFUL GAME MEETS HARD EVIDENCE.")

    # ── Season selector ──────────────────────────────────────────────────
    seasons = get_available_seasons()
    season = st.sidebar.selectbox(
        "📅 Season", seasons, index=0,
        help="Pick a season. Past seasons exclude market values, salaries and "
             "potential grading (current-season only).",
    )
    is_current = season == CURRENT_SEASON
    st.session_state["sel_season"] = season

    # ── Possession-adjust toggle (raw by default) ────────────────────────
    padj_on = st.sidebar.toggle(
        "⚖️ Possession-adjust stats", value=False, key="padj_on",
        help="Scale stats for how much possession a team has — useful for fairly "
             "comparing defenders across dominant vs. low-possession teams. Off = raw numbers.",
    )

    _season_tag = "current · Opta" if is_current else "historical"
    _padj_tag = " · ⚖️ possession-adjusted" if padj_on else ""
    st.caption(f"📅 Season: **{season}** ({_season_tag}){_padj_tag}")
    if not is_current:
        st.sidebar.caption(
            "ℹ️ Historical season — market values, salaries and potential "
            "grading are available for the current season only."
        )

    # Load data
    data = load_data(season)
    df_total = data["total"]

    if df_total.empty:
        # Diagnostic info for debugging cloud deployment
        _diag_lines = [f"OPTA_DIR = {OPTA_DIR}", f"season = {season}"]
        for _dn, _fn in LEAGUE_FOLDERS.items():
            _fp = os.path.join(OPTA_DIR, _fn)
            _exists = os.path.isdir(_fp)
            _src = "jugadores_seasonstats.csv" if is_current else "jugadores_historical.csv"
            _csv_exists = os.path.exists(os.path.join(_fp, _src))
            _diag_lines.append(f"  {_dn}: dir={_exists}, {_src}={_csv_exists}")
        st.error("No data found for this season.\n\n" + "\n".join(_diag_lines))
        return

    # Tabs
    tab_analysis, tab_profile, tab_gk, tab_potential, tab_compare, tab_team, tab_team_cmp, tab_explorer = st.tabs([
        "🔬 Player Lab", "🪪 Player Profile", "🧤 GK Analysis", "🌟 Potential Grading",
        "⚔️ Player Comparison", "🏟️ Team Profile", "🏟️ Team Comparison", "🔍 Data Explorer",
    ])

    with tab_analysis:
        render_player_lab(data, is_current=is_current)
    with tab_profile:
        render_profile(data, is_current=is_current)
    with tab_gk:
        render_gk_analysis(data)
    with tab_potential:
        render_potential_grading(data, is_current=is_current)
    with tab_compare:
        render_player_comparison(data)
    with tab_team:
        render_team_profile(data)
    with tab_team_cmp:
        render_team_comparison(data)
    with tab_explorer:
        render_explorer(data)


if __name__ == "__main__":
    main()

