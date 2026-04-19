"""
FORENSICS XG: INTELLIGENCE UNIT
Football Analytics — Interactive data analysis tool.
Ask questions and get charts, stats, and insights from Europe's top 6 leagues.
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
}

# Squad roster overrides are no longer needed — Opta data includes team info directly.
SQUAD_ROSTER_OVERRIDES = {}

CHART_COLORS = px.colors.qualitative.Vivid

# ── Player Financials CSV ────────────────────────────────────────────────────
_FINANCIALS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Player Financials", "player_financials.csv")


@st.cache_data(ttl=86400, show_spinner=False)
def _load_financials_csv():
    """Load pre-fetched market values and salaries from CSV (if it exists)."""
    if not os.path.exists(_FINANCIALS_CSV):
        return {}
    df = pd.read_csv(_FINANCIALS_CSV, encoding="utf-8-sig")
    lookup = {}
    for _, r in df.iterrows():
        key = str(r.get("short_name", "")).strip()
        if key:
            mv = r.get("market_value", "")
            sal = r.get("salary", "")
            lookup[key] = {
                "market_value": mv if pd.notna(mv) and mv != "" else None,
                "salary": sal if pd.notna(sal) and sal != "" else None,
            }
    return lookup


META_COLS = {"nombre", "posicion", "posicion_detail", "league_display", "Player",
             "equipo", "Appearances", "Time Played", "estimated_90s"}

# ── Metric groupings (Opta column names) ─────────────────────────────────────

OFFENSIVE_METRICS = [
    "Goals", "Goals Openplay", "Total Shots", "Shots On Target ( inc goals )",
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
    "Key Passes (Attempt Assists)", "Goal Assists",
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
]

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


def _make_match_key(name):
    """First-initial + last-name key for matching: 'A. Isak' -> 'a.isak'."""
    parts = name.split()
    if not parts:
        return ""
    return parts[0][0] + "." + parts[-1] if len(parts) > 1 else parts[0]


def _build_team_lookup():
    """No longer needed — Opta data includes equipo directly."""
    return pd.DataFrame()


def _load_league_opta(folder_path):
    """Load a single jugadores_seasonstats.csv from an Opta league folder."""
    csv_path = os.path.join(folder_path, "jugadores_seasonstats.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    return df


@st.cache_data(show_spinner="Loading football data...")
def load_data():
    """Load all Opta data.  Returns dict with 'total' and 'per90' DataFrames.
    
    Opta provides only season totals.  Per-90 values are computed from
    Time Played: stat_per90 = stat_total / (Time Played / 90).
    """
    _CACHE_VERSION = 15  # position/role override in Player Profile

    total_frames = []

    for display_name, folder_name in LEAGUE_FOLDERS.items():
        folder_path = os.path.join(OPTA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        df = _load_league_opta(folder_path)
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

    # Assign all derived columns at once (avoids DataFrame fragmentation)
    if _derived:
        combined = pd.concat([combined, pd.DataFrame(_derived, index=combined.index)], axis=1)

    # ── Sub-classify defenders into CB vs FB ───────────────────────
    # Opta only provides "Defender" — use wide-attacking vs central-defensive
    # balance to split into Centre-Back vs Full-Back.
    _DEF_WIDE = ["Successful Crosses & Corners", "Successful Crosses open play",
                 "Successful Dribbles", "Key Passes (Attempt Assists)",
                 "Total Touches In Opposition Box"]
    _DEF_CENTRAL = ["Aerial Duels won", "Total Clearances", "Blocked Shots",
                    "Blocks", "Headed Goals", "Interceptions"]
    def_mask = combined["posicion"] == "Centre-Back"
    if def_mask.sum() > 10:
        wide_avail = [m for m in _DEF_WIDE if m in combined.columns]
        central_avail = [m for m in _DEF_CENTRAL if m in combined.columns]
        if wide_avail and central_avail:
            def_idx = combined.index[def_mask]
            wide_pct = combined.loc[def_idx, wide_avail].rank(pct=True).mean(axis=1)
            central_pct = combined.loc[def_idx, central_avail].rank(pct=True).mean(axis=1)
            balance = wide_pct - central_pct
            fb_idx = balance[balance > 0].index
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

    # ── Drop players with zero minutes ──────────────────────────────
    combined = combined[combined["Time Played"].fillna(0) > 0].reset_index(drop=True)

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
                                        "Retention %", "Own Half Pass %"}:
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
                    poss_pct = max(30.0, min(70.0, poss_pct))
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
    else:
        result["padj"] = combined.copy()
        result["padj_per90"] = per90.copy()

    return result


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

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_player_photo(player_name, team=None):
    """Try to fetch a player photo URL from TheSportsDB."""
    try:
        q = urllib.parse.quote(player_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        players = data.get("player")
        if players:
            def _photo(p):
                return p.get("strCutout") or p.get("strThumb") or p.get("strRender")
            # If team hint provided, prefer a player on that team
            if team:
                team_lower = team.lower().replace(" fc", "").replace("fc ", "").strip()
                for p in players:
                    p_team = (p.get("strTeam") or "").lower().replace(" fc", "").replace("fc ", "").strip()
                    if team_lower in p_team or p_team in team_lower:
                        photo = _photo(p)
                        if photo:
                            return photo
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
        ("Through Balls", "Through balls"),
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
        ("Assists", "Goal Assists"),
        ("Shots on Target", "Shots On Target ( inc goals )"),
        ("Big Chances", "Total Big Chances Scored"),
    ],
}

GK_PIZZA_METRICS = {
    "Shot-Stopping": [
        ("Saves", "Saves Made"),
        ("Save %", "Save %"),
        ("Goals Prevented", "Goals Prevented"),
        ("Big Chances Saved", "Total Big Chances Saved"),
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
    "Shot-Stopping": ["Saves Made", "Save %", "Goals Prevented",
                      "Total Big Chances Saved"],
    "Command": ["Catches", "Punches", "Penalties Saved"],
    "Distribution": ["GK Successful Distribution", "Successful Launches",
                     "Launch %"],
    "Sweeping": ["Recoveries", "Total Clearances", "Interceptions"],
}

STRIKER_PROFILE_CATEGORIES = {
    "Finishing": ["Goals", "Non-Penalty Goals", "Total Shots",
                  "Shots On Target ( inc goals )", "Total Touches In Opposition Box",
                  "Total Big Chances Scored"],
    "Chance Creation": ["Goal Assists", "Key Passes (Attempt Assists)",
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
    "Chance Creation": ["Goal Assists", "Key Passes (Attempt Assists)",
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
    "Chance Creation": ["Goal Assists", "Key Passes (Attempt Assists)",
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
    "Attacking": ["Goals", "Non-Penalty Goals", "Total Shots",
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
                "Successful Long Passes", "Long Pass %",
                "Successful Crosses & Corners", "Cross %",
                "Forward Passes", "Short Pass %"],
    "Dribbling & Carrying": ["Successful Dribbles", "Dribble %",
                              "Unsuccessful Dribbles",
                              "Progressive Carries", "Carries", "Overruns"],
    "Ball Progression": ["Progressive Carries", "Carries",
                         "Through balls", "Final Third Touches",
                         "Forward Passes"],
    "Passing Safety": ["Retention %", "Pass %", "Short Pass %",
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
    "Inside Forward": {
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
        "Attacking": 0.15, "Defending": 0.03, "Passing": 0.30,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.10,
    },
    "Shadow Striker": {
        "Attacking": 0.35, "Defending": 0.05, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.05,
    },
    "Creative Playmaker": {
        "Attacking": 0.10, "Defending": 0.03, "Passing": 0.35,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.10,
    },
    "Pressing 10": {
        "Attacking": 0.20, "Defending": 0.15, "Passing": 0.20,
        "Dribbling & Carrying": 0.10, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
    # --- Central Midfield roles ---
    "Anchor Man": {
        "Attacking": 0.03, "Defending": 0.35, "Passing": 0.20,
        "Dribbling & Carrying": 0.02, "Ball Progression": 0.05,
        "Passing Safety": 0.25,
    },
    "Box-to-Box": {
        "Attacking": 0.15, "Defending": 0.20, "Passing": 0.15,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
    "Deep-Lying Playmaker": {
        "Attacking": 0.05, "Defending": 0.10, "Passing": 0.35,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.20,
        "Passing Safety": 0.20,
    },
    "Ball-Winning CM": {
        "Attacking": 0.05, "Defending": 0.35, "Passing": 0.15,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.05,
        "Passing Safety": 0.20,
    },
    "Mezzala": {
        "Attacking": 0.20, "Defending": 0.05, "Passing": 0.15,
        "Dribbling & Carrying": 0.25, "Ball Progression": 0.20,
        "Passing Safety": 0.05,
    },
    # --- Centre-Back roles ---
    "Defensive Rock": {
        "Attacking": 0.02, "Defending": 0.50, "Passing": 0.10,
        "Dribbling & Carrying": 0.03, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    "Ball-Playing CB": {
        "Attacking": 0.05, "Defending": 0.25, "Passing": 0.25,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.15,
        "Passing Safety": 0.20,
    },
    "Aerial Defender": {
        "Attacking": 0.03, "Defending": 0.50, "Passing": 0.10,
        "Dribbling & Carrying": 0.02, "Ball Progression": 0.05,
        "Passing Safety": 0.10,
    },
    "Ball-Winning CB": {
        "Attacking": 0.03, "Defending": 0.45, "Passing": 0.10,
        "Dribbling & Carrying": 0.02, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    # --- Full-Back roles ---
    "Attacking Full-Back": {
        "Attacking": 0.15, "Defending": 0.15, "Passing": 0.20,
        "Dribbling & Carrying": 0.20, "Ball Progression": 0.20,
        "Passing Safety": 0.05,
    },
    "Defensive Full-Back": {
        "Attacking": 0.05, "Defending": 0.40, "Passing": 0.15,
        "Dribbling & Carrying": 0.05, "Ball Progression": 0.05,
        "Passing Safety": 0.15,
    },
    "Creative Full-Back": {
        "Attacking": 0.10, "Defending": 0.15, "Passing": 0.30,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.15,
        "Passing Safety": 0.10,
    },
    "Inverted Full-Back": {
        "Attacking": 0.10, "Defending": 0.15, "Passing": 0.25,
        "Dribbling & Carrying": 0.15, "Ball Progression": 0.20,
        "Passing Safety": 0.10,
    },
    # --- Goalkeeper roles ---
    "Shot-Stopper": {
        "Shot-Stopping": 0.50, "Command": 0.20,
        "Distribution": 0.10, "Sweeping": 0.10,
    },
    "Sweeper Keeper": {
        "Shot-Stopping": 0.25, "Command": 0.15,
        "Distribution": 0.20, "Sweeping": 0.35,
    },
    "Ball-Playing Goalkeeper": {
        "Shot-Stopping": 0.25, "Command": 0.15,
        "Distribution": 0.40, "Sweeping": 0.15,
    },
}

# ── Role-specific KPI profiles (curated key metrics per role) ────────────────
# Each role maps category_name → (weight, [key_metrics]).
# Overall grade = weighted avg of KPI category percentiles.
# Ball Security now uses rate stats (higher = better) — NOT inverted.
_KPI_INVERTED_CATS = set()

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
    "Inside Forward": {
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
        "Creativity":      (0.30, ["Total Big Chances Created", "Key Passes (Attempt Assists)", "Goal Assists", "Total Touches In Opposition Box"]),
        "Passing Quality": (0.25, ["Pass %", "Forward Passes", "Through balls"]),
        "Progression":     (0.20, ["Progressive Carries", "Final Third Touches"]),
        "Ball Security":   (0.15, ["Retention %", "Pass %", "Dribble %"]),
        "Finishing":       (0.10, ["Non-Penalty Goals", "Shots On Target ( inc goals )"]),
    },
    "Shadow Striker": {
        "Finishing":     (0.35, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )", "Total Big Chances Scored"]),
        "Movement":      (0.25, ["Goals from Inside Box", "Total Touches In Opposition Box"]),
        "Dribbling":     (0.20, ["Successful Dribbles", "Dribble %", "Progressive Carries"]),
        "Creativity":    (0.15, ["Key Passes (Attempt Assists)", "Goal Assists"]),
        "Ball Security": (0.05, ["Retention %", "Dribble %"]),
    },
    "Creative Playmaker": {
        "Creativity":      (0.30, ["Key Passes (Attempt Assists)", "Goal Assists", "Through balls", "Total Big Chances Created"]),
        "Passing Quality": (0.25, ["Pass %", "Total Passes", "Forward Passes"]),
        "Progression":     (0.20, ["Progressive Carries", "Final Third Touches"]),
        "Ball Security":   (0.15, ["Retention %", "Pass %", "Dribble %"]),
        "Finishing":       (0.10, ["Goals", "Shots On Target ( inc goals )"]),
    },
    "Pressing 10": {
        "Pressing":      (0.25, ["Recoveries", "Tackles Won", "Interceptions"]),
        "Creativity":    (0.25, ["Key Passes (Attempt Assists)", "Goal Assists", "Through balls"]),
        "Finishing":     (0.25, ["Goals", "Non-Penalty Goals", "Shots On Target ( inc goals )"]),
        "Duels":         (0.15, ["Ground Duels won", "Ground Duel %", "Duels won"]),
        "Ball Security": (0.10, ["Retention %", "Pass %"]),
    },
    # --- Central Midfield roles ---
    "Anchor Man": {
        "Defensive Shield": (0.30, ["Interceptions", "Tackles Won", "Tackle Win %", "Recoveries"]),
        "Aerial":           (0.15, ["Aerial Duels won", "Aerial Win %"]),
        "Distribution":     (0.25, ["Pass %", "Total Passes", "Successful Long Passes", "Long Pass %"]),
        "Ball Security":    (0.20, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Positioning":      (0.10, ["Blocked Shots", "Blocks", "Total Clearances"]),
    },
    "Box-to-Box": {
        "Defensive Work":   (0.25, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Attacking Output": (0.20, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)"]),
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
    "Ball-Winning CM": {
        "Tackling":         (0.30, ["Tackles Won", "Tackle Win %", "Total Tackles"]),
        "Defensive Impact": (0.25, ["Interceptions", "Recoveries", "Blocked Shots"]),
        "Duels":            (0.20, ["Aerial Duels won", "Aerial Win %", "Ground Duels won", "Ground Duel %"]),
        "Ball Security":    (0.15, ["Retention %", "Pass %"]),
        "Distribution":     (0.10, ["Pass %", "Total Passes"]),
    },
    "Mezzala": {
        "Carrying":         (0.25, ["Progressive Carries", "Successful Dribbles", "Dribble %"]),
        "Attacking Output": (0.25, ["Goals", "Goal Assists", "Key Passes (Attempt Assists)"]),
        "Progression":      (0.20, ["Forward Passes", "Through balls", "Final Third Touches"]),
        "Passing Quality":  (0.20, ["Pass %", "Short Pass %"]),
        "Ball Security":    (0.10, ["Retention %", "Dribble %"]),
    },
    # --- Centre-Back roles ---
    "Defensive Rock": {
        "Tackling":      (0.30, ["Tackles Won", "Tackle Win %", "Interceptions"]),
        "Aerial":        (0.25, ["Aerial Duels won", "Aerial Win %"]),
        "Clearances":    (0.20, ["Total Clearances", "Blocked Shots", "Blocks"]),
        "Ball Security": (0.15, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Distribution":  (0.10, ["Pass %", "Successful Long Passes"]),
    },
    "Ball-Playing CB": {
        "Distribution":       (0.30, ["Pass %", "Successful Long Passes", "Long Pass %", "Forward Passes"]),
        "Progression":        (0.20, ["Progressive Carries", "Through balls"]),
        "Defensive Solidity": (0.20, ["Interceptions", "Tackles Won", "Total Clearances"]),
        "Ball Security":      (0.20, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Aerial":             (0.10, ["Aerial Duels won", "Aerial Win %"]),
    },
    "Aerial Defender": {
        "Aerial":             (0.35, ["Aerial Duels won", "Aerial Win %"]),
        "Defensive Solidity": (0.25, ["Total Clearances", "Interceptions", "Blocks"]),
        "Tackling":           (0.20, ["Tackles Won", "Tackle Win %", "Total Tackles"]),
        "Distribution":       (0.10, ["Pass %", "Successful Long Passes"]),
        "Ball Security":      (0.10, ["Retention %", "Pass %"]),
    },
    "Ball-Winning CB": {
        "Tackling":           (0.30, ["Tackles Won", "Tackle Win %", "Total Tackles"]),
        "Duels":              (0.25, ["Ground Duels won", "Ground Duel %", "Duels won"]),
        "Defensive Solidity": (0.20, ["Interceptions", "Recoveries", "Blocks"]),
        "Ball Security":      (0.15, ["Retention %", "Pass %", "Own Half Pass %"]),
        "Distribution":       (0.10, ["Pass %", "Total Passes"]),
    },
    # --- Full-Back roles ---
    "Attacking Full-Back": {
        "Attacking Output": (0.30, ["Goal Assists", "Key Passes (Attempt Assists)", "Successful Crosses & Corners", "Cross %"]),
        "Progression":      (0.25, ["Progressive Carries", "Forward Passes", "Final Third Touches"]),
        "Dribbling":        (0.20, ["Successful Dribbles", "Dribble %"]),
        "Defensive Duty":   (0.15, ["Tackles Won", "Interceptions"]),
        "Ball Security":    (0.10, ["Retention %", "Pass %"]),
    },
    "Defensive Full-Back": {
        "Defensive Solidity": (0.30, ["Tackles Won", "Tackle Win %", "Interceptions"]),
        "Aerial":             (0.15, ["Aerial Duels won", "Aerial Win %"]),
        "Duels":              (0.25, ["Ground Duels won", "Ground Duel %", "Duels won"]),
        "Distribution":       (0.15, ["Pass %", "Successful Long Passes"]),
        "Ball Security":      (0.15, ["Retention %", "Pass %"]),
    },
    "Creative Full-Back": {
        "Crossing":        (0.25, ["Successful Crosses & Corners", "Cross %"]),
        "Creativity":      (0.25, ["Key Passes (Attempt Assists)", "Goal Assists", "Through balls", "Total Big Chances Created"]),
        "Passing Quality": (0.20, ["Pass %", "Forward Passes", "Long Pass %"]),
        "Progression":     (0.15, ["Progressive Carries", "Successful Dribbles"]),
        "Ball Security":   (0.15, ["Retention %", "Pass %"]),
    },
    "Inverted Full-Back": {
        "Passing Quality": (0.30, ["Pass %", "Forward Passes", "Short Pass %", "Through balls"]),
        "Progression":     (0.25, ["Progressive Carries", "Successful Dribbles"]),
        "Distribution":    (0.20, ["Successful Long Passes", "Long Pass %"]),
        "Defensive Duty":  (0.15, ["Tackles Won", "Interceptions", "Recoveries"]),
        "Ball Security":   (0.10, ["Retention %", "Pass %"]),
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

GK_ATTRIBUTE_GRADE_CATEGORIES = {
    "Shot-Stopping": ["Saves Made", "Save %",
                      "Goals Prevented",
                      "Total Big Chances Saved",
                      "Penalties Saved"],
    "Command": ["Catches", "Punches", "Aerial Duels won",
                "Aerial Duels", "Aerial Win %"],
    "Distribution": ["GK Successful Distribution", "Successful Launches",
                     "Launch %"],
    "Sweeping": ["Recoveries", "Total Clearances", "Interceptions"],
}

# ── Percentile-based role profiles (Opta metrics) ───────────────────────────
WINGER_ROLE_PROFILES = {
    "Inside Forward": [
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
    "Anchor Man": [
        "Interceptions", "Total Clearances", "Blocked Shots", "Blocks",
        "Aerial Duels", "Aerial Duels won", "Aerial Win %",
        "Total Tackles", "Tackles Won",
        "Duels", "Duels won",
    ],
    "Box-to-Box": [
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
    "Ball-Winning CM": [
        "Recoveries",
        "Ground Duels", "Ground Duels won",
        "Total Tackles", "Tackles Won",
        "Duels", "Duels won",
    ],
    "Mezzala": [
        "Progressive Carries", "Carries",
        "Successful Dribbles",
        "Overruns",
        "Goals", "Non-Penalty Goals", "Total Shots",
        "Total Touches In Opposition Box",
    ],
}

CAM_ROLE_PROFILES = {
    "Classic 10": [
        "Key Passes (Attempt Assists)", "Goal Assists",
        "Total Big Chances Created",
        "Through balls",
        "Successful Dribbles",
        "Final Third Touches",
    ],
    "Shadow Striker": [
        "Goals", "Non-Penalty Goals", "Total Shots",
        "Shots On Target ( inc goals )",
        "Total Touches In Opposition Box",
        "Key Passes (Attempt Assists)",
        "Total Big Chances Scored",
    ],
    "Creative Playmaker": [
        "Successful Crosses & Corners", "Successful Crosses open play",
        "Key Passes (Attempt Assists)", "Goal Assists",
        "Total Big Chances Created",
        "Forward Passes", "Through balls",
    ],
    "Pressing 10": [
        "Recoveries",
        "Total Tackles", "Tackles Won",
        "Goals", "Key Passes (Attempt Assists)", "Goal Assists",
        "Ground Duels", "Ground Duels won",
    ],
}

CENTRE_BACK_ROLE_PROFILES = {
    "Defensive Rock": [
        "Total Tackles", "Tackles Won", "Interceptions",
        "Total Clearances", "Blocked Shots", "Blocks",
        "Duels", "Duels won",
        "Ground Duels", "Ground Duels won",
    ],
    "Ball-Playing CB": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Successful Long Passes",
        "Forward Passes",
        "Progressive Carries", "Carries",
    ],
    "Aerial Defender": [
        "Aerial Duels", "Aerial Duels won", "Aerial Win %",
        "Total Clearances", "Headed Goals",
        "Duels", "Duels won",
    ],
    "Ball-Winning CB": [
        "Recoveries",
        "Ground Duels", "Ground Duels won",
        "Total Tackles", "Tackles Won",
        "Interceptions",
        "Duels", "Duels won",
    ],
}

FULL_BACK_ROLE_PROFILES = {
    "Attacking Full-Back": [
        "Goal Assists", "Key Passes (Attempt Assists)",
        "Successful Crosses & Corners", "Successful Crosses open play",
        "Progressive Carries", "Carries",
        "Successful Dribbles",
        "Total Touches In Opposition Box", "Goals", "Total Shots",
        "Total Big Chances Created",
    ],
    "Defensive Full-Back": [
        "Total Tackles", "Tackles Won", "Interceptions",
        "Total Clearances", "Blocked Shots", "Blocks",
        "Duels", "Duels won",
        "Ground Duels", "Ground Duels won",
    ],
    "Creative Full-Back": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Pass %",
        "Successful Long Passes",
        "Forward Passes",
        "Through balls", "Key Passes (Attempt Assists)",
    ],
    "Inverted Full-Back": [
        "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
        "Progressive Carries", "Carries",
        "Recoveries",
        "Ground Duels", "Ground Duels won",
        "Successful Dribbles",
    ],
}

GOALKEEPER_ROLE_PROFILES = {
    "Shot-Stopper": [
        "Saves Made", "Goals Prevented", "Save %",
        "Total Big Chances Saved", "Penalties Saved",
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


def _percentile_to_grade(pct):
    """Convert an overall percentile (0-100) to a letter grade."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return "F"


def _grade_html(grade, label, pct=None):
    """Return styled HTML for a grade card."""
    color = _GRADE_COLORS.get(grade, "#888")
    pct_html = f"<div style='font-size:10px;color:#777;margin-top:2px;'>{pct:.0f}th</div>" if pct is not None else ""
    return (
        f"<div style='text-align:center;'>"
        f"<div style='font-size:11px;color:#aaa;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:36px;font-weight:bold;color:{color};'>{grade}</div>"
        f"{pct_html}"
        f"</div>"
    )


@st.cache_data(ttl=86400, show_spinner="Classifying roles…")
def _classify_position_roles(df_total, position):
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
    kpi = _ROLE_KPI_PROFILES.get(kpi_role) if kpi_role else None
    if kpi:
        cats = {name: metrics for name, (weight, metrics) in kpi.items()}
        inv_cats = _KPI_INVERTED_CATS
    else:
        cats = GK_ATTRIBUTE_GRADE_CATEGORIES if _is_gk else ATTRIBUTE_GRADE_CATEGORIES
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
        role_series = _classify_position_roles(ref, position)
        matching_names = ref.loc[
            ref.index.intersection(role_series[role_series == role].index), "nombre"
        ]
        peers = peers[peers["nombre"].isin(matching_names)]
    if league:
        peers = peers[peers["league_display"] == league]

    if len(peers) < 5:
        return {attr: ("N/A", None) for attr in cats}

    pcts = _compute_percentiles(row_data, peers, cats, inverted_cats=inv_cats)

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


def _compute_percentiles(player_row, df_peers, categories, inverted_cats=None):
    """Per-category percentile (0-100) for a player vs peers.
    Ranks each metric individually against peers and averages the per-metric
    percentiles within each category — consistent with the pizza chart display
    so that high-magnitude stats don't dominate low-magnitude ones."""
    if inverted_cats is None:
        inverted_cats = _INVERTED_GRADE_CATS
    result = {}
    for cat, metrics in categories.items():
        avail = [m for m in metrics if m in df_peers.columns]
        if not avail:
            result[cat] = 0
            continue
        metric_pcts = []
        for m in avail:
            val = player_row.get(m, 0)
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else (val or 0)
            peer_vals = df_peers[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            metric_pcts.append(pct)
        cat_pct = sum(metric_pcts) / len(metric_pcts)
        if cat in inverted_cats:
            cat_pct = 100 - cat_pct
        result[cat] = round(cat_pct, 1)
    return result


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
def _build_player_lab_table(grade_df, role_df, mode_label=""):
    """Pre-compute grades and roles for every player (vectorized, cached 24 h).

    *grade_df* is the DataFrame used for percentile grading (Total or Per 90).
    *role_df*  is always the Total DataFrame (role classification uses season totals).
    """
    financials = _load_financials_csv()
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
            pct_ranks = gdf.loc[pos_group.index, avail].rank(pct=True) * 100
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
            pct_ranks = gdf.loc[group.index, avail].rank(pct=True) * 100
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
        _ov_mpct[pos] = gdf.loc[pos_group.index, _metric_cols].rank(pct=True) * 100
    _lg_mpct = {}
    for (pos, lg), group in gdf.groupby(["posicion", "league_display"]):
        if len(group) < 5:
            continue
        _lg_mpct[(pos, lg)] = gdf.loc[group.index, _metric_cols].rank(pct=True) * 100

    # ── Classify roles (always from total data) ─────────────────────────
    role_map = {}
    for position in role_df["posicion"].unique():
        if position in POSITION_ROLE_PROFILES:
            roles = _classify_position_roles(role_df, position)
            for r_idx, r_role in roles.items():
                name = role_df.at[r_idx, "nombre"]
                role_map[name] = r_role

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

    # Financials lookup (vectorized)
    _mv_series = gdf["nombre"].map(lambda n: (financials.get(n) or {}).get("market_value") or "—")
    _sal_series = gdf["nombre"].map(lambda n: (financials.get(n) or {}).get("salary") or "—")

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

    out["Overall Grade"] = _vec_grade(out["Overall %ile"])
    out["League Grade"] = _vec_grade(out["League %ile"])

    # Clean up temp columns
    gdf.drop(columns=["_role"], inplace=True, errors="ignore")

    return out


_STAT_MODES = ["Total", "Per 90", "Padj", "Padj Per 90"]

_STAT_MODE_MAP = {
    "Total": "total",
    "Per 90": "per90",
    "Padj": "padj",
    "Padj Per 90": "padj_per90",
}


def _select_df(data, stat_mode):
    """Return the DataFrame corresponding to *stat_mode*."""
    return data[_STAT_MODE_MAP.get(stat_mode, "total")]


def render_player_lab(data):
    df_total = data["total"]
    st.subheader("🔬 Player Lab")
    st.caption("Filter players by position, role, grade, market value, and salary to find your ideal targets.")

    # Stat mode selector
    lab_stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="lab_stat_mode")
    grade_df = _select_df(data, lab_stat_mode)

    if lab_stat_mode in ("Per 90", "Padj Per 90") and not grade_df.empty:
        _MIN_90S_P90 = 5
        _has_mins = grade_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
        grade_src = grade_df[_has_mins].copy()
    else:
        grade_src = grade_df

    # Build / retrieve the grade table
    with st.spinner("Computing player grades…"):
        lab_df = _build_player_lab_table(grade_src, df_total, mode_label=lab_stat_mode)

    # ── Filters ──────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
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
        role_options = sorted(role_pool["Role"].unique())
        sel_roles = st.multiselect("Role", role_options, default=[], key="lab_roles")

    # Grade range selector
    grade_col, grade_type_col = st.columns([3, 1])
    with grade_type_col:
        _attr_grade_cols = [c for c in lab_df.columns if c.endswith(" Grade") and c not in ("Overall Grade", "League Grade")]
        grade_basis = st.selectbox("Grade type",
            ["Overall Grade", "League Grade"] + _attr_grade_cols,
            key="lab_grade_type",
            help="Individual attribute grades (e.g. 'Defending Grade') are vs same-league peers. '(Europe)' variants compare Europe-wide.")
    with grade_col:
        min_idx, max_idx = st.select_slider(
            "Grade range",
            options=list(range(len(_ALL_GRADES))),
            value=(0, len(_ALL_GRADES) - 1),
            format_func=lambda i: _ALL_GRADES[i],
            key="lab_grade_range",
        )
        min_grade_idx, max_grade_idx = min_idx, max_idx

    # Market Value & Salary sliders
    val1, val2 = st.columns(2)
    with val1:
        mv_range = st.slider("Transfermarkt Value (€M)", 0.0, 250.0, (0.0, 250.0),
                              step=1.0, key="lab_mv_range")
    with val2:
        sal_range = st.slider("Gross Annual Salary (€M)", 0.0, 60.0, (0.0, 60.0),
                               step=0.5, key="lab_sal_range")

    enable_financials = (mv_range != (0.0, 250.0)) or (sal_range != (0.0, 60.0))

    # ── Apply filters ────────────────────────────────────────────────────
    filtered = lab_df.copy()
    if sel_leagues:
        filtered = filtered[filtered["League"].isin(sel_leagues)]
    if sel_positions:
        filtered = filtered[filtered["Position"].isin(sel_positions)]
    if sel_roles:
        filtered = filtered[filtered["Role"].isin(sel_roles)]

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

    # Show league-scoped attribute grades in the table (Europe-wide variants hidden by default)
    _attr_display = [c for c in filtered.columns
                     if c.endswith(" Grade") and c not in ("Overall Grade", "League Grade")
                     and not c.endswith(" Grade (Europe)")]
    # When a specific league is selected, hide the Overall columns
    if sel_leagues:
        display_cols = ["Player", "Team", "League", "Pos", "Role",
                        "League Grade", "League %ile"] + _attr_display + [
                        "Market Value", "Salary"]
    else:
        display_cols = ["Player", "Team", "League", "Pos", "Role",
                        "Overall Grade", "Overall %ile",
                        "League Grade", "League %ile"] + _attr_display + [
                        "Market Value", "Salary"]
    show = filtered[[c for c in display_cols if c in filtered.columns]].reset_index(drop=True)
    st.dataframe(show, use_container_width=True, height=600)


# ── UI: Player Profile ──────────────────────────────────────────────────────

def render_profile(data):
    df_total = data["total"]
    df_per90 = data["per90"]
    st.subheader("🪪 Player Profile")

    # Selectors
    c1, c2 = st.columns(2)
    with c1:
        league_sel = st.selectbox("League", ["All"] + sorted(df_total["league_display"].unique()), key="prof_lg")
    with c2:
        pool = df_total if league_sel == "All" else df_total[df_total["league_display"] == league_sel]
        player_names = sorted(pool["nombre"].unique())
        player_sel = st.selectbox("Player", player_names, key="prof_pl")

    if not player_sel:
        return

    row = pool[pool["nombre"] == player_sel].iloc[0]
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
    role = _classify_role(row, position, df_total)
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
    use_per90 = stat_mode in ("Per 90", "Padj Per 90")
    _active_df = _select_df(data, stat_mode)

    if use_per90 and not _active_df.empty:
        # Filter out low-minutes players whose inflated per-90 rates
        # would skew the percentile rankings for regular starters.
        _MIN_90S_P90 = 5
        _is_sel = _active_df["nombre"] == player_sel
        _has_mins = _active_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
        p90_filtered = _active_df[_is_sel | _has_mins]
        peers_all = p90_filtered.copy()
        p90_match = p90_filtered[p90_filtered["nombre"] == player_sel]
        row_data = dict(p90_match.iloc[0]) if not p90_match.empty else dict(row)
        grade_df = p90_filtered
        grade_row = row_data
    elif stat_mode in ("Total", "Padj"):
        _src = _active_df
        peers_all = _src.copy()
        row_match = _src[_src["nombre"] == player_sel]
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

    # ── Build tooltip with sub-grade breakdown ────────────────────────
    _sub_lines = "&#10;".join(
        f"{attr}: {g} ({pct:.0f}th)" for attr, (g, pct) in attr_grades.items() if pct is not None
    )
    _ov_color = _GRADE_COLORS.get(_overall_grade, "#888")
    _stat_ctx = stat_mode if stat_mode != "Total" else "Season Totals"
    _scope_ctx = league if scope_mode == "League" else "All Europe"
    _basis_ctx = f"{role}s" if basis_mode == "Role" else f"{position}s"

    # ── Header Card: [Photo | Name + Grade / Info] ─────────────────────
    _hdr_photo, _hdr_info, _hdr_spacer = st.columns([1, 3, 2])
    with _hdr_photo:
        player_photo = _fetch_player_photo(row.get("nombre", "?"), team=row.get("equipo"))
        if player_photo:
            st.image(player_photo, width=140)
        else:
            st.markdown(f"<div style='width:140px;height:140px;border-radius:50%;background:#2d6a4f;display:flex;align-items:center;justify-content:center;font-size:48px;color:white;'>{row.get('nombre', '?')[0]}</div>", unsafe_allow_html=True)
    with _hdr_info:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:24px;'>"
            f"<div>"
            f"<div style='font-size:1.8rem;font-weight:700;'>{row.get('nombre', '?')}</div>"
            f"<div style='margin-top:2px;'><strong>{league}</strong></div>"
            f"</div>"
            f"<span style='cursor:help;display:inline-flex;flex-direction:column;align-items:center;' title='{_sub_lines}'>"
            f"<span style='font-size:12px;color:#aaa;'>Overall Grade</span>"
            f"<span style='font-size:72px;font-weight:bold;color:{_ov_color};line-height:1;'>{_overall_grade}</span>"
            f"<span style='font-size:11px;color:#777;'>{_overall_pct:.1f}th pctl</span>"
            f"</span></div>",
            unsafe_allow_html=True,
        )
        _pos_label = f"{pos_detail} ({position})" if not _position_changed else f"{position} *(was {_orig_position})*"
        st.markdown(f"**Position:** {_pos_label} · **Role:** {role}")
    st.caption(f"Grade: {_stat_ctx} · vs {_basis_ctx} in {_scope_ctx}")

    # ── Market Value & Salary ────────────────────────────────────────────
    _player_team = row.get("equipo")
    _fin = _load_financials_csv().get(row["nombre"])
    if _fin:
        market_val = _fin["market_value"]
        salary_val = _fin["salary"]
    else:
        _full_name = _resolve_full_name(row["nombre"], team=_player_team)
        market_val = _fetch_transfermarkt_value(_full_name, team=_player_team)
        salary_val = _fetch_capology_salary(_full_name, team=_player_team)
    mv_col, sal_col = st.columns(2)
    with mv_col:
        st.metric("💰 Transfermarkt Market Value", market_val or "N/A")
    with sal_col:
        st.metric("💶 Gross Annual Salary (Capology)", salary_val or "N/A")

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

    # ── Pizza Chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🍕 Percentile Pizza Chart{mode_label}")
    fig_pizza = _build_pizza_chart(row_data, peers, row["nombre"], position, is_gk=_is_gk)
    if fig_pizza:
        st.plotly_chart(fig_pizza, use_container_width=True)
    else:
        st.info("Not enough data for the pizza chart.")

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
                st.caption(f"{pct:.0f}th %ile · Benchmark: {p_avg:.0f}th")

    # ── Detailed Stats ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Detailed Statistics{mode_label}")
    if _is_gk:
        t1, t2, t3, t4 = st.tabs(["🧤 Shot-Stopping", "🏟️ Command", "📊 Distribution", "🧹 Sweeping"])
        with t1:
            ss = {m: round(row_data.get(m, 0) or 0, 2) for m in
                  ["Saves Made", "Save %",
                   "Goals Prevented",
                   "Total Big Chances Saved",
                   "Penalties Saved"] if m in peers.columns}
            if ss:
                st.dataframe(pd.DataFrame([ss]).T.rename(columns={0: "Value"}), use_container_width=True)
        with t2:
            cmd = {m: round(row_data.get(m, 0) or 0, 2) for m in
                   ["Catches", "Punches", "Aerial Duels won",
                    "Aerial Duels", "Aerial Win %"] if m in peers.columns}
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
        filtered = filtered[filtered["nombre"].str.contains(search, case=False, na=False)]

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

    # ── Squad Depth ────────────────────────────────────────────────────
    st.markdown("### 🏟️ Squad Depth")

    # ── Position coordinates on pitch (0-100 scale, bottom=own goal) ──
    _POS_XY = {
        "GK":  (50, 8),
        "LB":  (15, 25), "CB": (50, 22), "RB": (85, 25),
        "LWB": (10, 35), "RWB": (90, 35),
        "DM":  (50, 40),
        "LM":  (12, 55), "CM": (50, 55), "RM": (88, 55),
        "LAM": (20, 68), "CAM": (50, 68), "RAM": (80, 68),
        "LW":  (15, 78), "ST": (50, 85), "RW": (85, 78),
    }
    _EXTRA_XY = {"LCB": (35, 22), "RCB": (65, 22)}

    # Multi-position flexibility: a player listed as key can also cover values
    _POS_FLEX = {
        "ST": ["LW", "RW"],
        "LW": ["LM", "LAM", "ST", "LB", "LWB"],
        "RW": ["RM", "RAM", "ST", "RB", "RWB"],
        "LM": ["LW", "LAM"], "RM": ["RW", "RAM"],
        "LAM": ["LW", "LM"], "RAM": ["RW", "RM"],
        "LWB": ["LB", "LW"], "RWB": ["RB", "RW"],
        "LB": ["LWB", "LW"], "RB": ["RWB", "RW"],
        "DM": ["CM"], "CM": ["DM", "CAM"], "CAM": ["CM"],
    }

    # ── Build position → player names mapping ────────────────────────
    # Map posicion_detail broad names to pitch position codes
    _DETAIL_TO_CODE = {
        "Goalkeeper": "GK", "GK": "GK",
        "Defender": "CB", "CB": "CB", "Centre-Back": "CB",
        "FB": "LB",  # Full-backs: first pass → LB, overflow → RB
        "Full-Back": "LB",
        "LB": "LB", "RB": "RB", "LWB": "LWB", "RWB": "RWB",
        "Midfielder": "CM", "CM": "CM", "Central Midfield": "CM",
        "DM": "DM", "LM": "LM", "RM": "RM",
        "CAM": "CAM", "Attacking Midfield": "CAM",
        "LAM": "LAM", "RAM": "RAM",
        "Forward": "ST", "ST": "ST", "CF": "ST", "Striker": "ST",
        "LW": "LW", "RW": "RW", "Wingers": "LW",
    }
    _pos_players = {}  # pos -> [name, ...]
    _fb_names = []  # collect FB players to split LB/RB later
    _wing_names = []  # collect generic Wingers to split LW/RW
    _fwd_rows = []  # collect generic Forward rows to sub-classify ST vs LW/RW
    for _, r in squad.iterrows():
        pos_raw = r.get("posicion_detail", "Unknown")
        name = r.get("nombre", "Unknown")
        if pos_raw == "Unknown" or name == "Unknown":
            continue
        code = _DETAIL_TO_CODE.get(pos_raw)
        if code is None:
            continue
        if pos_raw in ("FB", "Full-Back"):
            _fb_names.append(name)
        elif pos_raw in ("Wingers",):
            _wing_names.append(name)
        elif pos_raw in ("Forward", "Striker"):
            _fwd_rows.append(r)
        else:
            _pos_players.setdefault(code, []).append(name)

    # Sub-classify Forwards into ST vs LW/RW using wide-play stats
    _FWD_WIDE = ["Successful Crosses open play", "Successful Dribbles",
                 "Total Touches In Opposition Box"]
    _FWD_CENTRAL = ["Goals", "Total Shots", "Aerial Duels won", "Headed Goals"]
    _fw_avail = [m for m in _FWD_WIDE if m in squad.columns]
    _fc_avail = [m for m in _FWD_CENTRAL if m in squad.columns]
    if len(_fwd_rows) > 2 and _fw_avail and _fc_avail:
        _fwd_df = pd.DataFrame(_fwd_rows)
        _fw_pct = _fwd_df[_fw_avail].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
        _fc_pct = _fwd_df[_fc_avail].apply(pd.to_numeric, errors="coerce").rank(pct=True).mean(axis=1)
        _balance = _fw_pct - _fc_pct
        _wide_fwds = []
        _central_fwds = []
        for i, r in enumerate(_fwd_rows):
            if _balance.iloc[i] > 0.1:
                _wide_fwds.append(r.get("nombre", "Unknown"))
            else:
                _central_fwds.append(r.get("nombre", "Unknown"))
        # Split wide forwards into LW / RW
        if _wide_fwds:
            half = len(_wide_fwds) // 2
            _pos_players.setdefault("LW", []).extend(_wide_fwds[:max(1, half)])
            _pos_players.setdefault("RW", []).extend(_wide_fwds[max(1, half):])
        _pos_players.setdefault("ST", []).extend(_central_fwds)
    else:
        for r in _fwd_rows:
            _pos_players.setdefault("ST", []).append(r.get("nombre", "Unknown"))

    # Split generic full-backs into LB / RB
    if _fb_names:
        half = len(_fb_names) // 2
        _pos_players.setdefault("LB", []).extend(_fb_names[:max(1, half)])
        _pos_players.setdefault("RB", []).extend(_fb_names[max(1, half):])

    # Split generic wingers into LW / RW
    if _wing_names:
        half = len(_wing_names) // 2
        _pos_players.setdefault("LW", []).extend(_wing_names[:max(1, half)])
        _pos_players.setdefault("RW", []).extend(_wing_names[max(1, half):])

    # Handle CB splitting: if multiple CBs, split into LCB / RCB
    _display_players = {}
    for pos, names in _pos_players.items():
        if pos == "CB" and len(names) > 1:
            half = len(names) // 2
            _display_players["LCB"] = names[:half]
            _display_players["RCB"] = names[half:]
        else:
            _display_players[pos] = names

    # ── Multi-position coverage: add players to empty nearby positions ──
    _all_xy = {**_POS_XY, **_EXTRA_XY}
    _occupied = set(_display_players.keys())
    _flex_additions = {}  # pos -> [name, ...]
    for pos, alts in _POS_FLEX.items():
        if pos in _occupied:
            continue  # already has players
        if pos not in _all_xy:
            continue
        for alt in alts:
            if alt in _pos_players:
                for n in _pos_players[alt]:
                    _flex_additions.setdefault(pos, []).append(n)
    # Merge flex additions (these are "can also play" suggestions)
    _flex_positions = set(_flex_additions.keys())

    fig_pitch = go.Figure()

    # ── Pitch shapes ────────────────────────────────────────────────
    _pitch_shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100,
             line=dict(color="white", width=2), fillcolor="#1a472a"),
        dict(type="line", x0=0, y0=50, x1=100, y1=50,
             line=dict(color="white", width=1)),
        dict(type="circle", x0=38, y0=38, x1=62, y1=62,
             line=dict(color="white", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=20, y0=0, x1=80, y1=16,
             line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=32, y0=0, x1=68, y1=6,
             line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=20, y0=84, x1=80, y1=100,
             line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=32, y0=94, x1=68, y1=100,
             line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
    ]

    # ── Plot position markers with player name lists ────────────────
    # Primary positions (solid markers)
    for pos, names in _display_players.items():
        if pos not in _all_xy:
            continue
        pos_x, pos_y = _all_xy[pos]
        name_list = "<br>".join(names[:6])  # cap at 6 for readability
        if len(names) > 6:
            name_list += f"<br>+{len(names)-6} more"
        fig_pitch.add_trace(go.Scatter(
            x=[pos_x], y=[pos_y], mode="markers",
            marker=dict(size=40, color="#2d6a4f",
                        line=dict(color="white", width=2), opacity=0.9),
            hovertext=f"<b>{pos}</b><br>{name_list}",
            hoverinfo="text", showlegend=False,
        ))
        fig_pitch.add_annotation(
            x=pos_x, y=pos_y + 1.5, text=f"<b>{pos}</b>",
            showarrow=False, font=dict(size=11, color="white"), opacity=0.95,
        )
        # Player names below the marker
        fig_pitch.add_annotation(
            x=pos_x, y=pos_y - 6, text="<br>".join(names[:4]),
            showarrow=False, font=dict(size=8, color="white"), opacity=0.85,
            align="center",
        )

    # Flex positions (dashed outline, lighter — "can also play here")
    for pos, names in _flex_additions.items():
        if pos not in _all_xy:
            continue
        pos_x, pos_y = _all_xy[pos]
        name_list = "<br>".join(names[:4])
        fig_pitch.add_trace(go.Scatter(
            x=[pos_x], y=[pos_y], mode="markers",
            marker=dict(size=35, color="rgba(45,106,79,0.35)",
                        line=dict(color="rgba(255,255,255,0.5)", width=1.5)),
            hovertext=f"<b>{pos}</b> (cover)<br>{name_list}",
            hoverinfo="text", showlegend=False,
        ))
        fig_pitch.add_annotation(
            x=pos_x, y=pos_y + 1.5, text=f"<b>{pos}</b>",
            showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.55)"),
        )
        fig_pitch.add_annotation(
            x=pos_x, y=pos_y - 6,
            text="<br>".join(f"({n})" for n in names[:3]),
            showarrow=False, font=dict(size=7, color="rgba(255,255,255,0.5)"),
            align="center",
        )

    fig_pitch.update_layout(
        shapes=_pitch_shapes,
        xaxis=dict(range=[-5, 105], visible=False, fixedrange=True),
        yaxis=dict(range=[-12, 112], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=700, width=550,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"{team_sel} — Squad Depth", font=dict(color="white")),
    )
    st.plotly_chart(fig_pitch, use_container_width=True)

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

    # ── Team Needs Analysis (Categorised) ─────────────────────────────────
    st.markdown("### 🔎 Team Needs Analysis")
    st.caption("Metrics where the team's per-player average falls below the league average, grouped by category.")
    any_need = False
    for cat_name, cat_metrics in TEAM_STAT_CATEGORIES.items():
        sq = squad[squad["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else squad
        lp = league_players[league_players["posicion"] == "Goalkeeper"] if cat_name == GK_CAT else league_players
        cat_below = []
        if sq.empty:
            continue
        for m in cat_metrics:
            if m not in sq.columns or m not in lp.columns:
                continue
            t_avg = sq[m].mean()
            l_avg = lp[m].mean()
            if l_avg > 0 and t_avg < l_avg:
                pct_gap = round((l_avg - t_avg) / l_avg * 100, 1)
                cat_below.append({
                    "Metric": m,
                    "Team Avg": round(t_avg, 2),
                    "League Avg": round(l_avg, 2),
                    "Gap": round(l_avg - t_avg, 2),
                    "Gap %": f"{pct_gap}%",
                })
        if cat_below:
            any_need = True
            needs_df = pd.DataFrame(cat_below).sort_values("Gap", ascending=False)
            with st.expander(f"{cat_name}  —  {len(cat_below)} areas below average", expanded=True):
                st.dataframe(needs_df.reset_index(drop=True), use_container_width=True)
    if not any_need:
        st.success("This team is above the league average in all tracked metrics!")

    # ── Player Scorecard & Grading ──────────────────────────────────────
    st.markdown("### 🃏 Player Scorecard & Grading")
    st.caption(
        "Each player is graded using role-specific KPI profiles — the same system as the Player Profile. "
        "The overall grade is based on whichever role fits the player best within their league peers. "
        "Grades range from S+ (97th+ percentile) to F (below 45th)."
    )

    # Local grade-color helper matching the 16-tier system
    def _sc_grade_color(grade):
        return _GRADE_COLORS.get(grade, "#555")

    # ── Separate outfield vs GK ─────────────────────────────────────
    _outfield_squad = squad[squad["posicion"] != "Goalkeeper"].copy()
    _gk_squad = squad[squad["posicion"] == "Goalkeeper"].copy()

    # ── Compute best-role grades for each squad player ───────────────
    _player_scorecards = {}  # name -> {role, position, overall_grade, overall_pct, attr_grades: {cat: (grade, pct)}}

    # --- Build league reference that matches the active stat mode ---
    _league_raw = df[df["league_display"] == league_sel]
    if stat_mode in ("Per 90", "Padj Per 90") and "estimated_90s" in _league_raw.columns:
        _MIN_90S_SC = 5
        _is_team = _league_raw["equipo"] == team_sel
        _has_mins = _league_raw["estimated_90s"].fillna(0) >= _MIN_90S_SC
        _league_df = _league_raw[_is_team | _has_mins].copy()
    else:
        _league_df = _league_raw.copy()

    # --- Outfield players ---
    for _, p_row in _outfield_squad.iterrows():
        name = p_row.get("nombre", "Unknown")
        position = p_row.get("posicion", "Unknown")
        if name == "Unknown" or position == "Unknown":
            continue

        # Try all roles for this position, pick the one with the highest overall pct
        _available_roles = list(POSITION_ROLE_PROFILES.get(position, {}).keys())
        if not _available_roles:
            _available_roles = [_classify_role(p_row, position, _league_df)]

        best_role = None
        best_pct = -1
        best_attr = {}
        for _try_role in _available_roles:
            _kpi = _ROLE_KPI_PROFILES.get(_try_role)
            attr = _compute_attribute_grades(
                dict(p_row), position, _league_df,
                league=league_sel, role=None,
                df_role_ref=_league_df, kpi_role=_try_role,
            )
            if _kpi:
                weights = {n: w for n, (w, _) in _kpi.items()}
            else:
                weights = _POSITION_GRADE_WEIGHTS.get(position, {})
            pcts = {k: pct for k, (_, pct) in attr.items() if pct is not None}
            if not pcts:
                continue
            if weights:
                w_sum = sum(weights.get(k, 0) for k in pcts)
                if w_sum > 0:
                    ov_pct = sum(pcts[k] * weights.get(k, 0) for k in pcts) / w_sum
                else:
                    ov_pct = sum(pcts.values()) / len(pcts)
            else:
                ov_pct = sum(pcts.values()) / len(pcts)
            if ov_pct > best_pct:
                best_pct = ov_pct
                best_role = _try_role
                best_attr = attr

        if best_role is None:
            continue

        _player_scorecards[name] = {
            "role": best_role,
            "position": position,
            "overall_grade": _percentile_to_grade(best_pct),
            "overall_pct": round(best_pct, 1),
            "attr_grades": best_attr,
            "is_gk": False,
        }

    # --- Goalkeepers ---
    for _, p_row in _gk_squad.iterrows():
        name = p_row.get("nombre", "Unknown")
        if name == "Unknown":
            continue
        attr = _compute_attribute_grades(
            dict(p_row), "Goalkeeper", _league_df,
            league=league_sel, role=None,
            df_role_ref=_league_df, kpi_role="Goalkeeper",
        )
        pcts = {k: pct for k, (_, pct) in attr.items() if pct is not None}
        ov_pct = sum(pcts.values()) / len(pcts) if pcts else 0
        _player_scorecards[name] = {
            "role": "Goalkeeper",
            "position": "Goalkeeper",
            "overall_grade": _percentile_to_grade(ov_pct),
            "overall_pct": round(ov_pct, 1),
            "attr_grades": attr,
            "is_gk": True,
        }

    # ── Build dropdown sorted by overall ────────────────────────────
    is_gk = False
    if not _player_scorecards:
        st.info(f"No graded players found for {team_sel}.")
    else:
        _sorted_cards = sorted(_player_scorecards.items(),
                               key=lambda x: x[1]["overall_pct"], reverse=True)
        _dropdown_options = []
        for name, sc in _sorted_cards:
            prefix = "🧤 " if sc["is_gk"] else ""
            cat_summary = " | ".join(
                f"{cn}: {cg}" for cn, (cg, cp) in sc["attr_grades"].items() if cp is not None
            )
            label = (f"{prefix}{name}  —  Overall: {sc['overall_grade']} "
                     f"({sc['overall_pct']:.0f}th)  ·  {sc['role']}  |  {cat_summary}")
            _dropdown_options.append((label, name))

        _labels = [opt[0] for opt in _dropdown_options]
        _sel_label = st.selectbox("Select a player", _labels, key="sc_player_sel")
        _sel_idx = _labels.index(_sel_label)
        _, player_name = _dropdown_options[_sel_idx]
        sc = _player_scorecards[player_name]
        is_gk = sc["is_gk"]
        attr_grades = sc["attr_grades"]

        # ── Header ──
        st.markdown(
            f"#### {'🧤 ' if is_gk else ''}{player_name}  —  Overall: "
            f"**{sc['overall_grade']}** ({sc['overall_pct']:.0f}th pctile)  ·  Role: **{sc['role']}**"
        )

        # ── Attribute tabs ──
        _valid_attrs = {k: v for k, v in attr_grades.items() if v[1] is not None}
        if _valid_attrs:
            cat_tabs = st.tabs(list(_valid_attrs.keys()))
            _kpi = _ROLE_KPI_PROFILES.get(sc["role"])
            for tab, (cat_name, (cat_grade, cat_pct)) in zip(cat_tabs, _valid_attrs.items()):
                with tab:
                    st.markdown(f"**{cat_name}**: {cat_grade} ({cat_pct:.0f}th pctile)")
                    # Get individual metric percentiles for this category
                    if _kpi and cat_name in _kpi:
                        _, cat_metrics = _kpi[cat_name]
                    elif is_gk and cat_name in GK_ATTRIBUTE_GRADE_CATEGORIES:
                        cat_metrics = GK_ATTRIBUTE_GRADE_CATEGORIES[cat_name]
                    elif cat_name in ATTRIBUTE_GRADE_CATEGORIES:
                        cat_metrics = ATTRIBUTE_GRADE_CATEGORIES[cat_name]
                    else:
                        cat_metrics = []

                    _avail = [m for m in cat_metrics if m in _league_df.columns]
                    if not _avail:
                        continue

                    # Compute individual metric percentiles vs league+position peers
                    _peers = _league_df[_league_df["posicion"] == sc["position"]]
                    if len(_peers) < 5:
                        _PEER_FALLBACK = {"Wingers": "Striker"}
                        fb_pos = _PEER_FALLBACK.get(sc["position"])
                        if fb_pos:
                            _peers = _league_df[_league_df["posicion"] == fb_pos]

                    _inv = _KPI_INVERTED_CATS if _kpi else _INVERTED_GRADE_CATS
                    p_row_data = squad[squad["nombre"] == player_name].iloc[0] if not squad[squad["nombre"] == player_name].empty else None
                    if p_row_data is None or len(_peers) < 5:
                        continue

                    detail_rows = []
                    metric_pcts = []
                    for m in _avail:
                        val = p_row_data.get(m, 0)
                        val = 0 if pd.isna(val) else val
                        peer_vals = _peers[m].fillna(0)
                        pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                        if m in _inv:
                            pct = 100 - pct
                        g = _percentile_to_grade(pct)
                        raw_val = round(float(val), 2) if pd.notna(val) else "-"
                        detail_rows.append({
                            "Metric": m,
                            f"Value{mode_label}": raw_val,
                            "Pctile": f"{pct:.0f}",
                            "Grade": g,
                        })
                        metric_pcts.append((m, pct, g))

                    if not detail_rows:
                        continue

                    fig_sc = go.Figure(go.Bar(
                        x=[r["Metric"] for r in detail_rows],
                        y=[p for _, p, _ in metric_pcts],
                        marker_color=[_sc_grade_color(g) for _, _, g in metric_pcts],
                        text=[g for _, _, g in metric_pcts],
                        textposition="outside",
                        customdata=list(zip(
                            [r[f"Value{mode_label}"] for r in detail_rows],
                            [f"{p:.0f}" for _, p, _ in metric_pcts],
                            [g for _, _, g in metric_pcts],
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

    # Player's own percentile vector
    idx = peers.index[peers["nombre"] == player_row["nombre"]]
    if idx.empty:
        return pd.DataFrame()
    player_vec = pct_df.loc[idx[0]].values.astype(float)

    # Cosine similarity
    norms = np.linalg.norm(pct_df.values, axis=1) * np.linalg.norm(player_vec)
    norms[norms == 0] = 1
    cos_sim = pct_df.values.dot(player_vec) / norms

    peers = peers.copy()
    peers["Similarity %"] = (cos_sim * 100).round(1)
    # Exclude the player themselves
    peers = peers[peers["nombre"] != player_row["nombre"]]
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
        "Stopper": ["Total Tackles", "Interceptions", "Total Clearances",
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


def render_player_comparison(data):
    st.subheader("⚔️ Player Comparison")

    compare_mode = st.radio("Mode", ["🔍 Find Similar Players", "⚔️ Compare Players"],
                            horizontal=True, key="cmp_mode_sel")

    # ── Stat mode ────────────────────────────────────────────────────────
    stat_mode = st.radio("Stat mode", _STAT_MODES, horizontal=True, key="cmp_stat_mode")
    df = _select_df(data, stat_mode)
    mode_label = f" ({stat_mode})" if stat_mode != "Total" else ""

    if compare_mode == "🔍 Find Similar Players":
        _render_find_similar(df, mode_label)
    else:
        _render_head_to_head(df, mode_label)


def _render_find_similar(df, mode_label):
    """Find players with a similar statistical profile."""
    st.markdown("---")
    st.markdown("#### Select a player to find similar profiles")

    # ── Player search ────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        all_names = sorted(df["nombre"].unique())
        player_sel = st.selectbox("Type or select a player", all_names, index=None,
                                  placeholder="Start typing a name...", key="sim_player")
    with c2:
        n_results = st.slider("Number of similar players", 5, 20, 10, key="sim_n")

    if not player_sel:
        st.info("Select a player above to find players with a similar statistical profile.")
        return

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
        idx = peers.index[peers["nombre"] == player_row["nombre"]]
        similar = pd.DataFrame()
        if not idx.empty:
            player_vec = pct_df.loc[idx[0]].values.astype(float)
            norms = np.linalg.norm(pct_df.values, axis=1) * np.linalg.norm(player_vec)
            norms[norms == 0] = 1
            cos_sim = pct_df.values.dot(player_vec) / norms
            peers = peers.copy()
            peers["Similarity %"] = (cos_sim * 100).round(1)
            peers = peers[peers["nombre"] != player_row["nombre"]]
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
    st.markdown(f"### Players similar to **{player_sel}**")
    st.caption(f"Profile: **{profile_sel}** | Metrics: {', '.join(avail_metrics)} | "
               f"Position: {position} {'(all positions searched)' if scope == 'All positions' else ''}")

    # ── Results table ────────────────────────────────────────────────────
    show_cols = ["nombre", "equipo", "league_display", "posicion_detail", "Similarity %"] + avail_metrics
    show_cols = [c for c in show_cols if c in similar.columns]
    st.dataframe(similar[show_cols].reset_index(drop=True), use_container_width=True)

    # ── Radar: target player vs top 3 similar ────────────────────────────
    top_names = similar["nombre"].head(3).tolist()
    radar_names = [player_sel] + top_names
    st.markdown(f"### 🕸️ Radar: {player_sel} vs Top Matches{mode_label}")
    fig = chart_radar(df, radar_names, avail_metrics,
                      f"{player_sel} vs Similar Players{mode_label}")
    st.plotly_chart(fig, use_container_width=True)

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

    # ── Player selection ─────────────────────────────────────────────────
    all_players = sorted(filt["nombre"].unique())
    selected = st.multiselect("Select players to compare (2–6)", all_players,
                              max_selections=6, key="cmp_players")

    if len(selected) < 2:
        st.info("Pick at least **2 players** to start the comparison.")
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
    fig = chart_radar(filt, selected, sel_metrics, f"Player Comparison{mode_label}")
    st.plotly_chart(fig, use_container_width=True)

    # ── Side-by-side stats table ─────────────────────────────────────────
    st.markdown("### 📋 Side-by-Side Stats")
    comp_rows = filt[filt["nombre"].isin(selected)].copy()
    display_cols = ["nombre", "equipo", "league_display", "posicion_detail"] + sel_metrics
    display_cols = [c for c in display_cols if c in comp_rows.columns]
    st.dataframe(comp_rows[display_cols].reset_index(drop=True), use_container_width=True)

    # ── Difference bar chart ─────────────────────────────────────────────
    if len(selected) == 2:
        st.markdown("### 📊 Head-to-Head Difference")
        p1_row = comp_rows[comp_rows["nombre"] == selected[0]]
        p2_row = comp_rows[comp_rows["nombre"] == selected[1]]
        if not p1_row.empty and not p2_row.empty:
            diffs = []
            for m in sel_metrics:
                v1 = p1_row.iloc[0].get(m, 0) or 0
                v2 = p2_row.iloc[0].get(m, 0) or 0
                diffs.append(round(v1 - v2, 2))
            fig_diff = go.Figure(go.Bar(
                x=sel_metrics, y=diffs,
                marker_color=["#2d6a4f" if d >= 0 else "#e63946" for d in diffs],
                text=[f"+{d}" if d > 0 else str(d) for d in diffs],
                textposition="outside",
            ))
            fig_diff.update_layout(
                title=f"{selected[0]} minus {selected[1]}",
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🕵️ FORENSICS XG: INTELLIGENCE UNIT")
    st.markdown("#### WHERE THE BEAUTIFUL GAME MEETS HARD EVIDENCE.")

    # Load data
    data = load_data()
    df_total = data["total"]

    if df_total.empty:
        # Diagnostic info for debugging cloud deployment
        _diag_lines = [f"OPTA_DIR = {OPTA_DIR}"]
        for _dn, _fn in LEAGUE_FOLDERS.items():
            _fp = os.path.join(OPTA_DIR, _fn)
            _exists = os.path.isdir(_fp)
            _csv = os.path.join(_fp, "jugadores_seasonstats.csv")
            _csv_exists = os.path.exists(_csv)
            _diag_lines.append(f"  {_dn}: dir={_exists}, csv={_csv_exists}")
        st.error("No data found. Make sure the Opta league folders are in the correct directory.\n\n" + "\n".join(_diag_lines))
        return

    # Tabs
    tab_analysis, tab_profile, tab_compare, tab_team, tab_team_cmp, tab_explorer = st.tabs([
        "🔬 Player Lab", "🪪 Player Profile",
        "⚔️ Player Comparison", "🏟️ Team Profile", "🏟️ Team Comparison", "🔍 Data Explorer"
    ])

    with tab_analysis:
        render_player_lab(data)
    with tab_profile:
        render_profile(data)
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

