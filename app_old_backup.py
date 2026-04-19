"""
Football Analytics AI - Interactive football data analysis tool.
Ask natural language questions and get charts, stats, and insights
from Europe's top 6 leagues (2025-2026 season).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
import re
import urllib.request
import urllib.parse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FORENSICS XG: INTELLIGENCE UNIT",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

LEAGUE_FOLDERS = {
    "Premier League": "Premier League",
    "LaLiga": "LaLiga",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Serie A": "Serie A",
    "Primeira Liga": "Primeira Liga",
}

CHART_COLORS = px.colors.qualitative.Vivid

META_COLS = {"liga", "temporada", "equipo", "id", "nombre", "posicion", "dorsal", "league_display"}

OFFENSIVE_METRICS = [
    "Goals", "Goal Assists", "Total Shots", "Shots On Target ( inc goals )",
    "Goals from Inside Box", "Goals from Outside Box", "Total Big Chances Scored",
    "Total Big Chances Missed", "Key Passes (Attempt Assists)", "Goals Openplay",
]
DEFENSIVE_METRICS = [
    "Total Tackles", "Tackles Won", "Interceptions", "Total Clearances",
    "Blocks", "Blocked Shots", "Aerial Duels won", "Ground Duels won", "Recoveries",
]
PASSING_METRICS = [
    "Total Passes", "Total Successful Passes ( Excl Crosses & Corners )",
    "Successful Long Passes", "Through balls", "Forward Passes",
    "Progressive Carries", "Successful Crosses & Corners", "Successful Dribbles",
]
GENERAL_METRICS = ["Appearances", "Starts", "Time Played", "Yellow Cards", "Total Red Cards"]

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { }
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

@st.cache_data(show_spinner="Loading football data...")
def load_data() -> pd.DataFrame:
    frames = []
    for display_name, folder in LEAGUE_FOLDERS.items():
        path = os.path.join(DATA_DIR, folder, "jugadores_seasonstats.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            df["league_display"] = display_name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Convert stat columns to numeric
    for col in combined.columns:
        if col not in META_COLS:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def add_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90-minute columns for key metrics."""
    out = df.copy()
    for m in ["Goals", "Goal Assists", "Total Shots", "Key Passes (Attempt Assists)",
              "Successful Dribbles", "Total Tackles", "Interceptions"]:
        if m in out.columns:
            out[f"{m} per 90"] = np.where(
                out["Time Played"] > 0,
                (out[m] / out["Time Played"]) * 90, 0
            ).round(2)
    return out


# All stat columns that should be converted when using per-90 mode
_ALL_STAT_COLS = (
    OFFENSIVE_METRICS + DEFENSIVE_METRICS + PASSING_METRICS +
    ["Aerial Duels", "Aerial Duels won", "Duels won", "Ground Duels won",
     "Carries", "Progressive Carries", "Through balls", "Saves Made",
     "Total Big Chances Scored", "Total Big Chances Created", "Blocked Shots",
     "Total Fouls Conceded", "Total Red Cards", "Yellow Cards"]
)


def _to_per90_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all stat columns in-place to per-90-minute values."""
    out = df.copy()
    tp = out["Time Played"].fillna(0)
    for col in _ALL_STAT_COLS:
        if col in out.columns:
            out[col] = np.where(tp > 0, (out[col].fillna(0) / tp) * 90, 0).round(3)
    return out


def _row_to_per90(row_series, minutes):
    """Convert a single player row dict to per-90 values."""
    out = dict(row_series)
    mins = minutes if minutes and minutes > 0 else 0
    for col in _ALL_STAT_COLS:
        if col in out:
            val = out[col]
            val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
            out[col] = round((val / mins) * 90, 3) if mins > 0 else 0
    return out


def filter_df(df, leagues=None, teams=None, positions=None, min_minutes=0):
    """Apply common filters."""
    out = df.copy()
    if leagues:
        out = out[out["league_display"].isin(leagues)]
    if teams:
        out = out[out["equipo"].isin(teams)]
    if positions:
        out = out[out["posicion"].isin(positions)]
    if min_minutes > 0 and "Time Played" in out.columns:
        out = out[out["Time Played"] >= min_minutes]
    return out

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
    """Radar chart with values normalized to 0-100 vs dataset max."""
    fig = go.Figure()
    for name in player_names:
        rows = df_full[df_full["nombre"].str.contains(re.escape(name), case=False, na=False)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        vals = []
        for m in metrics:
            v = row.get(m, 0)
            v = 0 if pd.isna(v) else v
            mx = df_full[m].max() if m in df_full.columns else 1
            vals.append(round(v / mx * 100, 1) if mx > 0 else 0)
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]],
            fill="toself", name=f"{row['nombre']} ({row['equipo']})", opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=title, height=560, template="plotly_white", showlegend=True,
    )
    return fig


def chart_pie(data, names, values, title, height=500):
    fig = px.pie(data, names=names, values=values, title=title,
                 color_discrete_sequence=CHART_COLORS, height=height)
    fig.update_layout(template="plotly_white")
    return fig

# ── AI Engine ────────────────────────────────────────────────────────────────

def _build_system_prompt(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in {"liga", "temporada", "id", "league_display"}]
    leagues = sorted(df["league_display"].unique().tolist())
    teams_sample = sorted(df["equipo"].unique().tolist())[:25]
    positions = sorted(df["posicion"].dropna().unique().tolist())
    return f"""You are a football/soccer analytics assistant. Given a user question about player data, return ONLY a JSON object (no markdown, no extra text) describing a chart to build.

DATA: Player season stats from 2025-2026 for these leagues: {leagues}
POSITIONS: {positions}
SAMPLE TEAMS: {teams_sample}
COLUMNS: {json.dumps(cols)}

JSON SCHEMA:
{{
  "intent": "chart" | "table" | "number",
  "chart_type": "bar" | "scatter" | "pie" | "radar" | "box" | "histogram",
  "title": "string",
  "description": "brief answer text",
  "filters": [{{"column":"...","op":"==|!=|>|<|>=|<=|in|contains","value":"..."}}],
  "sort_by": "column or null",
  "sort_ascending": false,
  "limit": 10,
  "x": "column",
  "y": "column",
  "color": "column or null",
  "orientation": "v|h",
  "group_by": "column or null",
  "agg_func": "sum|mean|count|max|min|null",
  "players_compare": ["name1","name2"],
  "metrics": ["col1","col2"]
}}

RULES:
- Use EXACT column names from the COLUMNS list.
- For name/team searches use op "contains" with partial text.
- Radar charts need players_compare + metrics (5-8 metrics).
- Filter out players with 0 or null in the main metric when ranking.
- Default limit 10 unless user specifies otherwise.
- Return ONLY valid JSON."""


def query_openai(question: str, df: pd.DataFrame, api_key: str, model: str) -> dict:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _build_system_prompt(df)},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    return json.loads(text)


def execute_spec(spec: dict, df: pd.DataFrame):
    """Build a chart from a JSON spec. Returns (figure, description)."""
    filtered = df.copy()

    # Apply filters
    for f in spec.get("filters", []):
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in filtered.columns:
            continue
        if op == "==":
            filtered = filtered[filtered[col] == val]
        elif op == "!=":
            filtered = filtered[filtered[col] != val]
        elif op == ">":
            filtered = filtered[filtered[col] > float(val)]
        elif op == "<":
            filtered = filtered[filtered[col] < float(val)]
        elif op == ">=":
            filtered = filtered[filtered[col] >= float(val)]
        elif op == "<=":
            filtered = filtered[filtered[col] <= float(val)]
        elif op == "in":
            filtered = filtered[filtered[col].isin(val if isinstance(val, list) else [val])]
        elif op == "contains":
            filtered = filtered[filtered[col].astype(str).str.contains(str(val), case=False, na=False)]

    # Aggregation
    y_col = spec.get("y", "Goals")
    if spec.get("group_by") and spec.get("agg_func"):
        gb = spec["group_by"]
        if gb in filtered.columns and y_col in filtered.columns:
            filtered = filtered.groupby(gb, as_index=False).agg({y_col: spec["agg_func"]})

    # Sort
    sort_col = spec.get("sort_by")
    if sort_col and sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=spec.get("sort_ascending", False))

    # Limit
    lim = spec.get("limit")
    if lim:
        filtered = filtered.head(int(lim))

    # Build chart
    ct = spec.get("chart_type", "bar")
    title = spec.get("title", "Chart")
    x = spec.get("x", "nombre")
    color = spec.get("color")
    orient = spec.get("orientation", "v")

    if ct == "radar":
        players = spec.get("players_compare", [])
        metrics = spec.get("metrics", [])
        if players and metrics:
            fig = chart_radar(df, players, metrics, title)
        else:
            fig = chart_bar(filtered, x, y_col, title, color, orient)
    elif ct == "scatter":
        fig = chart_scatter(filtered, x, y_col, title, color, spec.get("size"))
    elif ct == "pie":
        fig = chart_pie(filtered, x, y_col, title)
    elif ct == "box":
        fig = px.box(filtered, x=x, y=y_col, title=title, color=color,
                     color_discrete_sequence=CHART_COLORS, height=520)
        fig.update_layout(template="plotly_white")
    elif ct == "histogram":
        fig = px.histogram(filtered, x=y_col, title=title, color=color,
                           color_discrete_sequence=CHART_COLORS, height=520)
        fig.update_layout(template="plotly_white")
    else:
        fig = chart_bar(filtered, x, y_col, title, color, orient)

    return fig, spec.get("description", "")

# ── Smart Parser (no-API fallback) ───────────────────────────────────────────

LEAGUE_KEYWORDS = {
    "premier league": "Premier League", "epl": "Premier League", "english": "Premier League", "england": "Premier League",
    "la liga": "LaLiga", "laliga": "LaLiga", "spanish": "LaLiga", "spain": "LaLiga",
    "bundesliga": "Bundesliga", "german": "Bundesliga", "germany": "Bundesliga",
    "ligue 1": "Ligue 1", "french": "Ligue 1", "france": "Ligue 1",
    "serie a": "Serie A", "italian": "Serie A", "italy": "Serie A",
    "primeira liga": "Primeira Liga", "portuguese": "Primeira Liga", "portugal": "Primeira Liga",
}

METRIC_KEYWORDS = {
    "goal": "Goals", "scorer": "Goals", "scoring": "Goals",
    "assist": "Goal Assists", "provider": "Goal Assists",
    "shot": "Total Shots", "shooting": "Total Shots",
    "tackle": "Total Tackles", "tackling": "Total Tackles",
    "pass": "Total Passes", "passing": "Total Passes",
    "intercept": "Interceptions",
    "clearance": "Total Clearances",
    "dribbl": "Successful Dribbles",
    "foul won": "Total Fouls Won",
    "yellow": "Yellow Cards", "booking": "Yellow Cards",
    "red card": "Total Red Cards",
    "appearance": "Appearances",
    "minute": "Time Played", "played": "Time Played",
    "save": "Saves Made",
    "clean sheet": "Clean Sheets",
    "aerial": "Aerial Duels won",
    "key pass": "Key Passes (Attempt Assists)",
    "big chance": "Total Big Chances Scored",
    "carry": "Carries", "progressive": "Progressive Carries",
    "through ball": "Through balls",
    "cross": "Successful Crosses & Corners",
    "block": "Blocks",
    "recover": "Recoveries",
}

# Team-need analysis categories: used when user asks "what players does X need"
TEAM_ANALYSIS_CATEGORIES = {
    "Goalscoring": {
        "metrics": ["Goals", "Goals Openplay", "Total Big Chances Scored", "Shots On Target ( inc goals )"],
        "description": "finishing & goal threat",
    },
    "Creativity": {
        "metrics": ["Goal Assists", "Key Passes (Attempt Assists)", "Through balls", "Total Big Chances Created"],
        "description": "chance creation",
    },
    "Defence": {
        "metrics": ["Total Tackles", "Interceptions", "Recoveries", "Total Clearances"],
        "description": "defensive solidity",
    },
    "Passing": {
        "metrics": ["Total Passes", "Successful Long Passes", "Progressive Carries", "Forward Passes"],
        "description": "ball progression & passing",
    },
    "Dribbling": {
        "metrics": ["Successful Dribbles", "Carries", "Progressive Carries"],
        "description": "ball carrying & dribbling",
    },
    "Aerial": {
        "metrics": ["Aerial Duels won", "Total Clearances"],
        "description": "aerial dominance",
    },
}

POSITION_KEYWORDS = {
    "forward": "Forward", "striker": "Forward", "attacker": "Forward",
    "midfielder": "Midfielder", "midfield": "Midfielder",
    "defender": "Defender", "centre back": "Defender", "fullback": "Defender",
    "goalkeeper": "Goalkeeper", "keeper": "Goalkeeper", "goalie": "Goalkeeper",
}

# Patterns that imply a defensive or attacking role even without an explicit metric keyword
ROLE_PATTERNS = {
    "defensive": {
        "metric": "Total Tackles",
        "multi_sort": ["Total Tackles", "Interceptions", "Recoveries"],
    },
    "holding": {
        "metric": "Total Tackles",
        "multi_sort": ["Total Tackles", "Interceptions", "Recoveries"],
    },
    "attacking": {
        "metric": "Goals",
        "multi_sort": ["Goals", "Goal Assists", "Key Passes (Attempt Assists)"],
    },
    "creative": {
        "metric": "Key Passes (Attempt Assists)",
        "multi_sort": ["Key Passes (Attempt Assists)", "Goal Assists", "Through balls"],
    },
}


def _analyse_team_needs(team_query: str, df: pd.DataFrame):
    """Analyse a team's stats vs league averages and identify weak areas."""
    # Find the team
    matches = df[df["equipo"].str.contains(re.escape(team_query), case=False, na=False)]
    if matches.empty:
        return None, f"Could not find a team matching **'{team_query}'**. Try using the full team name."

    team_name = matches["equipo"].iloc[0]
    league = matches["league_display"].iloc[0]

    # Get league peers (only players with minutes)
    league_df = df[(df["league_display"] == league) & (df["Time Played"].notna()) & (df["Time Played"] > 90)]
    team_df = league_df[league_df["equipo"] == team_name]

    if team_df.empty:
        return None, f"No sufficient data for **{team_name}**."

    # Compute team totals vs league average team totals per category
    league_teams = league_df.groupby("equipo")
    results = {}
    for cat, info in TEAM_ANALYSIS_CATEGORIES.items():
        avail_metrics = [m for m in info["metrics"] if m in league_df.columns]
        if not avail_metrics:
            continue
        team_total = team_df[avail_metrics].sum().sum()
        all_totals = league_teams.apply(lambda g: g[avail_metrics].sum().sum(), include_groups=False)
        league_avg = all_totals.mean()
        league_max_val = all_totals.max()
        percentile = (all_totals < team_total).sum() / len(all_totals) * 100
        results[cat] = {
            "team_total": round(team_total, 1),
            "league_avg": round(league_avg, 1),
            "percentile": round(percentile, 1),
            "description": info["description"],
            "diff_pct": round((team_total - league_avg) / league_avg * 100, 1) if league_avg > 0 else 0,
        }

    if not results:
        return None, "Insufficient data for team analysis."

    # Sort by weakness (lowest percentile first)
    sorted_cats = sorted(results.items(), key=lambda x: x[1]["percentile"])

    # Build radar chart showing team profile
    categories = [cat for cat, _ in sorted_cats]
    percentiles = [info["percentile"] for _, info in sorted_cats]
    categories_closed = categories + [categories[0]]
    percentiles_closed = percentiles + [percentiles[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=percentiles_closed, theta=categories_closed,
        fill="toself", name=team_name,
        line=dict(color="#e63946", width=2), fillcolor="rgba(230,57,70,0.25)",
    ))
    # Add 50th percentile reference
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(categories_closed), theta=categories_closed,
        name="League Average (50th %ile)",
        line=dict(color="gray", dash="dash", width=1), fill=None,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{team_name} — Team Profile vs {league} Average",
        height=560, template="plotly_white", showlegend=True,
    )

    # Build text description
    weaknesses = [f"**{cat}** ({info['description']}, {info['percentile']:.0f}th %ile, {info['diff_pct']:+.0f}% vs avg)"
                  for cat, info in sorted_cats if info["percentile"] < 40]
    strengths = [f"**{cat}** ({info['description']}, {info['percentile']:.0f}th %ile)"
                 for cat, info in sorted_cats if info["percentile"] >= 65]

    desc = f"### {team_name} — Squad Analysis ({league})\n\n"
    if weaknesses:
        desc += "**🔴 Areas to reinforce:**\n" + "\n".join(f"- {w}" for w in weaknesses) + "\n\n"
        weak_cats = [cat for cat, info in sorted_cats if info["percentile"] < 40]
        if "Goalscoring" in weak_cats:
            desc += "→ *Consider signing a clinical striker or goal-threat forward.*\n"
        if "Creativity" in weak_cats:
            desc += "→ *Look for a creative midfielder / playmaker with high assist numbers.*\n"
        if "Defence" in weak_cats:
            desc += "→ *Prioritize a commanding centre-back or disciplined defensive midfielder.*\n"
        if "Passing" in weak_cats:
            desc += "→ *A deep-lying playmaker who can progress the ball would help.*\n"
        if "Dribbling" in weak_cats:
            desc += "→ *An agile winger or dribbler to beat defenders in 1v1 situations.*\n"
        if "Aerial" in weak_cats:
            desc += "→ *Add aerial presence — a tall centre-back or target-man forward.*\n"
        desc += "\n"
    else:
        desc += "No major weaknesses detected — this is a well-rounded squad.\n\n"
    if strengths:
        desc += "**🟢 Strengths:**\n" + "\n".join(f"- {s}" for s in strengths) + "\n"

    return fig, desc


def smart_parse(question: str, df: pd.DataFrame):
    """Attempt to answer a question using keyword matching. Returns (fig, description) or (None, msg)."""
    q = question.lower().strip()

    # ── Team-needs detection ─────────────────────────────────────────────
    need_patterns = [
        r"what (?:kind of |type of )?players? (?:do|does|should|would|could) (.+?) (?:need|sign|acquire|buy|get|recruit|target|fit)",
        r"(?:which|what) players? (?:should|would|could|can|might|will) (.+?) (?:sign|acquire|buy|get|recruit|target)",
        r"(?:which|what) players? (?:would|could|can|might|should|will) (?:fit|suit|help|improve|strengthen|benefit|work for|be good for) (.+)",
        r"who (?:should|would|could|can) (.+?) (?:sign|acquire|buy|get|recruit|target)",
        r"(?:suggest|recommend).* (?:for|to) (.+)",
        r"(?:players?|signings?|transfers?) (?:that |who |which )?(?:would |could |can |might )?(?:fit|suit|help|improve|strengthen|benefit|work for|be good for) (.+)",
        r"what (?:do|does) (.+?) need",
        r"(.+?) (?:need|needs|lacking|missing|weak|weakness)",
        r"improve (.+?)(?:'s| squad| team| roster)",
        r"reinforce (.+)",
        r"signings? for (.+)",
        r"transfer targets? for (.+)",
        r"recruit.* for (.+)",
        r"(.+?) (?:should (?:sign|acquire|buy|get))",
        r"best (?:signings?|transfers?|acquisitions?|players?) for (.+)",
        r"who (?:would|could|can) (.+?) (?:sign|acquire|buy)",
        r"(?:analyse|analyze|analysis|scout|scouting) (.+?)(?:'s)? (?:squad|team|roster|needs)",
    ]
    team_match = None
    for pat in need_patterns:
        m = re.search(pat, q)
        if m:
            team_match = m.group(1).strip().strip("?.,!")
            break

    if team_match:
        return _analyse_team_needs(team_match, df)

    # ── Standard query flow ──────────────────────────────────────────────

    # Extract number
    nums = re.findall(r"\b(\d+)\b", q)
    n = min(int(nums[0]), 50) if nums else 10

    # Detect league
    league = None
    for kw, val in LEAGUE_KEYWORDS.items():
        if kw in q:
            league = val
            break

    # Detect explicit metric
    metric = None
    for kw, val in METRIC_KEYWORDS.items():
        if kw in q:
            metric = val
            break

    # Detect position
    pos = None
    for kw, val in POSITION_KEYWORDS.items():
        if kw in q:
            pos = val
            break

    # Detect role modifier (e.g. "defensive midfielder", "attacking fullback")
    role_info = None
    for kw, info in ROLE_PATTERNS.items():
        if kw in q:
            role_info = info
            break

    # If no explicit metric was found, infer from role or fall back to position defaults
    if metric is None:
        if role_info:
            metric = role_info["metric"]
        elif pos == "Goalkeeper":
            metric = "Saves Made"
        elif pos == "Defender":
            metric = "Total Tackles"
        else:
            metric = "Goals"

    # Filter
    filtered = df.copy()
    if league:
        filtered = filtered[filtered["league_display"] == league]
    if pos:
        filtered = filtered[filtered["posicion"] == pos]
    if metric in filtered.columns:
        filtered = filtered[filtered[metric].notna() & (filtered[metric] > 0)]

    if metric not in filtered.columns or filtered.empty:
        return None, "I couldn't find data matching your question. Try asking about top scorers, most assists, best tacklers, etc."

    # Multi-metric composite score when a role modifier was detected
    if role_info and not any(kw in q for kw in METRIC_KEYWORDS):
        sort_cols = [c for c in role_info["multi_sort"] if c in filtered.columns]
        if sort_cols:
            for c in sort_cols:
                mx = filtered[c].max()
                filtered[f"_norm_{c}"] = filtered[c].fillna(0) / mx if mx > 0 else 0
            filtered["_composite"] = sum(filtered[f"_norm_{c}"] for c in sort_cols)
            top = filtered.nlargest(n, "_composite")
            metric_label = " + ".join(sort_cols)
            title = f"Top {n} Players by Composite ({metric_label})"
            parts = []
            if league:
                parts.append(league)
            if pos:
                parts.append(f"{pos}s")
            if role_info:
                role_name = [k for k, v in ROLE_PATTERNS.items() if v is role_info][0].title()
                parts.append(role_name)
            if parts:
                title += f" ({', '.join(parts)})"
            fig = chart_bar(top, "nombre", metric, title, color="equipo")
            desc = f"Showing top {n} players ranked by a composite of **{metric_label}**."
            if league:
                desc += f" Filtered to **{league}**."
            if pos:
                desc += f" Position: **{pos}**."
            return fig, desc

    # Build top-N chart
    top = filtered.nlargest(n, metric)
    title = f"Top {n} Players by {metric}"
    parts = []
    if league:
        parts.append(league)
    if pos:
        parts.append(f"{pos}s")
    if parts:
        title += f" ({', '.join(parts)})"

    fig = chart_bar(top, "nombre", metric, title, color="equipo")
    desc = f"Showing top {n} players by **{metric}**."
    if league:
        desc += f" Filtered to **{league}**."
    if pos:
        desc += f" Position: **{pos}**."
    return fig, desc

# ── UI: Chat Tab ─────────────────────────────────────────────────────────────

def render_chat(df: pd.DataFrame, api_key: str, model: str):
    st.subheader("💬 Ask a Question")
    st.caption("Examples: *\"Top 10 scorers in the Premier League\"* · *\"Compare Salah and Haaland\"* · *\"Best passers in LaLiga\"*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("fig"):
                st.plotly_chart(msg["fig"], use_container_width=True)

    question = st.chat_input("Ask about football stats...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            fig, desc = None, ""
            # Try AI first
            if api_key and OPENAI_AVAILABLE:
                try:
                    spec = query_openai(question, df, api_key, model)
                    fig, desc = execute_spec(spec, df)
                except Exception as e:
                    desc = f"AI query failed ({e}). Falling back to smart parser."
                    fig = None

            # Fallback to smart parser
            if fig is None:
                fig, desc = smart_parse(question, df)

            if desc:
                st.markdown(desc)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            elif not desc:
                desc = "Sorry, I couldn't understand that. Try rephrasing or use the Pre-built Analyses tab."
                st.markdown(desc)

    st.session_state.messages.append({"role": "assistant", "text": desc, "fig": fig})

# ── UI: Pre-built Analyses Tab ───────────────────────────────────────────────

def render_analyses(df: pd.DataFrame):
    analysis = st.selectbox("Choose an analysis", [
        "🏆 Top Scorers",
        "🎯 Top Assist Providers",
        "⚡ Goals + Assists Leaders",
        "🎯 Shot Conversion Rate",
        "📊 Player Radar Comparison",
        "🏟️ Team Aggregated Stats",
        "🛡️ Defensive Leaders",
        "📈 Minutes vs Output Scatter",
        "🌍 Cross-League Comparison",
    ])

    # Common filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_leagues = st.multiselect("League", sorted(df["league_display"].unique()), default=[])
    with c2:
        avail_teams = sorted(df["equipo"].unique()) if not sel_leagues else sorted(
            df[df["league_display"].isin(sel_leagues)]["equipo"].unique())
        sel_teams = st.multiselect("Team", avail_teams, default=[])
    with c3:
        sel_pos = st.multiselect("Position", sorted(df["posicion"].dropna().unique()), default=[])
    with c4:
        top_n = st.slider("Players to show", 5, 40, 15)

    filt = filter_df(df, sel_leagues or None, sel_teams or None, sel_pos or None)

    # Per-90 toggle
    an_stat_mode = st.radio("Stat mode", ["Total", "Per 90 minutes"], horizontal=True, key="an_stat_mode")
    an_per90 = an_stat_mode == "Per 90 minutes"
    if an_per90:
        filt = _to_per90_df(filt)

    _suffix = " (per 90)" if an_per90 else ""

    # ── Top Scorers ──────────────────────────────────────────────────────
    if analysis == "🏆 Top Scorers":
        data = filt[filt["Goals"].notna() & (filt["Goals"] > 0)].nlargest(top_n, "Goals")
        fig = chart_bar(data, "nombre", "Goals", f"Top {top_n} Scorers{_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            data[["nombre", "equipo", "league_display", "posicion", "Goals",
                  "Goal Assists", "Total Shots", "Appearances"]].reset_index(drop=True),
            use_container_width=True,
        )

    # ── Top Assisters ────────────────────────────────────────────────────
    elif analysis == "🎯 Top Assist Providers":
        data = filt[filt["Goal Assists"].notna() & (filt["Goal Assists"] > 0)].nlargest(top_n, "Goal Assists")
        fig = chart_bar(data, "nombre", "Goal Assists", f"Top {top_n} Assist Providers{_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            data[["nombre", "equipo", "league_display", "posicion", "Goal Assists",
                  "Goals", "Key Passes (Attempt Assists)", "Appearances"]].reset_index(drop=True),
            use_container_width=True,
        )

    # ── G+A Leaders ──────────────────────────────────────────────────────
    elif analysis == "⚡ Goals + Assists Leaders":
        tmp = filt.copy()
        tmp["G+A"] = tmp["Goals"].fillna(0) + tmp["Goal Assists"].fillna(0)
        data = tmp[tmp["G+A"] > 0].nlargest(top_n, "G+A")
        fig = px.bar(data, x="nombre", y=["Goals", "Goal Assists"], title=f"Top {top_n} by Goals + Assists{_suffix}",
                     barmode="stack", color_discrete_sequence=CHART_COLORS, height=520)
        fig.update_layout(template="plotly_white", xaxis_tickangle=-45, yaxis_title="Goals + Assists")
        st.plotly_chart(fig, use_container_width=True)

    # ── Shot Conversion ──────────────────────────────────────────────────
    elif analysis == "🎯 Shot Conversion Rate":
        tmp = filt.copy()
        tmp = tmp[(tmp["Total Shots"].notna()) & (tmp["Total Shots"] >= 10)]
        tmp["Conversion %"] = ((tmp["Goals"].fillna(0) / tmp["Total Shots"]) * 100).round(1)
        data = tmp.nlargest(top_n, "Conversion %")
        fig = chart_bar(data, "nombre", "Conversion %",
                        f"Top {top_n} Shot Conversion Rate (min 10 shots){_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            data[["nombre", "equipo", "league_display", "Goals", "Total Shots", "Conversion %"]].reset_index(drop=True),
            use_container_width=True,
        )

    # ── Radar Comparison ─────────────────────────────────────────────────
    elif analysis == "📊 Player Radar Comparison":
        st.info("Select 2-5 players to compare on a radar chart.")
        all_players = sorted(filt[filt["Appearances"].notna() & (filt["Appearances"] > 0)]["nombre"].unique())
        selected = st.multiselect("Select players", all_players, max_selections=5)
        metric_options = [m for m in OFFENSIVE_METRICS + DEFENSIVE_METRICS + PASSING_METRICS if m in filt.columns]
        sel_metrics = st.multiselect("Select metrics (5-8 recommended)", metric_options,
                                     default=metric_options[:6] if len(metric_options) >= 6 else metric_options)
        if len(selected) >= 2 and len(sel_metrics) >= 3:
            fig = chart_radar(filt, selected, sel_metrics, f"Player Radar Comparison{_suffix}")
            st.plotly_chart(fig, use_container_width=True)
            # Side-by-side stats
            compare_df = filt[filt["nombre"].isin(selected)][["nombre", "equipo", "posicion"] + sel_metrics]
            st.dataframe(compare_df.reset_index(drop=True), use_container_width=True)
        elif selected:
            st.warning("Please select at least 2 players and 3 metrics.")

    # ── Team Stats ───────────────────────────────────────────────────────
    elif analysis == "🏟️ Team Aggregated Stats":
        metrics_for_team = ["Goals", "Goal Assists", "Total Shots", "Total Tackles",
                            "Interceptions", "Total Passes", "Yellow Cards"]
        avail = [m for m in metrics_for_team if m in filt.columns]
        team_agg = filt.groupby(["equipo", "league_display"], as_index=False)[avail].sum()
        metric_choice = st.selectbox("Rank teams by", avail, index=0)
        data = team_agg.nlargest(top_n, metric_choice)
        fig = chart_bar(data, "equipo", metric_choice,
                        f"Top {top_n} Teams by {metric_choice}{_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data.reset_index(drop=True), use_container_width=True)

    # ── Defensive Leaders ────────────────────────────────────────────────
    elif analysis == "🛡️ Defensive Leaders":
        def_metric = st.selectbox("Defensive metric", [m for m in DEFENSIVE_METRICS if m in filt.columns])
        data = filt[filt[def_metric].notna() & (filt[def_metric] > 0)].nlargest(top_n, def_metric)
        fig = chart_bar(data, "nombre", def_metric, f"Top {top_n} by {def_metric}{_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            data[["nombre", "equipo", "league_display", "posicion", def_metric, "Appearances"]].reset_index(drop=True),
            use_container_width=True,
        )

    # ── Minutes vs Output ────────────────────────────────────────────────
    elif analysis == "📈 Minutes vs Output Scatter":
        y_metric = st.selectbox("Output metric", ["Goals", "Goal Assists", "Total Shots",
                                                    "Key Passes (Attempt Assists)", "Total Tackles"])
        tmp = filt[(filt["Time Played"].notna()) & (filt["Time Played"] > 90) &
                   (filt[y_metric].notna()) & (filt[y_metric] > 0)]
        fig = chart_scatter(tmp, "Time Played", y_metric,
                            f"Minutes Played vs {y_metric}{_suffix}", color="posicion", size=y_metric)
        st.plotly_chart(fig, use_container_width=True)

    # ── Cross-League Comparison ──────────────────────────────────────────
    elif analysis == "🌍 Cross-League Comparison":
        metric_choice = st.selectbox("Compare leagues by average", [
            "Goals", "Goal Assists", "Total Shots", "Total Passes",
            "Total Tackles", "Interceptions", "Successful Dribbles"])
        # Only players with appearances
        tmp = filt[filt["Appearances"].notna() & (filt["Appearances"] > 0)]
        league_avg = tmp.groupby("league_display", as_index=False)[metric_choice].mean().round(2)
        league_avg = league_avg.sort_values(metric_choice, ascending=False)
        fig = chart_bar(league_avg, "league_display", metric_choice,
                        f"Average {metric_choice} per Player by League{_suffix}", color="league_display")
        st.plotly_chart(fig, use_container_width=True)

# ── UI: Data Explorer Tab ────────────────────────────────────────────────────

# Category weights for player profile ranking
PROFILE_CATEGORIES = {
    "Offensive": ["Goals", "Goal Assists", "Total Shots", "Shots On Target ( inc goals )",
                  "Total Big Chances Scored", "Key Passes (Attempt Assists)", "Goals Openplay"],
    "Defensive": ["Total Tackles", "Tackles Won", "Interceptions", "Total Clearances",
                  "Blocks", "Blocked Shots", "Recoveries"],
    "Passing": ["Total Passes", "Total Successful Passes ( Excl Crosses & Corners )",
                "Successful Long Passes", "Forward Passes", "Through balls"],
    "Possession": ["Successful Dribbles", "Carries", "Progressive Carries",
                   "Duels won", "Ground Duels won"],
    "Aerial": ["Aerial Duels won", "Aerial Duels"],
    "Discipline": ["Yellow Cards", "Total Red Cards", "Total Fouls Conceded"],
}

ROLE_RULES = [
    # (label, position, conditions: list of (metric, threshold) tuples)
    ("Prolific Striker", "Forward", [("Goals", 10)]),
    ("Target Man", "Forward", [("Aerial Duels won", 30), ("Goals", 3)]),
    ("Creative Forward", "Forward", [("Goal Assists", 4), ("Key Passes (Attempt Assists)", 15)]),
    ("Poacher", "Forward", [("Goals from Inside Box", 6)]),
    ("Playmaker", "Midfielder", [("Key Passes (Attempt Assists)", 20)]),
    ("Box-to-Box", "Midfielder", [("Total Tackles", 20), ("Goals", 2)]),
    ("Defensive Midfielder", "Midfielder", [("Total Tackles", 30), ("Interceptions", 15)]),
    ("Creative Midfielder", "Midfielder", [("Goal Assists", 4), ("Through balls", 3)]),
    ("Ball-Winning Midfielder", "Midfielder", [("Recoveries", 50), ("Total Tackles", 25)]),
    ("Ball-Playing CB", "Defender", [("Total Passes", 300), ("Total Clearances", 15)]),
    ("Defensive Rock", "Defender", [("Total Tackles", 25), ("Interceptions", 20)]),
    ("Attacking Full-Back", "Defender", [("Successful Crosses & Corners", 5), ("Goal Assists", 2)]),
    ("Aerial Defender", "Defender", [("Aerial Duels won", 40)]),
    ("Shot-Stopper", "Goalkeeper", [("Saves Made", 20)]),
    ("Sweeper Keeper", "Goalkeeper", [("Total Passes", 150), ("Saves Made", 10)]),
]


def _safe_int(val):
    """Safely convert to int, returning 0 for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _classify_role(row, position):
    """Return the most fitting role label for a player."""
    for label, pos, conditions in ROLE_RULES:
        if pos != position:
            continue
        # Check if all conditions are met
        all_met = True
        for m, t in conditions:
            val = _safe_int(row.get(m, 0))
            if val < t:
                all_met = False
                break
        if all_met:
            return label
    # Fallback
    defaults = {"Forward": "Forward", "Midfielder": "Midfielder",
                "Defender": "Defender", "Goalkeeper": "Goalkeeper"}
    return defaults.get(position, position or "Unknown")


def _compute_percentiles(player_row, df_peers, categories):
    """Compute per-category percentile (0-99) for a player vs peers."""
    result = {}
    for cat, metrics in categories.items():
        avail = [m for m in metrics if m in df_peers.columns]
        if not avail:
            result[cat] = 0
            continue
        player_sum = sum(player_row.get(m, 0) or 0 for m in avail)
        peer_sums = df_peers[avail].fillna(0).sum(axis=1)
        pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100
        # For Discipline, invert — fewer cards = better
        if cat == "Discipline":
            pct = 100 - pct
        result[cat] = round(pct, 1)
    return result


def _compute_metric_percentiles(player_row, df_peers, metrics):
    """Compute individual per-metric percentiles for the pizza chart."""
    result = {}
    for m in metrics:
        if m not in df_peers.columns:
            result[m] = 0
            continue
        val = player_row.get(m, 0)
        val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
        peer_vals = df_peers[m].fillna(0)
        pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
        result[m] = round(pct, 1)
    return result


# Detailed metrics for the pizza chart, grouped by category
PIZZA_METRICS = {
    "Defensive": [
        ("Tackles Won", "Tackles Won"),
        ("Interceptions", "Interceptions"),
        ("Recoveries", "Recoveries"),
        ("Clearances", "Total Clearances"),
        ("Blocks", "Blocks"),
    ],
    "Aerial": [
        ("Aerial Duels", "Aerial Duels"),
        ("Aerial Won", "Aerial Duels won"),
    ],
    "Passing": [
        ("Total Passes", "Total Passes"),
        ("Long Passes", "Successful Long Passes"),
        ("Forward Passes", "Forward Passes"),
        ("Through Balls", "Through balls"),
        ("Key Passes", "Key Passes (Attempt Assists)"),
    ],
    "Possession": [
        ("Dribbles", "Successful Dribbles"),
        ("Carries", "Carries"),
        ("Progressive Carries", "Progressive Carries"),
        ("Duels Won", "Duels won"),
    ],
    "Attacking": [
        ("Goals", "Goals"),
        ("Assists", "Goal Assists"),
        ("Shots on Target", "Shots On Target ( inc goals )"),
        ("Big Chances", "Total Big Chances Scored"),
    ],
}

PIZZA_CATEGORY_COLORS = {
    "Defensive": "#457b9d",
    "Aerial": "#f4a261",
    "Passing": "#2a9d8f",
    "Possession": "#e9c46a",
    "Attacking": "#e63946",
}


def _build_pizza_chart(player_row, df_peers, player_name, position):
    """Build a nightingale rose (pizza) chart showing per-metric percentiles."""
    labels = []
    values = []
    colors = []
    category_labels = []

    for cat, metric_list in PIZZA_METRICS.items():
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

    # Add wedges grouped by category
    cats_seen = set()
    for i, (lbl, val, clr, cat) in enumerate(zip(labels, values, colors, category_labels)):
        show_legend = cat not in cats_seen
        cats_seen.add(cat)
        fig.add_trace(go.Barpolar(
            r=[val],
            theta=[theta_centers[i]],
            width=[slice_angle - 1],
            marker=dict(
                color=clr, opacity=0.85,
                line=dict(color="#1a1a2e", width=1.5),
            ),
            name=cat if show_legend else None,
            legendgroup=cat,
            showlegend=show_legend,
            hovertemplate=f"<b>{lbl}</b><br>{cat}<br>Percentile: {val:.0f}<extra></extra>",
        ))

    # Add percentile numbers as a Scatterpolar text trace (polar coords = perfect alignment)
    text_r = [max(v, 10) + 10 for v in values]
    fig.add_trace(go.Scatterpolar(
        r=text_r,
        theta=theta_centers,
        mode="text",
        text=[f"<b>{v:.0f}</b>" for v in values],
        textfont=dict(size=11, color="white"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 115],
                showticklabels=False, showline=False,
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickvals=theta_centers,
                ticktext=labels,
                tickfont=dict(size=9, color="#ccc"),
                gridcolor="rgba(255,255,255,0.05)",
                direction="clockwise",
            ),
            bgcolor="#1a1a2e",
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        title=dict(
            text=f"<b>{player_name}</b> vs {position}s<br>"
                 f"<span style='font-size:12px;color:#aaa'>Percentile rankings · 0-100 scale · 50 = average</span>",
            font=dict(size=16, color="#f4a261"),
        ),
        height=650,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
            font=dict(size=12, color="#eee"),
        ),
        margin=dict(t=80, b=60, l=80, r=80),
    )
    return fig


# Scatter plot metric presets by position
SCATTER_METRIC_OPTIONS = [
    "Goals", "Goal Assists", "Total Shots", "Shots On Target ( inc goals )",
    "Total Big Chances Scored", "Key Passes (Attempt Assists)",
    "Total Passes", "Successful Long Passes", "Forward Passes", "Through balls",
    "Progressive Carries", "Successful Dribbles", "Carries",
    "Total Tackles", "Tackles Won", "Interceptions", "Total Clearances",
    "Blocks", "Recoveries", "Aerial Duels won", "Ground Duels won",
    "Duels won", "Aerial Duels", "Appearances", "Time Played",
    "Saves Made", "Yellow Cards",
    # Per 90 variants
    "Goals per 90", "Goal Assists per 90", "Total Shots per 90",
    "Key Passes (Attempt Assists) per 90", "Successful Dribbles per 90",
    "Total Tackles per 90", "Interceptions per 90",
]

SCATTER_DEFAULTS = {
    "Forward":     ("Total Shots", "Goals"),
    "Midfielder":  ("Total Passes", "Total Tackles"),
    "Defender":    ("Total Tackles", "Aerial Duels won"),
    "Goalkeeper":  ("Saves Made", "Total Passes"),
}


def _build_scatter_plot(player_row, df_peers, player_name, position, x_col, y_col):
    """Build a scatter plot with the player highlighted among same-position peers."""
    if x_col not in df_peers.columns or y_col not in df_peers.columns:
        return None

    pos_peers = df_peers[df_peers["posicion"] == position].copy()
    if pos_peers.empty:
        return None

    pos_peers = pos_peers[(pos_peers[x_col].notna()) & (pos_peers[y_col].notna())].copy()
    if pos_peers.empty:
        return None

    pos_peers["is_selected"] = pos_peers["nombre"] == player_name

    # Detect outliers using IQR on both axes
    def _iqr_outliers(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return series > (q3 + 1.5 * iqr)

    pos_peers["_outlier"] = _iqr_outliers(pos_peers[x_col]) | _iqr_outliers(pos_peers[y_col])

    fig = go.Figure()

    # Regular players (gray, no text)
    others = pos_peers[~pos_peers["is_selected"] & ~pos_peers["_outlier"]]
    fig.add_trace(go.Scatter(
        x=others[x_col], y=others[y_col],
        mode="markers",
        marker=dict(size=7, color="#555", opacity=0.45),
        text=others["nombre"] + " (" + others["equipo"] + ")",
        hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
        name=f"Other {position}s",
        showlegend=True,
    ))

    # Outlier players (labeled)
    outliers = pos_peers[~pos_peers["is_selected"] & pos_peers["_outlier"]]
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers[x_col], y=outliers[y_col],
            mode="markers+text",
            marker=dict(size=9, color="#e9c46a", opacity=0.8,
                        line=dict(width=1, color="#f4a261")),
            text=outliers["nombre"],
            textposition="top center",
            textfont=dict(size=9, color="#e9c46a"),
            hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
            name="Outliers",
            showlegend=True,
        ))

    # Selected player (highlighted green)
    selected = pos_peers[pos_peers["is_selected"]]
    if not selected.empty:
        fig.add_trace(go.Scatter(
            x=selected[x_col], y=selected[y_col],
            mode="markers+text",
            marker=dict(size=16, color="#2d6a4f", symbol="circle",
                        line=dict(width=2.5, color="white")),
            text=selected["nombre"],
            textposition="top center",
            textfont=dict(size=13, color="#52b788", family="Arial Black"),
            hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<extra></extra>",
            name=player_name,
            showlegend=True,
        ))

    # Average lines to create quadrants
    avg_x = pos_peers[x_col].mean()
    avg_y = pos_peers[y_col].mean()
    fig.add_vline(x=avg_x, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text=f"Avg {x_col}", annotation_font_color="#888",
                  annotation_font_size=10)
    fig.add_hline(y=avg_y, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text=f"Avg {y_col}", annotation_font_color="#888",
                  annotation_font_size=10)

    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b> vs All {position}s Across Europe",
            font=dict(size=16, color="#f4a261"),
        ),
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=560,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        legend=dict(font=dict(color="#eee")),
    )
    return fig


# ── Photo helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_team_logo(team_name: str) -> str | None:
    """Try to fetch a team badge URL from TheSportsDB (free, no key required)."""
    try:
        q = urllib.parse.quote(team_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        teams = data.get("teams")
        if teams:
            return teams[0].get("strBadge") or teams[0].get("strTeamBadge")
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_player_photo(player_name: str) -> str | None:
    """Try to fetch a player photo URL from TheSportsDB."""
    try:
        q = urllib.parse.quote(player_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        players = data.get("player")
        if players:
            return players[0].get("strCutout") or players[0].get("strThumb") or players[0].get("strRender")
    except Exception:
        pass
    return None


def render_profile(df: pd.DataFrame):
    st.subheader("🪪 Player Profile")

    # Selectors
    c1, c2, c3 = st.columns(3)
    with c1:
        league_sel = st.selectbox("League", ["All"] + sorted(df["league_display"].unique()), key="prof_lg")
    with c2:
        pool = df if league_sel == "All" else df[df["league_display"] == league_sel]
        team_sel = st.selectbox("Team", ["All"] + sorted(pool["equipo"].unique()), key="prof_tm")
    with c3:
        pool2 = pool if team_sel == "All" else pool[pool["equipo"] == team_sel]
        active = pool2[pool2["Appearances"].notna() & (pool2["Appearances"] > 0)]
        player_names = sorted(active["nombre"].unique())
        player_sel = st.selectbox("Player", player_names, key="prof_pl")

    if not player_sel:
        return

    row = active[active["nombre"] == player_sel].iloc[0]
    position = row.get("posicion", "Unknown")
    team = row.get("equipo", "")
    league = row.get("league_display", "")
    role = _classify_role(row, position)

    # ── Header Card with photos ────────────────────────────────────────
    st.markdown("---")
    photo_col, h1, h2, h3, h4 = st.columns([0.8, 2, 1, 1, 1])
    with photo_col:
        player_photo = _fetch_player_photo(row["nombre"])
        if player_photo:
            st.image(player_photo, width=110)
        else:
            st.markdown("<div style='width:100px;height:100px;border-radius:50%;background:#2d6a4f;"
                        "display:flex;align-items:center;justify-content:center;font-size:36px;color:white;'>"
                        f"{row['nombre'][0]}</div>", unsafe_allow_html=True)
        team_logo = _fetch_team_logo(team)
        if team_logo:
            st.image(team_logo, width=50)
    with h1:
        st.markdown(f"## {row['nombre']}")
        st.markdown(f"**{team}** · {league}")
        st.markdown(f"**Position:** {position} · **Role:** {role}")
        if pd.notna(row.get("dorsal")) and not (isinstance(row.get("dorsal"), float) and np.isnan(row.get("dorsal"))):
            st.markdown(f"**Squad Number:** {_safe_int(row['dorsal'])}")
    with h2:
        st.metric("Appearances", _safe_int(row.get("Appearances")))
        st.metric("Minutes", f"{_safe_int(row.get('Time Played')):,}")
    with h3:
        st.metric("Goals", _safe_int(row.get("Goals")))
        st.metric("Assists", _safe_int(row.get("Goal Assists")))
    with h4:
        st.metric("Shots", _safe_int(row.get("Total Shots")))
        st.metric("Tackles", _safe_int(row.get("Total Tackles")))

    # ── Transfermarkt Link ───────────────────────────────────────────────
    search_name = row["nombre"].replace(" ", "+")
    tm_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={search_name}"
    st.markdown(f"🔗 [Look up on Transfermarkt]({tm_url})")

    # ── Stat Mode Toggle (applies to all charts below) ───────────────────
    stat_mode = st.radio("Stat mode", ["Total", "Per 90 minutes"], horizontal=True, key="prof_stat_mode")
    use_per90 = stat_mode == "Per 90 minutes"
    minutes_played = row.get("Time Played", 0) or 0

    # Build peers dataframe and player row in the chosen mode
    peers = df[(df["Time Played"].notna()) & (df["Time Played"] >= 90)]
    if use_per90:
        peers = _to_per90_df(peers)
        row_data = _row_to_per90(row, minutes_played)
    else:
        row_data = dict(row)
    mode_label = "/90" if use_per90 else ""

    # ── Pizza Chart (Percentile Rose) ───────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🍕 Percentile Pizza Chart{' (Per 90)' if use_per90 else ''} — vs players with 90+ minutes")

    fig_pizza = _build_pizza_chart(row_data, peers, row["nombre"], position)
    if fig_pizza:
        st.plotly_chart(fig_pizza, use_container_width=True)
    else:
        st.info("Not enough data to build the pizza chart for this player.")

    # ── FBref-style Scouting Report ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 Scouting Report (FBref-style){' — Per 90' if use_per90 else ''}")
    pos_peers_fbref = peers[peers["posicion"] == position]
    fig_fbref = _build_fbref_bar_chart(row_data, pos_peers_fbref, row["nombre"], position)
    if fig_fbref:
        st.plotly_chart(fig_fbref, use_container_width=True)
    else:
        st.info("Not enough data for the scouting report.")

    # ── Scatter Plot (Peer Comparison) ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Peer Scatter Plot — Where Does the Player Stand?")
    st.caption("Choose which metrics to compare on each axis.")
    avail_metrics = [m for m in SCATTER_METRIC_OPTIONS if m in df.columns]
    defaults = SCATTER_DEFAULTS.get(position, ("Total Passes", "Goals"))
    default_x = defaults[0] if defaults[0] in avail_metrics else avail_metrics[0]
    default_y = defaults[1] if defaults[1] in avail_metrics else avail_metrics[1]
    sc1, sc2 = st.columns(2)
    with sc1:
        x_metric = st.selectbox("X-Axis Metric", avail_metrics,
                                index=avail_metrics.index(default_x), key="scatter_x")
    with sc2:
        y_metric = st.selectbox("Y-Axis Metric", avail_metrics,
                                index=avail_metrics.index(default_y), key="scatter_y")
    fig_scatter = _build_scatter_plot(row_data, peers, row["nombre"], position, x_metric, y_metric)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough data to build the scatter plot for this player.")

    # ── FBref-style Positional Percentile Averages ────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 FBref-Style Percentile Summary — {position}s{' (Per 90)' if use_per90 else ''}")
    st.caption(f"Average percentile per category, computed vs all {position}s with 90+ mins across Europe.")
    pos_peers = peers[peers["posicion"] == position]
    pcts = _compute_percentiles(row_data, pos_peers, PROFILE_CATEGORIES)

    # Overall average percentile
    avg_pct = round(sum(pcts.values()) / max(len(pcts), 1), 1)
    st.markdown(f"**Overall Percentile Average: `{avg_pct:.0f}`**")

    bar_cols = st.columns(len(pcts))
    colors = {"Offensive": "#e63946", "Defensive": "#457b9d", "Passing": "#2a9d8f",
              "Possession": "#e9c46a", "Aerial": "#f4a261", "Discipline": "#264653"}
    for col_ui, (cat, pct) in zip(bar_cols, pcts.items()):
        with col_ui:
            color = colors.get(cat, "#2d6a4f")
            st.markdown(f"**{cat}**")
            st.progress(min(int(pct), 100))
            st.caption(f"{pct:.0f}th %ile vs {position}s")

    # ── Detailed Stats ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Detailed Statistics{' (Per 90)' if use_per90 else ''}")

    def _per90_val(val):
        if not use_per90 or minutes_played <= 0:
            return round(val, 2) if isinstance(val, float) else val
        return round((val / minutes_played) * 90, 2)

    t1, t2, t3 = st.tabs(["⚔️ Offensive", "🛡️ Defensive", "📊 Passing & Possession"])
    with t1:
        off_metrics = [m for m in OFFENSIVE_METRICS if m in df.columns]
        off_data = {(f"{m} /90" if use_per90 else m): _per90_val(row.get(m, 0) if not pd.isna(row.get(m, 0)) else 0) for m in off_metrics}
        st.dataframe(pd.DataFrame([off_data]).T.rename(columns={0: "Value"}), use_container_width=True)
    with t2:
        def_metrics = [m for m in DEFENSIVE_METRICS if m in df.columns]
        def_data = {(f"{m} /90" if use_per90 else m): _per90_val(row.get(m, 0) if not pd.isna(row.get(m, 0)) else 0) for m in def_metrics}
        st.dataframe(pd.DataFrame([def_data]).T.rename(columns={0: "Value"}), use_container_width=True)
    with t3:
        pass_metrics = [m for m in PASSING_METRICS if m in df.columns]
        pass_data = {(f"{m} /90" if use_per90 else m): _per90_val(row.get(m, 0) if not pd.isna(row.get(m, 0)) else 0) for m in pass_metrics}
        st.dataframe(pd.DataFrame([pass_data]).T.rename(columns={0: "Value"}), use_container_width=True)

# ── FBref-style position percentile templates ────────────────────────────────

FBREF_TEMPLATES = {
    "Forward": {
        "Non-Penalty Goals": ["Goals"],
        "Shots Total": ["Total Shots"],
        "Assists": ["Goal Assists"],
        "Shot-Creating Actions": ["Key Passes (Attempt Assists)", "Through balls"],
        "Passes Attempted": ["Total Passes"],
        "Progressive Passes": ["Forward Passes"],
        "Dribbles Completed": ["Successful Dribbles"],
        "Carries": ["Carries", "Progressive Carries"],
        "Aerial Duels Won": ["Aerial Duels won"],
        "Tackles": ["Total Tackles"],
        "Interceptions": ["Interceptions"],
    },
    "Midfielder": {
        "Goals": ["Goals"],
        "Assists": ["Goal Assists"],
        "Shot-Creating Actions": ["Key Passes (Attempt Assists)", "Through balls"],
        "Passes Attempted": ["Total Passes"],
        "Pass Completion": ["Total Successful Passes ( Excl Crosses & Corners )"],
        "Progressive Passes": ["Forward Passes", "Successful Long Passes"],
        "Dribbles Completed": ["Successful Dribbles"],
        "Progressive Carries": ["Progressive Carries"],
        "Tackles": ["Total Tackles", "Tackles Won"],
        "Interceptions": ["Interceptions"],
        "Blocks": ["Blocks"],
        "Aerial Duels Won": ["Aerial Duels won"],
    },
    "Defender": {
        "Goals": ["Goals"],
        "Assists": ["Goal Assists"],
        "Passes Attempted": ["Total Passes"],
        "Pass Completion": ["Total Successful Passes ( Excl Crosses & Corners )"],
        "Progressive Passes": ["Forward Passes", "Successful Long Passes"],
        "Carries": ["Carries", "Progressive Carries"],
        "Tackles": ["Total Tackles", "Tackles Won"],
        "Interceptions": ["Interceptions"],
        "Blocks": ["Blocks", "Blocked Shots"],
        "Clearances": ["Total Clearances"],
        "Aerial Duels Won": ["Aerial Duels won"],
    },
    "Goalkeeper": {
        "Saves": ["Saves Made"],
        "Pass Completion": ["Total Successful Passes ( Excl Crosses & Corners )"],
        "Passes Attempted": ["Total Passes"],
        "Long Passes": ["Successful Long Passes"],
        "Aerial Duels Won": ["Aerial Duels won"],
    },
}


def _build_fbref_bar_chart(player_row, df_peers, player_name, position):
    """Build a horizontal bar chart resembling FBref's per-position percentile scouting report."""
    template = FBREF_TEMPLATES.get(position, FBREF_TEMPLATES["Midfielder"])
    labels = []
    pct_values = []
    bar_colors = []

    for display_name, cols in template.items():
        avail = [c for c in cols if c in df_peers.columns]
        if not avail:
            continue
        player_val = sum(player_row.get(c, 0) or 0 for c in avail)
        peer_vals = df_peers[avail].fillna(0).sum(axis=1)
        pct = (peer_vals < player_val).sum() / max(len(peer_vals), 1) * 100
        pct = round(pct, 1)
        labels.append(display_name)
        pct_values.append(pct)
        # Color: green if >= 66, amber if >= 33, red if < 33
        if pct >= 66:
            bar_colors.append("#2d6a4f")
        elif pct >= 33:
            bar_colors.append("#e9c46a")
        else:
            bar_colors.append("#e63946")

    if not labels:
        return None

    # Reverse for top-down display
    labels.reverse()
    pct_values.reverse()
    bar_colors.reverse()

    fig = go.Figure(go.Bar(
        y=labels, x=pct_values,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#1a1a2e", width=1)),
        text=[f"{v:.0f}" for v in pct_values],
        textposition="outside",
        textfont=dict(color="#eee", size=12),
        hovertemplate="<b>%{y}</b><br>Percentile: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b> — Scouting Report vs {position}s<br>"
                 f"<span style='font-size:12px;color:#aaa'>Percentile · 0-100 · compared to same position across Europe</span>",
            font=dict(size=15, color="#f4a261"),
        ),
        xaxis=dict(range=[0, 110], title="Percentile", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(tickfont=dict(size=12, color="#ccc")),
        height=max(380, 36 * len(labels)),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        margin=dict(l=160, r=40, t=80, b=40),
    )
    return fig


# ── Team Profile ─────────────────────────────────────────────────────────────

def render_team_profile(df: pd.DataFrame):
    st.subheader("🏟️ Team Profile")

    c1, c2 = st.columns(2)
    with c1:
        league_sel = st.selectbox("League", ["All"] + sorted(df["league_display"].unique()), key="tp_lg")
    with c2:
        pool = df if league_sel == "All" else df[df["league_display"] == league_sel]
        team_names = sorted(pool["equipo"].unique())
        team_sel = st.selectbox("Team", team_names, key="tp_tm")

    if not team_sel:
        return

    team_df = df[(df["equipo"] == team_sel) & (df["Appearances"].notna()) & (df["Appearances"] > 0)]
    league_name = team_df["league_display"].iloc[0] if not team_df.empty else ""
    league_df = df[(df["league_display"] == league_name) & (df["Time Played"].notna()) & (df["Time Played"] > 90)]

    if team_df.empty:
        st.warning("No active player data for this team.")
        return

    # ── Header with team logo ────────────────────────────────────────────
    st.markdown("---")
    logo_col, info_col, m1, m2, m3 = st.columns([0.6, 2, 1, 1, 1])
    with logo_col:
        logo = _fetch_team_logo(team_sel)
        if logo:
            st.image(logo, width=90)
    with info_col:
        st.markdown(f"## {team_sel}")
        st.markdown(f"**{league_name}**")
        st.markdown(f"**Squad size:** {len(team_df)} active players")
    with m1:
        st.metric("Total Goals", _safe_int(team_df["Goals"].sum()) if "Goals" in team_df.columns else 0)
        st.metric("Total Assists", _safe_int(team_df["Goal Assists"].sum()) if "Goal Assists" in team_df.columns else 0)
    with m2:
        st.metric("Total Shots", _safe_int(team_df["Total Shots"].sum()) if "Total Shots" in team_df.columns else 0)
        st.metric("Total Tackles", _safe_int(team_df["Total Tackles"].sum()) if "Total Tackles" in team_df.columns else 0)
    with m3:
        st.metric("Total Passes", f"{_safe_int(team_df['Total Passes'].sum()):,}" if "Total Passes" in team_df.columns else "0")
        st.metric("Avg Appearances", f"{team_df['Appearances'].mean():.1f}" if "Appearances" in team_df.columns else "0")

    # Per-90 toggle for the whole team profile
    tp_stat_mode = st.radio("Stat mode", ["Total", "Per 90 minutes"], horizontal=True, key="tp_stat_mode")
    tp_per90 = tp_stat_mode == "Per 90 minutes"
    if tp_per90:
        team_df = _to_per90_df(team_df)
        league_df = _to_per90_df(league_df)
    _tp_suffix = " (per 90)" if tp_per90 else ""

    # ── Team Radar vs League Average ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 Team Profile vs League Average")

    league_teams = league_df.groupby("equipo")
    radar_cats = []
    team_pcts = []
    for cat, info in TEAM_ANALYSIS_CATEGORIES.items():
        avail = [m for m in info["metrics"] if m in league_df.columns]
        if not avail:
            continue
        team_total = team_df[avail].sum().sum()
        all_totals = league_teams.apply(lambda g, a=avail: g[a].sum().sum(), include_groups=False)
        pct = (all_totals < team_total).sum() / max(len(all_totals), 1) * 100
        radar_cats.append(cat)
        team_pcts.append(round(pct, 1))

    if radar_cats:
        cats_c = radar_cats + [radar_cats[0]]
        vals_c = team_pcts + [team_pcts[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself", name=team_sel,
            line=dict(color="#e63946", width=2), fillcolor="rgba(230,57,70,0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[50] * len(cats_c), theta=cats_c,
            name="League Average", line=dict(color="gray", dash="dash", width=1), fill=None,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{team_sel} — Category Percentiles in {league_name}{_tp_suffix}",
            height=500, template="plotly_white", showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Top Performers ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⭐ Key Players")

    def _player_with_photo(name):
        """Return an HTML snippet with player photo or fallback initial."""
        photo = _fetch_player_photo(name)
        if photo:
            return f'<img src="{photo}" width="28" style="border-radius:50%;vertical-align:middle;"> {name}'
        return f'<span style="display:inline-block;width:28px;height:28px;border-radius:50%;background:#2d6a4f;color:white;text-align:center;line-height:28px;font-size:14px;vertical-align:middle;">{name[0]}</span> {name}'

    kp1, kp2, kp3 = st.columns(3)
    with kp1:
        st.markdown("**Top Scorers**")
        if "Goals" in team_df.columns:
            scorers = team_df.nlargest(5, "Goals")[["nombre", "posicion", "Goals"]].reset_index(drop=True)
            if not scorers.empty:
                st.dataframe(scorers, use_container_width=True, hide_index=True)
    with kp2:
        st.markdown("**Top Assisters**")
        if "Goal Assists" in team_df.columns:
            assisters = team_df.nlargest(5, "Goal Assists")[["nombre", "posicion", "Goal Assists"]].reset_index(drop=True)
            if not assisters.empty:
                st.dataframe(assisters, use_container_width=True, hide_index=True)
    with kp3:
        st.markdown("**Top Tacklers**")
        if "Total Tackles" in team_df.columns:
            tacklers = team_df.nlargest(5, "Total Tackles")[["nombre", "posicion", "Total Tackles"]].reset_index(drop=True)
            if not tacklers.empty:
                st.dataframe(tacklers, use_container_width=True, hide_index=True)

    # ── Position Breakdown ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Squad Composition")
    pos_counts = team_df["posicion"].value_counts().reset_index()
    pos_counts.columns = ["Position", "Count"]
    p1, p2 = st.columns(2)
    with p1:
        fig_pie = px.pie(pos_counts, names="Position", values="Count",
                         title="Position Distribution", color_discrete_sequence=CHART_COLORS,
                         hole=0.35)
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)
    with p2:
        # Age / experience proxy: minutes distribution
        if "Time Played" in team_df.columns:
            fig_mins = px.bar(team_df.nlargest(15, "Time Played"),
                              x="nombre", y="Time Played", color="posicion",
                              title="Minutes Distribution (Top 15)",
                              color_discrete_sequence=CHART_COLORS, height=380)
            fig_mins.update_layout(template="plotly_white", xaxis_tickangle=-45,
                                   yaxis_title="Minutes Played")
            st.plotly_chart(fig_mins, use_container_width=True)

    # ── Strengths & Weaknesses ───────────────────────────────────────────
    if radar_cats:
        st.markdown("---")
        st.markdown("### 🔍 Strengths & Weaknesses")
        sorted_areas = sorted(zip(radar_cats, team_pcts), key=lambda x: x[1])
        strengths = [(c, p) for c, p in sorted_areas if p >= 60]
        weaknesses = [(c, p) for c, p in sorted_areas if p < 40]
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**✅ Strengths**")
            if strengths:
                for cat, pct in reversed(strengths):
                    st.markdown(f"- **{cat}** — {pct:.0f}th percentile")
            else:
                st.caption("No standout strengths identified.")
        with s2:
            st.markdown("**⚠️ Areas to Improve**")
            if weaknesses:
                for cat, pct in weaknesses:
                    desc = TEAM_ANALYSIS_CATEGORIES.get(cat, {}).get("description", "")
                    st.markdown(f"- **{cat}** ({desc}) — {pct:.0f}th percentile")
            else:
                st.caption("No major weaknesses — well-balanced squad.")

    # ── Full Squad Table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Squad")
    stat_cols = ["Goals", "Goal Assists", "Total Shots", "Total Tackles", "Total Passes"]
    squad_base = ["nombre", "posicion", "dorsal", "Appearances", "Time Played"]
    show = [c for c in squad_base if c in team_df.columns]
    show += [c for c in stat_cols if c in team_df.columns]
    squad_display = team_df[show].sort_values("Time Played", ascending=False).reset_index(drop=True)
    st.dataframe(squad_display, use_container_width=True, height=400)


def render_explorer(df: pd.DataFrame):
    st.subheader("🔍 Data Explorer")
    c1, c2, c3 = st.columns(3)
    with c1:
        leagues = st.multiselect("Filter league", sorted(df["league_display"].unique()), key="exp_league")
    with c2:
        avail = sorted(df["equipo"].unique()) if not leagues else sorted(
            df[df["league_display"].isin(leagues)]["equipo"].unique())
        teams = st.multiselect("Filter team", avail, key="exp_team")
    with c3:
        search = st.text_input("Search player name", key="exp_search")

    filtered = df.copy()
    if leagues:
        filtered = filtered[filtered["league_display"].isin(leagues)]
    if teams:
        filtered = filtered[filtered["equipo"].isin(teams)]
    if search:
        filtered = filtered[filtered["nombre"].str.contains(search, case=False, na=False)]

    # Column selector
    default_cols = ["nombre", "equipo", "league_display", "posicion", "Appearances",
                    "Goals", "Goal Assists", "Total Shots", "Total Passes", "Total Tackles", "Time Played"]
    avail_cols = [c for c in default_cols if c in filtered.columns]
    all_cols = list(filtered.columns)
    show_cols = st.multiselect("Columns to display", all_cols, default=avail_cols, key="exp_cols")

    # Per-90 toggle
    exp_stat_mode = st.radio("Stat mode", ["Total", "Per 90 minutes"], horizontal=True, key="exp_stat_mode")
    if exp_stat_mode == "Per 90 minutes":
        filtered = _to_per90_df(filtered)

    if show_cols:
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True, height=500)
    st.caption(f"Showing {len(filtered):,} players")

    # Download
    csv_data = filtered[show_cols].to_csv(index=False).encode("utf-8") if show_cols else b""
    st.download_button("📥 Download filtered data as CSV", csv_data, "football_data.csv", "text/csv")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🕵️ FORENSICS XG: INTELLIGENCE UNIT")
    st.markdown("#### WHERE THE BEAUTIFUL GAME MEETS HARD EVIDENCE.")

    # Load data
    df = load_data()
    if df.empty:
        st.error("No data found. Make sure the league CSV folders are in the parent directory.")
        return

    df = add_per90(df)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = ""
        model = "gpt-4o-mini"
        if OPENAI_AVAILABLE:
            api_key = st.text_input("OpenAI API Key", type="password",
                                    help="Optional — enables AI-powered natural language queries")
            model = st.selectbox("AI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        else:
            st.info("Install `openai` package to enable AI chat. The smart parser and pre-built analyses work without it.")

        st.divider()
        st.header("📊 Data Overview")

        total_players = len(df[df["Appearances"].notna() & (df["Appearances"] > 0)])
        st.metric("Active Players", f"{total_players:,}")
        st.metric("Leagues", df["league_display"].nunique())
        st.metric("Teams", df["equipo"].nunique())

        if "Goals" in df.columns:
            top_idx = df["Goals"].idxmax()
            if pd.notna(top_idx):
                top = df.loc[top_idx]
                st.metric("Top Scorer", f"{top['nombre']}", delta=f"{_safe_int(top['Goals'])} goals")
        if "Goal Assists" in df.columns:
            top_idx = df["Goal Assists"].idxmax()
            if pd.notna(top_idx):
                top = df.loc[top_idx]
                st.metric("Top Assister", f"{top['nombre']}", delta=f"{_safe_int(top['Goal Assists'])} assists")

    # Tabs
    tab_chat, tab_analysis, tab_profile, tab_team, tab_explorer = st.tabs([
        "💬 Ask a Question", "📊 Pre-built Analyses", "🪪 Player Profile",
        "🏟️ Team Profile", "🔍 Data Explorer"
    ])

    with tab_chat:
        render_chat(df, api_key, model)
    with tab_analysis:
        render_analyses(df)
    with tab_profile:
        render_profile(df)
    with tab_team:
        render_team_profile(df)
    with tab_explorer:
        render_explorer(df)


if __name__ == "__main__":
    main()
    
