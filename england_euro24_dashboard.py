import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ast
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="England Euro 2024 – Offensive Analysis", layout="wide")

DATA_DIR = r"c:\Users\kenzi\Downloads\AnalisisDeJuego_Europea_Marzo26"
EVENTS_DIR = os.path.join(DATA_DIR, "Events")
ENGLAND = "England"
FINAL_THIRD_X = 80.0

# ── COLOUR PALETTE ──────────────────────────────────────────────────────────
ENG_RED   = "#CF081F"
ENG_BLUE  = "#1B3D6D"
ENG_WHITE = "#FFFFFF"
PITCH_BG  = "#1a1a2e"
LINE_COL  = "#c0c0c0"

# ── PLOTLY PITCH HELPERS ────────────────────────────────────────────────────

def _full_pitch_shapes(highlight_final_third=False, ft_color=ENG_RED):
    """Return Plotly shapes list for a full StatsBomb pitch (120 x 80)."""
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=LINE_COL, width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=LINE_COL, width=1.5)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50,
             line=dict(color=LINE_COL, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=LINE_COL, width=1.5)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=LINE_COL, width=1.5)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=LINE_COL, width=1.5)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=LINE_COL, width=1.5)),
        dict(type="rect", x0=-2, y0=36, x1=0, y1=44,
             fillcolor="white", line=dict(color="white", width=1)),
        dict(type="rect", x0=120, y0=36, x1=122, y1=44,
             fillcolor="white", line=dict(color="white", width=1)),
        dict(type="circle", x0=11, y0=39, x1=13, y1=41,
             fillcolor=LINE_COL, line=dict(color=LINE_COL, width=0)),
        dict(type="circle", x0=107, y0=39, x1=109, y1=41,
             fillcolor=LINE_COL, line=dict(color=LINE_COL, width=0)),
    ]
    if highlight_final_third:
        shapes.insert(0, dict(
            type="rect", x0=FINAL_THIRD_X, y0=0, x1=120, y1=80,
            fillcolor=ft_color, opacity=0.08, line=dict(width=0),
        ))
    return shapes


def _full_pitch_layout(title="", height=550, shapes=None):
    return dict(
        shapes=shapes or [],
        xaxis=dict(range=[-4, 124], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-4, 84], showgrid=False, zeroline=False, visible=False,
                   autorange="reversed"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=height,
        title=dict(text=title, font=dict(color="white", size=15)),
        legend=dict(font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
    )


def _half_pitch_shapes():
    return [
        dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color=LINE_COL, width=2)),
        dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color=LINE_COL, width=1.5)),
        dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color=LINE_COL, width=1.5)),
        dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color=LINE_COL, width=1.5)),
        dict(type="circle", x0=39, y0=107, x1=41, y1=109,
             fillcolor=LINE_COL, line=dict(color=LINE_COL, width=0)),
        dict(type="rect", x0=36, y0=120, x1=44, y1=122,
             fillcolor="white", line=dict(color="white", width=1)),
        dict(type="circle", x0=28, y0=96, x1=52, y1=120,
             line=dict(color=LINE_COL, width=1.5, dash="dot"),
             fillcolor="rgba(0,0,0,0)"),
    ]


def _half_pitch_layout(title="", height=650, shapes=None):
    return dict(
        shapes=shapes or _half_pitch_shapes(),
        xaxis=dict(range=[-2, 82], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[58, 124], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=height,
        title=dict(text=title, font=dict(color="white", size=15)),
        legend=dict(font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
    )

PLOTLY_CFG = {"scrollZoom": True}

# ── HELPERS ─────────────────────────────────────────────────────────────────

def safe_loc(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return None
    if isinstance(val, (list, np.ndarray)):
        return val
    return None


@st.cache_data(show_spinner="Loading data …")
def load_data():
    matches = pd.read_excel(os.path.join(DATA_DIR, "SB_Euro24_matchdetails.xlsx"))
    passes  = pd.read_excel(os.path.join(EVENTS_DIR, "SB_Euro24_Events_passes.xlsx"))
    carries = pd.read_excel(os.path.join(EVENTS_DIR, "SB_Euro24_Events_carrys.xlsx"))
    shots   = pd.read_excel(os.path.join(EVENTS_DIR, "SB_Euro24_Events_shots.xlsx"))

    eng_matches = matches[
        (matches["home_team"] == ENGLAND) | (matches["away_team"] == ENGLAND)
    ].copy()
    eng_match_ids = eng_matches["match_id"].tolist()

    passes  = passes[(passes["team"] == ENGLAND) & (passes["match_id"].isin(eng_match_ids))].copy()
    carries = carries[(carries["team"] == ENGLAND) & (carries["match_id"].isin(eng_match_ids))].copy()
    shots   = shots[(shots["team"] == ENGLAND) & (shots["match_id"].isin(eng_match_ids))].copy()

    for df, cols in [
        (passes,  ["location", "pass_end_location"]),
        (carries, ["location", "carry_end_location"]),
        (shots,   ["location", "shot_end_location"]),
    ]:
        for c in cols:
            parsed = df[c].apply(safe_loc)
            df[f"{c}_x"] = parsed.apply(lambda v: v[0] if v and len(v) >= 2 else np.nan)
            df[f"{c}_y"] = parsed.apply(lambda v: v[1] if v and len(v) >= 2 else np.nan)
            if c == "shot_end_location":
                df[f"{c}_z"] = parsed.apply(lambda v: v[2] if v and len(v) >= 3 else np.nan)

    def opponent_label(row, md):
        m = md[md["match_id"] == row["match_id"]]
        if m.empty:
            return ""
        m = m.iloc[0]
        opp = m["away_team"] if m["home_team"] == ENGLAND else m["home_team"]
        stage = m["competition_stage"]
        return f"{opp} ({stage})"

    for df in [passes, carries, shots]:
        df["match_label"] = df.apply(lambda r: opponent_label(r, eng_matches), axis=1)

    return eng_matches, passes, carries, shots


eng_matches, passes, carries, shots = load_data()
match_labels = sorted(passes["match_label"].unique())

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/be/Flag_of_England.svg/200px-Flag_of_England.svg.png", width=80)
st.sidebar.title("🏴 England Euro 2024")
st.sidebar.markdown("**Offensive Ball Progression**")

selected_matches = st.sidebar.multiselect("Filter by match", match_labels, default=match_labels)

view = st.sidebar.radio(
    "Dashboard View",
    [
        "1 – Passes into Final Third",
        "2 – Progressive Passes",
        "3 – Carries into Final Third",
        "4 – Pass Network (Build-up)",
        "5 – Shot Map & xG",
        "6 – Final Third Entries by Player",
        "7 – Key Passes (Shot Assists)",
        "8 – Chance Creation Breakdown",
        "9 – Crosses Map",
        "10 – Shot Direction Map",
    ],
)

half_filter = st.sidebar.radio("Half", ["Both", "1st Half", "2nd Half"], horizontal=True)

# Player filter
all_players = sorted(
    set(passes["player"].dropna().unique())
    | set(carries["player"].dropna().unique())
    | set(shots["player"].dropna().unique())
)
selected_players = st.sidebar.multiselect(
    "Filter by player", all_players, default=[],
    help="Leave empty to show all players"
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: StatsBomb Open Data · Euro 2024")

# ── Apply filters ───────────────────────────────────────────────────────────

def apply_filters(df):
    out = df[df["match_label"].isin(selected_matches)].copy()
    if half_filter == "1st Half":
        out = out[out["period"] == 1]
    elif half_filter == "2nd Half":
        out = out[out["period"] == 2]
    if selected_players:
        out = out[out["player"].isin(selected_players)]
    return out

f_passes  = apply_filters(passes)
f_carries = apply_filters(carries)
f_shots   = apply_filters(shots)


# ── Arrow trace helper ──────────────────────────────────────────────────────

def _arrow_traces(df, x0, y0, x1, y1, color, hover_fn, name=""):
    """Create scatter traces that look like arrows (lines + end markers)."""
    traces = []
    xs, ys = [], []
    for _, r in df.iterrows():
        xs += [r[x0], r[x1], None]
        ys += [r[y0], r[y1], None]
    traces.append(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=color, width=1.3),
        opacity=0.45, hoverinfo="skip", showlegend=False, name=name,
    ))
    traces.append(go.Scatter(
        x=df[x1], y=df[y1], mode="markers",
        marker=dict(size=6, color=color, symbol="arrow-up",
                    line=dict(width=0.5, color="white"), opacity=0.8),
        text=df.apply(hover_fn, axis=1), hoverinfo="text",
        hoverlabel=dict(bgcolor=PITCH_BG, font_size=12, font_color="white"),
        name=name, showlegend=bool(name),
    ))
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 1 – Passes into Final Third
# ═══════════════════════════════════════════════════════════════════════════
if view.startswith("1 "):
    st.title("Passes into the Final Third")
    st.caption("Completed passes that start outside the final third (x < 80) and end inside it (x ≥ 80)")

    mask = (
        (f_passes["location_x"] < FINAL_THIRD_X) &
        (f_passes["pass_end_location_x"] >= FINAL_THIRD_X) &
        (f_passes["pass_outcome"].isna())
    )
    ft_passes = f_passes[mask]

    col1, col2, col3 = st.columns(3)
    col1.metric("Final-Third Entries (Pass)", len(ft_passes))
    col2.metric("Unique Passers", ft_passes["player"].nunique())
    col3.metric("Avg per Match", f"{len(ft_passes)/max(len(selected_matches),1):.1f}")

    fig = go.Figure()
    for tr in _arrow_traces(
        ft_passes, "location_x", "location_y",
        "pass_end_location_x", "pass_end_location_y",
        color=ENG_RED,
        hover_fn=lambda r: (
            f"<b>{r['player']}</b><br>"
            f"To: {r['pass_recipient'] if pd.notna(r.get('pass_recipient')) else '—'}<br>"
            f"Min {int(r['minute'])}' | {r['match_label']}"
        ),
        name="Pass into Final 3rd",
    ):
        fig.add_trace(tr)

    fig.update_layout(**_full_pitch_layout(
        title="England – Passes into the Final Third",
        shapes=_full_pitch_shapes(highlight_final_third=True, ft_color=ENG_RED),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    # Top passers bar chart
    st.subheader("Top Passers into the Final Third")
    top = ft_passes.groupby("player").size().reset_index(name="Count").sort_values("Count", ascending=False).head(10)
    bar = px.bar(top, y="player", x="Count", orientation="h", color_discrete_sequence=[ENG_RED])
    bar.update_layout(
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Passes into Final Third", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 2 – Progressive Passes
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("2"):
    st.title("Progressive Passes")
    st.caption("Completed passes that move the ball ≥ 10 m closer to the opponent's goal")

    completed = f_passes[f_passes["pass_outcome"].isna()].copy()
    completed["dist_to_goal_start"] = 120 - completed["location_x"]
    completed["dist_to_goal_end"]   = 120 - completed["pass_end_location_x"]
    completed["progress"] = completed["dist_to_goal_start"] - completed["dist_to_goal_end"]
    prog = completed[(completed["progress"] >= 10) & (completed["pass_end_location_x"] >= 60)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Progressive Passes", len(prog))
    col2.metric("Unique Passers", prog["player"].nunique())
    col3.metric("Avg per Match", f"{len(prog)/max(len(selected_matches),1):.1f}")

    fig = go.Figure()

    if len(prog):
        p95 = prog["progress"].quantile(0.95)
        prog = prog.copy()
        prog["color_val"] = prog["progress"].clip(upper=p95)
    else:
        prog = prog.copy()
        prog["color_val"] = []

    xs, ys = [], []
    for _, r in prog.iterrows():
        xs += [r["location_x"], r["pass_end_location_x"], None]
        ys += [r["location_y"], r["pass_end_location_y"], None]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=ENG_RED, width=1.2),
        opacity=0.4, hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=prog["pass_end_location_x"], y=prog["pass_end_location_y"],
        mode="markers",
        marker=dict(
            size=8, color=prog["color_val"],
            colorscale=[[0, ENG_BLUE], [1, ENG_RED]],
            showscale=True, colorbar=dict(title="Metres", tickfont=dict(color="white"),
                                          title_font=dict(color="white")),
            line=dict(width=0.5, color="white"), opacity=0.8,
        ),
        text=prog.apply(
            lambda r: (
                f"<b>{r['player']}</b><br>"
                f"Progress: {r['progress']:.1f}m<br>"
                f"To: {r['pass_recipient'] if pd.notna(r.get('pass_recipient')) else '—'}<br>"
                f"Min {int(r['minute'])}' | {r['match_label']}"
            ), axis=1),
        hoverinfo="text",
        hoverlabel=dict(bgcolor=PITCH_BG, font_size=12, font_color="white"),
        name="Progressive Pass",
    ))

    fig.update_layout(**_full_pitch_layout(
        title="England – Progressive Passes (colour = metres gained)",
        shapes=_full_pitch_shapes(),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    st.subheader("Top Progressive Passers")
    top = prog.groupby("player").agg(
        Count=("id", "size"), Avg_m=("progress", "mean")
    ).sort_values("Count", ascending=False).head(10).reset_index()
    top["Avg_m"] = top["Avg_m"].round(1)
    bar = go.Figure()
    bar.add_trace(go.Bar(
        y=top["player"], x=top["Count"], orientation="h",
        marker_color=ENG_BLUE,
        text=top["Avg_m"].apply(lambda v: f"avg {v}m"), textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    bar.update_layout(
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Progressive Passes", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=10, r=80, t=10, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 3 – Carries into Final Third
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("3"):
    st.title("Carries into the Final Third")
    st.caption("Ball carries that start outside the final third and end inside it")

    mask = (
        (f_carries["location_x"] < FINAL_THIRD_X) &
        (f_carries["carry_end_location_x"] >= FINAL_THIRD_X)
    )
    ft_carries = f_carries[mask]

    col1, col2, col3 = st.columns(3)
    col1.metric("Final-Third Entries (Carry)", len(ft_carries))
    col2.metric("Unique Carriers", ft_carries["player"].nunique())
    col3.metric("Avg per Match", f"{len(ft_carries)/max(len(selected_matches),1):.1f}")

    fig = go.Figure()
    for tr in _arrow_traces(
        ft_carries, "location_x", "location_y",
        "carry_end_location_x", "carry_end_location_y",
        color=ENG_BLUE,
        hover_fn=lambda r: (
            f"<b>{r['player']}</b><br>"
            f"Min {int(r['minute'])}' | {r['match_label']}"
        ),
        name="Carry into Final 3rd",
    ):
        fig.add_trace(tr)

    fig.update_layout(**_full_pitch_layout(
        title="England – Carries into the Final Third",
        shapes=_full_pitch_shapes(highlight_final_third=True, ft_color=ENG_BLUE),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    # Carry end-point heatmap
    st.subheader("Carry End-Point Density (Final Third)")
    if len(ft_carries):
        heat = go.Figure(go.Histogram2dContour(
            x=ft_carries["carry_end_location_y"],
            y=ft_carries["carry_end_location_x"],
            colorscale="Reds", showscale=False,
            contours=dict(showlabels=False),
            ncontours=15, opacity=0.7, hoverinfo="skip",
        ))
        heat.update_layout(**_half_pitch_layout(
            title="Where England carried the ball into the final third",
            shapes=_half_pitch_shapes(),
        ))
        st.plotly_chart(heat, use_container_width=True, config=PLOTLY_CFG)

    st.subheader("Top Carriers into Final Third")
    top = ft_carries.groupby("player").size().reset_index(name="Count").sort_values("Count", ascending=False).head(10)
    bar = px.bar(top, y="player", x="Count", orientation="h", color_discrete_sequence=[ENG_BLUE])
    bar.update_layout(
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Carries into Final Third", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 4 – Pass Network
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("4"):
    st.title("Build-Up Pass Network")
    st.caption("Average positions & passing connections for completed passes")

    completed = f_passes[f_passes["pass_outcome"].isna()].copy()

    avg_pos = completed.groupby("player").agg(
        x=("location_x", "mean"), y=("location_y", "mean"), count=("id", "size")
    ).reset_index()
    avg_pos = avg_pos[avg_pos["count"] >= 5]

    pairs = completed[completed["pass_recipient"].notna()].copy()
    pairs["pair"] = pairs.apply(lambda r: tuple(sorted([r["player"], r["pass_recipient"]])), axis=1)
    pair_counts = pairs.groupby("pair").size().reset_index(name="count")
    pair_counts = pair_counts[pair_counts["count"] >= 3]

    fig = go.Figure()

    if len(pair_counts):
        max_w = pair_counts["count"].max()
        for _, r in pair_counts.iterrows():
            p1, p2 = r["pair"]
            pos1 = avg_pos[avg_pos["player"] == p1]
            pos2 = avg_pos[avg_pos["player"] == p2]
            if pos1.empty or pos2.empty:
                continue
            w = 1 + 5 * (r["count"] / max_w)
            fig.add_trace(go.Scatter(
                x=[pos1["x"].values[0], pos2["x"].values[0]],
                y=[pos1["y"].values[0], pos2["y"].values[0]],
                mode="lines", line=dict(color=ENG_WHITE, width=w),
                opacity=0.45, hoverinfo="skip", showlegend=False,
            ))

    if len(avg_pos):
        max_c = avg_pos["count"].max()
        avg_pos["msize"] = 12 + 28 * (avg_pos["count"] / max_c)
        avg_pos["surname"] = avg_pos["player"].apply(
            lambda p: p.split()[-1] if " " in p else p)
        fig.add_trace(go.Scatter(
            x=avg_pos["x"], y=avg_pos["y"], mode="markers+text",
            marker=dict(size=avg_pos["msize"], color=ENG_RED,
                        line=dict(width=1.5, color="white")),
            text=avg_pos["surname"], textposition="bottom center",
            textfont=dict(color="white", size=9),
            hovertext=avg_pos.apply(
                lambda r: f"<b>{r['player']}</b><br>Passes: {r['count']}<br>Avg pos: ({r['x']:.1f}, {r['y']:.1f})",
                axis=1),
            hoverinfo="text",
            hoverlabel=dict(bgcolor=PITCH_BG, font_size=12, font_color="white"),
            name="Player",
        ))

    fig.update_layout(**_full_pitch_layout(
        title="England – Pass Network (Build-Up Play)",
        shapes=_full_pitch_shapes(),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    st.subheader("Pass Direction Breakdown")
    completed["direction"] = "Sideways"
    completed.loc[completed["pass_end_location_x"] - completed["location_x"] > 5, "direction"] = "Forward"
    completed.loc[completed["location_x"] - completed["pass_end_location_x"] > 5, "direction"] = "Backward"
    dir_counts = completed["direction"].value_counts().reset_index()
    dir_counts.columns = ["Direction", "Count"]
    pie = px.pie(dir_counts, names="Direction", values="Count",
                 color_discrete_sequence=[ENG_RED, ENG_BLUE, LINE_COL])
    pie.update_layout(
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        font=dict(color="white"), margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 5 – Shot Map & xG
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("5"):
    st.title("Shot Map & Expected Goals")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shots", len(f_shots))
    col2.metric("Total xG", f"{f_shots['shot_statsbomb_xg'].sum():.2f}")
    goals = f_shots[f_shots["shot_outcome"] == "Goal"]
    col3.metric("Goals", len(goals))
    col4.metric("xG per Shot", f"{f_shots['shot_statsbomb_xg'].mean():.3f}" if len(f_shots) else "0")

    plotly_fig = go.Figure()
    shot_data = f_shots.copy()
    shot_data["is_goal"] = shot_data["shot_outcome"] == "Goal"
    shot_data["marker_size"] = shot_data["shot_statsbomb_xg"] * 40 + 8

    for is_goal, label, color, symbol in [
        (True, "Goal ★", ENG_RED, "star"),
        (False, "No Goal", ENG_BLUE, "circle"),
    ]:
        subset = shot_data[shot_data["is_goal"] == is_goal]
        if subset.empty:
            continue
        plotly_fig.add_trace(go.Scatter(
            x=subset["location_y"], y=subset["location_x"],
            mode="markers", name=label,
            marker=dict(size=subset["marker_size"], color=color, symbol=symbol,
                        line=dict(width=1, color="white"),
                        opacity=0.9 if is_goal else 0.65),
            text=subset.apply(
                lambda r: (
                    f"<b>{r['player']}</b><br>"
                    f"xG: {r['shot_statsbomb_xg']:.3f}<br>"
                    f"Outcome: {r['shot_outcome']}<br>"
                    f"Minute: {int(r['minute'])}'<br>"
                    f"Body Part: {r['shot_body_part']}<br>"
                    f"Technique: {r['shot_technique']}<br>"
                    f"Match: {r['match_label']}"
                ), axis=1),
            hoverinfo="text",
            hoverlabel=dict(bgcolor=PITCH_BG, font_size=13, font_color="white"),
        ))

    plotly_fig.update_layout(**_half_pitch_layout(
        title="England – Shot Map (hover for details, size = xG)",
        shapes=_half_pitch_shapes(),
    ))
    st.plotly_chart(plotly_fig, use_container_width=True, config=PLOTLY_CFG)

    st.subheader("Shot Log")
    shot_table = f_shots[["player", "minute", "shot_statsbomb_xg", "shot_outcome",
                           "shot_body_part", "shot_technique", "match_label"]].copy()
    shot_table.columns = ["Player", "Minute", "xG", "Outcome", "Body Part", "Technique", "Match"]
    shot_table = shot_table.sort_values(["Match", "Minute"]).reset_index(drop=True)
    shot_table.index += 1
    st.dataframe(shot_table.style.format({"xG": "{:.3f}"}), use_container_width=True)

    st.subheader("Cumulative xG Timeline (per match)")
    xg_fig = go.Figure()
    for label in selected_matches:
        m = f_shots[f_shots["match_label"] == label].sort_values("minute")
        if m.empty:
            continue
        m = m.copy()
        m["cum_xg"] = m["shot_statsbomb_xg"].cumsum()
        xg_fig.add_trace(go.Scatter(
            x=m["minute"], y=m["cum_xg"], mode="lines+markers", name=label,
            line=dict(shape="hv", width=2), marker=dict(size=6),
            text=m.apply(
                lambda r: f"<b>{r['player']}</b><br>xG: {r['shot_statsbomb_xg']:.3f}<br>Cum xG: {r['cum_xg']:.3f}<br>Min {int(r['minute'])}'",
                axis=1),
            hoverinfo="text",
        ))
    xg_fig.update_layout(
        xaxis=dict(title="Minute", color="white", gridcolor="#333"),
        yaxis=dict(title="Cumulative xG", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG,
        legend=dict(font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.3)"),
        height=350, margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(xg_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 6 – Final Third Entries by Player
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("6"):
    st.title("Final Third Entries by Player")
    st.caption("Combined pass + carry entries into the final third, broken down by player")

    p_mask = (
        (f_passes["location_x"] < FINAL_THIRD_X) &
        (f_passes["pass_end_location_x"] >= FINAL_THIRD_X) &
        (f_passes["pass_outcome"].isna())
    )
    pass_entries = f_passes[p_mask].groupby("player").size().reset_index(name="pass_entries")

    c_mask = (
        (f_carries["location_x"] < FINAL_THIRD_X) &
        (f_carries["carry_end_location_x"] >= FINAL_THIRD_X)
    )
    carry_entries = f_carries[c_mask].groupby("player").size().reset_index(name="carry_entries")

    combined = pass_entries.merge(carry_entries, on="player", how="outer").fillna(0)
    combined["total"] = combined["pass_entries"] + combined["carry_entries"]
    combined = combined.sort_values("total", ascending=False).head(15)

    bar = go.Figure()
    bar.add_trace(go.Bar(
        y=combined["player"], x=combined["pass_entries"], orientation="h",
        name="Pass entries", marker_color=ENG_RED,
    ))
    bar.add_trace(go.Bar(
        y=combined["player"], x=combined["carry_entries"], orientation="h",
        name="Carry entries", marker_color=ENG_BLUE,
    ))
    bar.update_layout(
        barmode="stack",
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Final Third Entries", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=450,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=40, b=40),
        title=dict(text="England – Final Third Entries by Player", font=dict(color="white", size=15)),
    )
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("Entry Zones (Left / Centre / Right)")
    ft_p = f_passes[p_mask].copy()
    ft_c = f_carries[c_mask].copy()

    def zone(y_val):
        if y_val < 26.67:
            return "Right"
        elif y_val < 53.33:
            return "Centre"
        else:
            return "Left"

    zones_p = ft_p["pass_end_location_y"].apply(zone).value_counts()
    zones_c = ft_c["carry_end_location_y"].apply(zone).value_counts()
    zone_df = pd.DataFrame({"Pass": zones_p, "Carry": zones_c}).fillna(0)
    zone_df = zone_df.reindex(["Left", "Centre", "Right"]).reset_index()
    zone_df.columns = ["Zone", "Pass", "Carry"]

    zone_fig = go.Figure()
    zone_fig.add_trace(go.Bar(x=zone_df["Zone"], y=zone_df["Pass"], name="Pass", marker_color=ENG_RED))
    zone_fig.add_trace(go.Bar(x=zone_df["Zone"], y=zone_df["Carry"], name="Carry", marker_color=ENG_BLUE))
    zone_fig.update_layout(
        barmode="group",
        xaxis=dict(title="", color="white"),
        yaxis=dict(title="Entries", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(zone_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 7 – Key Passes (Shot Assists)
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("7"):
    st.title("Key Passes – Passes Leading to Shots")
    st.caption("Every pass that directly set up a shot (shot assist) or goal (goal assist)")

    key_passes = f_passes[f_passes["pass_shot_assist"] == 1].copy()
    goal_assists = f_passes[f_passes["pass_goal_assist"] == 1].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shot Assists", len(key_passes))
    col2.metric("Goal Assists", len(goal_assists))
    col3.metric("Unique Creators", key_passes["player"].nunique())
    col4.metric("Avg Key Passes / Match", f"{len(key_passes)/max(len(selected_matches),1):.1f}")

    # ---------- pitch map ----------
    fig = go.Figure()

    # Non-goal-assist key passes
    non_ga = key_passes[key_passes["pass_goal_assist"] != 1]
    if len(non_ga):
        for tr in _arrow_traces(
            non_ga, "location_x", "location_y",
            "pass_end_location_x", "pass_end_location_y",
            color=ENG_RED,
            hover_fn=lambda r: (
                f"<b>{r['player']}</b> → {r['pass_recipient'] if pd.notna(r.get('pass_recipient')) else '—'}<br>"
                f"Min {int(r['minute'])}' | {r['match_label']}<br>"
                f"Type: {'Cross' if r.get('pass_cross')==1 else 'Cut-back' if r.get('pass_cut_back')==1 else 'Through ball' if r.get('pass_through_ball')==1 else 'Regular'}"
            ),
            name="Shot Assist",
        ):
            fig.add_trace(tr)

    # Goal assists highlighted
    if len(goal_assists):
        for tr in _arrow_traces(
            goal_assists, "location_x", "location_y",
            "pass_end_location_x", "pass_end_location_y",
            color="#FFD700",
            hover_fn=lambda r: (
                f"⚽ <b>GOAL ASSIST</b><br>"
                f"<b>{r['player']}</b> → {r['pass_recipient'] if pd.notna(r.get('pass_recipient')) else '—'}<br>"
                f"Min {int(r['minute'])}' | {r['match_label']}"
            ),
            name="Goal Assist ⚽",
        ):
            fig.add_trace(tr)

    fig.update_layout(**_full_pitch_layout(
        title="England – Key Passes (Red = Shot Assist, Gold = Goal Assist)",
        shapes=_full_pitch_shapes(highlight_final_third=True, ft_color=ENG_RED),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    # ---------- key pass type breakdown ----------
    st.subheader("Key Pass Type")
    kp = key_passes.copy()
    def _kp_type(r):
        if r.get("pass_cross") == 1:
            return "Cross"
        elif r.get("pass_cut_back") == 1:
            return "Cut-back"
        elif r.get("pass_through_ball") == 1:
            return "Through Ball"
        else:
            return "Regular / Other"
    kp["kp_type"] = kp.apply(_kp_type, axis=1)
    type_counts = kp["kp_type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    type_fig = px.bar(type_counts, x="Type", y="Count",
                      color_discrete_sequence=[ENG_RED, ENG_BLUE, "#FFD700", LINE_COL])
    type_fig.update_layout(
        xaxis=dict(title="", color="white"),
        yaxis=dict(title="Key Passes", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=300,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(type_fig, use_container_width=True)

    # ---------- top creators ----------
    st.subheader("Top Chance Creators")
    top_kp = key_passes.groupby("player").size().reset_index(name="Key Passes").sort_values("Key Passes", ascending=False).head(10)
    bar = px.bar(top_kp, y="player", x="Key Passes", orientation="h",
                 color_discrete_sequence=[ENG_RED])
    bar.update_layout(
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Shot-Creating Passes", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)

    # ---------- key pass log ----------
    st.subheader("Key Pass Log")
    log = key_passes[["player", "pass_recipient", "minute", "match_label"]].copy()
    log["Goal Assist"] = key_passes["pass_goal_assist"].apply(lambda v: "✅" if v == 1 else "")
    log.columns = ["Passer", "Recipient", "Minute", "Match", "Goal Assist"]
    log = log.sort_values(["Match", "Minute"]).reset_index(drop=True)
    log.index += 1
    st.dataframe(log, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 8 – Chance Creation Breakdown
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("8"):
    st.title("Chance Creation Breakdown")
    st.caption("How England build their shooting opportunities – patterns, zones, and key actions  \n"
               "*(Penalty-shootout shots excluded; in-game penalties kept)*")

    # Exclude penalty-shootout shots (penalties at minute >= 120 in Switzerland match 3942227)
    _shootout_mask = (
        (f_shots["match_id"] == 3942227)
        & (f_shots["shot_type"] == "Penalty")
        & (f_shots["minute"] >= 120)
    )
    cc_shots = f_shots[~_shootout_mask].copy()
    # Rename 'Other' play pattern to 'Penalty' for clarity
    cc_shots["play_pattern"] = cc_shots["play_pattern"].replace({"Other": "Penalty"})

    # Link shots to their key passes
    shots_with_kp = cc_shots[cc_shots["shot_key_pass_id"].notna()].copy()
    shots_no_kp = cc_shots[cc_shots["shot_key_pass_id"].isna()].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shots", len(cc_shots))
    col2.metric("Shots from Key Pass", len(shots_with_kp))
    col3.metric("Unassisted / Individual", len(shots_no_kp))
    col4.metric("Total xG", f"{cc_shots['shot_statsbomb_xg'].sum():.2f}")

    # ---------- play pattern breakdown (how chances arise) ----------
    st.subheader("Shot Origin – Play Pattern")
    pp = cc_shots["play_pattern"].value_counts().reset_index()
    pp.columns = ["Pattern", "Shots"]
    pp_xg = cc_shots.groupby("play_pattern")["shot_statsbomb_xg"].sum().reset_index()
    pp_xg.columns = ["Pattern", "xG"]
    pp = pp.merge(pp_xg, on="Pattern")
    pp = pp.sort_values("Shots", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(pp, names="Pattern", values="Shots",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title="Shots by Play Pattern")
        pie.update_layout(
            plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
            font=dict(color="white"), margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(color="white", size=14),
        )
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        xg_bar = px.bar(pp, x="Pattern", y="xG",
                        color_discrete_sequence=[ENG_RED], title="xG by Play Pattern")
        xg_bar.update_layout(
            xaxis=dict(title="", color="white", tickangle=-35),
            yaxis=dict(title="xG", color="white", gridcolor="#333"),
            plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
            margin=dict(l=10, r=10, t=40, b=80),
            title_font=dict(color="white", size=14),
        )
        st.plotly_chart(xg_bar, use_container_width=True)

    # ---------- key-pass action type that led to shots ----------
    st.subheader("Types of Passes Creating Chances")
    kp_all = f_passes[f_passes["pass_shot_assist"] == 1].copy()
    def _action_type(r):
        if r.get("pass_cross") == 1:
            return "Cross"
        elif r.get("pass_cut_back") == 1:
            return "Cut-back"
        elif r.get("pass_through_ball") == 1:
            return "Through Ball"
        elif r.get("pass_switch") == 1:
            return "Switch"
        else:
            return "Short / Regular"
    kp_all["action"] = kp_all.apply(_action_type, axis=1)
    act = kp_all["action"].value_counts().reset_index()
    act.columns = ["Action", "Count"]

    act_bar = px.bar(act, x="Action", y="Count",
                     color="Action",
                     color_discrete_map={
                         "Cross": ENG_RED, "Cut-back": "#FFD700",
                         "Through Ball": ENG_BLUE, "Switch": LINE_COL,
                         "Short / Regular": "#4CAF50",
                     })
    act_bar.update_layout(
        xaxis=dict(title="", color="white"),
        yaxis=dict(title="Key Passes", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=320,
        margin=dict(l=10, r=10, t=10, b=40), showlegend=False,
    )
    st.plotly_chart(act_bar, use_container_width=True)

    # ---------- chance creation by match ----------
    st.subheader("Chances Created per Match")
    match_kp = kp_all.groupby("match_label").size().reset_index(name="Key Passes")
    match_shots = cc_shots.groupby("match_label").agg(
        Shots=("id", "size"), xG=("shot_statsbomb_xg", "sum")
    ).reset_index()
    match_cc = match_shots.merge(match_kp, on="match_label", how="left").fillna(0)
    match_cc = match_cc.sort_values("Shots", ascending=False)
    match_cc["xG"] = match_cc["xG"].round(2)

    cc_fig = go.Figure()
    cc_fig.add_trace(go.Bar(
        x=match_cc["match_label"], y=match_cc["Shots"],
        name="Shots", marker_color=ENG_BLUE,
    ))
    cc_fig.add_trace(go.Bar(
        x=match_cc["match_label"], y=match_cc["Key Passes"],
        name="Key Passes", marker_color=ENG_RED,
    ))
    cc_fig.add_trace(go.Scatter(
        x=match_cc["match_label"], y=match_cc["xG"],
        name="xG", mode="lines+markers",
        line=dict(color="#FFD700", width=2), marker=dict(size=8),
        yaxis="y2",
    ))
    cc_fig.update_layout(
        barmode="group",
        xaxis=dict(title="", color="white", tickangle=-25),
        yaxis=dict(title="Count", color="white", gridcolor="#333"),
        yaxis2=dict(title="xG", color="#FFD700", overlaying="y", side="right",
                    showgrid=False),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=380,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=50, t=10, b=80),
    )
    st.plotly_chart(cc_fig, use_container_width=True)

    # ---------- shot zone heatmap ----------
    st.subheader("Shot Zone Density")
    if len(cc_shots):
        heat = go.Figure(go.Histogram2dContour(
            x=cc_shots["location_y"],
            y=cc_shots["location_x"],
            colorscale="YlOrRd", showscale=True,
            colorbar=dict(title="Density", tickfont=dict(color="white"),
                          title_font=dict(color="white")),
            contours=dict(showlabels=False),
            ncontours=12, opacity=0.7, hoverinfo="skip",
        ))
        heat.update_layout(**_half_pitch_layout(
            title="Where England take their shots from",
            shapes=_half_pitch_shapes(),
        ))
        st.plotly_chart(heat, use_container_width=True, config=PLOTLY_CFG)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 9 – Crosses Map
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("9"):
    st.title("Crosses – Delivery Map")
    st.caption("All crosses attempted by England, showing origin, end point, and outcome")

    crosses = f_passes[f_passes["pass_cross"] == 1].copy()
    crosses["completed"] = crosses["pass_outcome"].isna()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Crosses", len(crosses))
    col2.metric("Completed", int(crosses["completed"].sum()))
    col3.metric("Completion %", f"{crosses['completed'].mean()*100:.0f}%" if len(crosses) else "0%")
    cross_assists = crosses[crosses["pass_shot_assist"] == 1]
    col4.metric("Led to Shot", len(cross_assists))

    # ---------- pitch map ----------
    fig = go.Figure()

    for comp, label, color in [
        (True, "Completed Cross", ENG_RED),
        (False, "Incomplete Cross", LINE_COL),
    ]:
        subset = crosses[crosses["completed"] == comp]
        if subset.empty:
            continue
        for tr in _arrow_traces(
            subset, "location_x", "location_y",
            "pass_end_location_x", "pass_end_location_y",
            color=color,
            hover_fn=lambda r: (
                f"<b>{r['player']}</b><br>"
                f"To: {r['pass_recipient'] if pd.notna(r.get('pass_recipient')) else '—'}<br>"
                f"Min {int(r['minute'])}' | {r['match_label']}<br>"
                f"{'✅ Completed' if r.get('completed') else '❌ Incomplete'}"
                f"{'  ⚡ Shot Assist' if r.get('pass_shot_assist')==1 else ''}"
            ),
            name=label,
        ):
            fig.add_trace(tr)

    fig.update_layout(**_full_pitch_layout(
        title="England – Crosses (Red = Completed, Grey = Incomplete)",
        shapes=_full_pitch_shapes(),
    ))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    # ---------- left vs right origin ----------
    st.subheader("Cross Origin – Left vs Right")
    crosses["side"] = crosses["location_y"].apply(
        lambda y: "Left" if y > 40 else "Right"
    )
    side_counts = crosses.groupby("side").agg(
        Total=("id", "size"),
        Completed=("completed", "sum"),
    ).reset_index()
    side_counts["Incomplete"] = side_counts["Total"] - side_counts["Completed"]

    side_fig = go.Figure()
    side_fig.add_trace(go.Bar(
        x=side_counts["side"], y=side_counts["Completed"],
        name="Completed", marker_color=ENG_RED,
    ))
    side_fig.add_trace(go.Bar(
        x=side_counts["side"], y=side_counts["Incomplete"],
        name="Incomplete", marker_color=LINE_COL,
    ))
    side_fig.update_layout(
        barmode="stack",
        xaxis=dict(title="", color="white"),
        yaxis=dict(title="Crosses", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=300,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(side_fig, use_container_width=True)

    # ---------- top crossers ----------
    st.subheader("Top Crossers")
    top_cross = crosses.groupby("player").agg(
        Total=("id", "size"), Completed=("completed", "sum")
    ).reset_index().sort_values("Total", ascending=False).head(10)
    top_cross["Incomplete"] = top_cross["Total"] - top_cross["Completed"]

    tc_fig = go.Figure()
    tc_fig.add_trace(go.Bar(
        y=top_cross["player"], x=top_cross["Completed"],
        orientation="h", name="Completed", marker_color=ENG_RED,
    ))
    tc_fig.add_trace(go.Bar(
        y=top_cross["player"], x=top_cross["Incomplete"],
        orientation="h", name="Incomplete", marker_color=LINE_COL,
    ))
    tc_fig.update_layout(
        barmode="stack",
        yaxis=dict(autorange="reversed", title="", color="white"),
        xaxis=dict(title="Crosses", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=380,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(tc_fig, use_container_width=True)

    # ---------- crosses per match ----------
    st.subheader("Crosses per Match")
    cpm = crosses.groupby("match_label").agg(
        Total=("id", "size"), Completed=("completed", "sum")
    ).reset_index().sort_values("Total", ascending=False)
    cpm_fig = px.bar(cpm, x="match_label", y=["Completed", "Total"],
                     barmode="overlay",
                     color_discrete_sequence=[ENG_RED, "rgba(192,192,192,0.4)"])
    cpm_fig.update_layout(
        xaxis=dict(title="", color="white", tickangle=-25),
        yaxis=dict(title="Crosses", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=320,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=10, b=80),
    )
    st.plotly_chart(cpm_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# VIEW 10 – Shot Direction Map
# ═══════════════════════════════════════════════════════════════════════════
elif view.startswith("10"):
    st.title("Shot Direction Map")
    st.caption("Trajectory of every England shot – colour-coded by which side of the goal it targets  \n"
               "*(Penalty-shootout shots excluded)*")

    # Exclude shootout penalties
    _so_mask = (
        (f_shots["match_id"] == 3942227)
        & (f_shots["shot_type"] == "Penalty")
        & (f_shots["minute"] >= 120)
    )
    sd_shots = f_shots[~_so_mask].copy()

    # --- Classify shot direction based on end_y relative to goal posts ---
    # StatsBomb: goal posts at y=36 and y=44, centre at y=40
    def _classify_direction(end_y):
        if pd.isna(end_y):
            return "Unknown"
        if end_y < 36:
            return "Wide Right"
        elif end_y < 38:
            return "Near Post (Right)"
        elif end_y <= 42:
            return "Centre"
        elif end_y <= 44:
            return "Far Post (Left)"
        else:
            return "Wide Left"

    sd_shots["shot_direction"] = sd_shots["shot_end_location_y"].apply(_classify_direction)

    DIR_COLORS = {
        "Near Post (Right)": "#FF4136",   # red
        "Centre":           "#FFD700",    # gold
        "Far Post (Left)":  "#0074D9",    # blue
        "Wide Right":       "#FF851B",    # orange
        "Wide Left":        "#B10DC9",    # purple
        "Unknown":          "#AAAAAA",    # grey
    }

    # --- Metrics ---
    on_target = sd_shots[sd_shots["shot_direction"].isin(["Near Post (Right)", "Centre", "Far Post (Left)"])]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shots", len(sd_shots))
    col2.metric("On Frame", len(on_target))
    col3.metric("Near Post (R)", len(sd_shots[sd_shots["shot_direction"] == "Near Post (Right)"]))
    col4.metric("Far Post (L)", len(sd_shots[sd_shots["shot_direction"] == "Far Post (Left)"]))

    c1, c2 = st.columns([3, 2])

    # --- Pitch map with shot trajectories ---
    with c1:
        fig = go.Figure()

        for direction in ["Near Post (Right)", "Centre", "Far Post (Left)", "Wide Right", "Wide Left", "Unknown"]:
            subset = sd_shots[sd_shots["shot_direction"] == direction]
            if subset.empty:
                continue
            color = DIR_COLORS[direction]
            # Arrow lines from shot origin to end location
            for tr in _arrow_traces(
                subset, "location_x", "location_y",
                "shot_end_location_x", "shot_end_location_y",
                color=color,
                hover_fn=lambda r, d=direction: (
                    "<b>{}</b><br>"
                    "Min {}' | {}<br>"
                    "Direction: {}<br>"
                    "xG: {:.3f} | {}<br>"
                    "{}".format(
                        r["player"], int(r["minute"]), r["match_label"],
                        d, r["shot_statsbomb_xg"], r["shot_outcome"],
                        r["shot_body_part"] if pd.notna(r.get("shot_body_part")) else "",
                    )
                ),
                name=direction,
            ):
                fig.add_trace(tr)

        # Draw goal-post markers on the pitch
        fig.add_shape(type="rect", x0=120, y0=36, x1=122, y1=44,
                      fillcolor="rgba(255,255,255,0.15)", line=dict(color="white", width=2))

        fig.update_layout(**_full_pitch_layout(
            title="England – Shot Trajectories by Goal Side",
            shapes=_full_pitch_shapes(highlight_final_third=False),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    # --- Direction distribution bar chart ---
    with c2:
        dir_counts = sd_shots["shot_direction"].value_counts().reindex(
            ["Near Post (Right)", "Centre", "Far Post (Left)", "Wide Right", "Wide Left", "Unknown"]
        ).dropna().reset_index()
        dir_counts.columns = ["Direction", "Count"]

        dir_bar = px.bar(
            dir_counts, x="Count", y="Direction", orientation="h",
            color="Direction",
            color_discrete_map=DIR_COLORS,
        )
        dir_bar.update_layout(
            yaxis=dict(title="", color="white", autorange="reversed"),
            xaxis=dict(title="Shots", color="white", gridcolor="#333"),
            plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
            showlegend=False, margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(dir_bar, use_container_width=True)

    # --- Direction by outcome ---
    st.subheader("Shot Direction × Outcome")
    ct = pd.crosstab(sd_shots["shot_direction"], sd_shots["shot_outcome"])
    # Reorder rows
    row_order = ["Near Post (Right)", "Centre", "Far Post (Left)", "Wide Right", "Wide Left"]
    ct = ct.reindex([r for r in row_order if r in ct.index])
    ct_fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="Outcome", y="Direction", color="Count"),
    )
    ct_fig.update_layout(
        xaxis=dict(color="white", tickangle=-25),
        yaxis=dict(color="white"),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=10, r=10, t=10, b=80),
        coloraxis_colorbar=dict(tickfont=dict(color="white"), title_font=dict(color="white")),
    )
    st.plotly_chart(ct_fig, use_container_width=True)

    # --- Goal-frame view (front-on) ---
    st.subheader("Goal-Frame View – Where Shots End Up")
    st.caption("Front-on view of the goal (8m wide × ~2.44m high). Only shots reaching the goal line shown.")

    # Filter to shots that reach x >= 118 (near goal line) and have z-coordinate
    gf = sd_shots[
        (sd_shots["shot_end_location_x"] >= 118)
        & (sd_shots["shot_end_location_y"].notna())
    ].copy()
    gf["end_z"] = sd_shots["shot_end_location_z"]
    gf = gf[gf["end_z"].notna()].copy()

    gf_fig = go.Figure()

    # Draw the goal frame
    gf_fig.add_shape(type="rect", x0=36, y0=0, x1=44, y1=2.44,
                     line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)")

    # Plot each shot
    for _, r in gf.iterrows():
        direction = r["shot_direction"]
        color = DIR_COLORS.get(direction, "#AAAAAA")
        is_goal = r["shot_outcome"] == "Goal"
        gf_fig.add_trace(go.Scatter(
            x=[r["shot_end_location_y"]],
            y=[r["end_z"]],
            mode="markers",
            marker=dict(
                size=14 if is_goal else 9,
                color=color,
                symbol="star" if is_goal else "circle",
                line=dict(width=1, color="white") if is_goal else dict(width=0.5, color="rgba(255,255,255,0.5)"),
            ),
            hovertext="<b>{}</b><br>{}' | {}<br>xG: {:.3f}<br>{}{}".format(
                r["player"], int(r["minute"]), r["match_label"],
                r["shot_statsbomb_xg"], r["shot_outcome"],
                "<br>" + r["shot_body_part"] if pd.notna(r.get("shot_body_part")) else "",
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    gf_fig.update_layout(
        xaxis=dict(range=[33, 47], title="Goal Width (y)", color="white",
                   showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.3, 4], title="Height (m)", color="white",
                   gridcolor="#333", zeroline=False),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG, height=350,
        margin=dict(l=40, r=10, t=10, b=40),
        annotations=[
            dict(x=36, y=-0.15, text="Right<br>Post", showarrow=False,
                 font=dict(color="white", size=10)),
            dict(x=44, y=-0.15, text="Left<br>Post", showarrow=False,
                 font=dict(color="white", size=10)),
            dict(x=40, y=2.6, text="Crossbar (2.44m)", showarrow=False,
                 font=dict(color="#888", size=9)),
        ],
    )
    # Crossbar line
    gf_fig.add_shape(type="line", x0=36, y0=2.44, x1=44, y1=2.44,
                     line=dict(color="white", width=2, dash="dot"))

    st.plotly_chart(gf_fig, use_container_width=True, config=PLOTLY_CFG)
    st.caption("⭐ Stars = Goals | Circles = Non-goals | Colour = direction side")
