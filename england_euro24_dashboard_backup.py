import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import ast
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="England Euro 2024 – Offensive Analysis", layout="wide")

DATA_DIR = r"c:\Users\kenzi\Downloads\AnalisisDeJuego_Europea_Marzo26"
EVENTS_DIR = os.path.join(DATA_DIR, "Events")

ENGLAND = "England"

# StatsBomb pitch: 120 x 80, attacking left→right.  Final third ≥ 80
FINAL_THIRD_X = 80.0

# ── HELPERS ─────────────────────────────────────────────────────────────────

def safe_loc(val):
    """Parse a location string like '[60.0, 40.0]' into a list of floats."""
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
    ball_receipts = pd.read_excel(os.path.join(EVENTS_DIR, "SB_Euro24_Events_ball_receipts.xlsx"))

    # England match IDs
    eng_matches = matches[
        (matches["home_team"] == ENGLAND) | (matches["away_team"] == ENGLAND)
    ].copy()
    eng_match_ids = eng_matches["match_id"].tolist()

    # Filter to England possession
    passes  = passes[(passes["team"] == ENGLAND) & (passes["match_id"].isin(eng_match_ids))].copy()
    carries = carries[(carries["team"] == ENGLAND) & (carries["match_id"].isin(eng_match_ids))].copy()
    shots   = shots[(shots["team"] == ENGLAND) & (shots["match_id"].isin(eng_match_ids))].copy()
    ball_receipts = ball_receipts[
        (ball_receipts["team"] == ENGLAND) & (ball_receipts["match_id"].isin(eng_match_ids))
    ].copy()

    # Parse locations
    for df, cols in [
        (passes,  ["location", "pass_end_location"]),
        (carries, ["location", "carry_end_location"]),
        (shots,   ["location"]),
        (ball_receipts, ["location"]),
    ]:
        for c in cols:
            parsed = df[c].apply(safe_loc)
            df[f"{c}_x"] = parsed.apply(lambda v: v[0] if v and len(v) >= 2 else np.nan)
            df[f"{c}_y"] = parsed.apply(lambda v: v[1] if v and len(v) >= 2 else np.nan)

    # Label for opponent per match
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

    return eng_matches, passes, carries, shots, ball_receipts


eng_matches, passes, carries, shots, ball_receipts = load_data()

match_labels = sorted(passes["match_label"].unique())

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/be/Flag_of_England.svg/200px-Flag_of_England.svg.png", width=80)
st.sidebar.title("🏴 England Euro 2024")
st.sidebar.markdown("**Offensive Ball Progression**")

selected_matches = st.sidebar.multiselect(
    "Filter by match", match_labels, default=match_labels
)

view = st.sidebar.radio(
    "Dashboard View",
    [
        "1 – Passes into Final Third",
        "2 – Progressive Passes",
        "3 – Carries into Final Third",
        "4 – Pass Network (Build-up)",
        "5 – Shot Map & xG",
        "6 – Final Third Entries by Player",
    ],
)

half_filter = st.sidebar.radio("Half", ["Both", "1st Half", "2nd Half"], horizontal=True)

# Apply filters
def apply_filters(df):
    out = df[df["match_label"].isin(selected_matches)].copy()
    if half_filter == "1st Half":
        out = out[out["period"] == 1]
    elif half_filter == "2nd Half":
        out = out[out["period"] == 2]
    return out

f_passes  = apply_filters(passes)
f_carries = apply_filters(carries)
f_shots   = apply_filters(shots)

# ── COLOUR PALETTE ──────────────────────────────────────────────────────────
ENG_RED   = "#CF081F"
ENG_BLUE  = "#1B3D6D"
ENG_WHITE = "#FFFFFF"
PITCH_BG  = "#1a1a2e"
LINE_COL  = "#c0c0c0"

# ── VIEW 1: Passes into Final Third ────────────────────────────────────────
if view.startswith("1"):
    st.title("Passes into the Final Third")
    st.caption("Completed passes that start outside the final third (x < 80) and end inside it (x ≥ 80)")

    mask = (
        (f_passes["location_x"] < FINAL_THIRD_X) &
        (f_passes["pass_end_location_x"] >= FINAL_THIRD_X) &
        (f_passes["pass_outcome"].isna())  # completed passes only
    )
    ft_passes = f_passes[mask]

    col1, col2, col3 = st.columns(3)
    col1.metric("Final-Third Entries (Pass)", len(ft_passes))
    col2.metric("Unique Passers", ft_passes["player"].nunique())
    col3.metric("Avg per Match", f"{len(ft_passes)/max(len(selected_matches),1):.1f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color=LINE_COL)
    pitch.draw(ax=ax)

    # final third shading
    ax.axvspan(FINAL_THIRD_X, 120, alpha=0.10, color=ENG_RED, zorder=1)

    pitch.arrows(
        ft_passes["location_x"], ft_passes["location_y"],
        ft_passes["pass_end_location_x"], ft_passes["pass_end_location_y"],
        ax=ax, color=ENG_RED, width=1.5, headwidth=6, headlength=4, alpha=0.6, zorder=2
    )
    ax.set_title("England – Passes into the Final Third", color="white", fontsize=14, fontweight="bold", pad=10)
    fig.patch.set_facecolor(PITCH_BG)
    st.pyplot(fig)

    # Top passers
    st.subheader("Top Passers into the Final Third")
    top = ft_passes.groupby("player").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(top["player"], top["count"], color=ENG_RED)
    ax2.invert_yaxis()
    ax2.set_xlabel("Passes into Final Third")
    ax2.set_facecolor(PITCH_BG)
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    fig2.patch.set_facecolor(PITCH_BG)
    for spine in ax2.spines.values():
        spine.set_color(LINE_COL)
    st.pyplot(fig2)

# ── VIEW 2: Progressive Passes ────────────────────────────────────────────
elif view.startswith("2"):
    st.title("Progressive Passes")
    st.caption("Completed passes that move the ball ≥ 10 m closer to the opponent's goal (excluding set pieces into own half)")

    completed = f_passes[f_passes["pass_outcome"].isna()].copy()
    # Progressive: end_x closer to goal by >= 10, and end in opponent half
    completed["dist_to_goal_start"] = 120 - completed["location_x"]
    completed["dist_to_goal_end"]   = 120 - completed["pass_end_location_x"]
    completed["progress"] = completed["dist_to_goal_start"] - completed["dist_to_goal_end"]

    prog = completed[(completed["progress"] >= 10) & (completed["pass_end_location_x"] >= 60)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Progressive Passes", len(prog))
    col2.metric("Unique Passers", prog["player"].nunique())
    col3.metric("Avg per Match", f"{len(prog)/max(len(selected_matches),1):.1f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color=LINE_COL)
    pitch.draw(ax=ax)

    # Colour by distance progressed
    norm = plt.Normalize(10, prog["progress"].quantile(0.95) if len(prog) else 30)
    cmap = LinearSegmentedColormap.from_list("eng", [ENG_BLUE, ENG_RED])

    for _, r in prog.iterrows():
        ax.annotate(
            "", xy=(r["pass_end_location_x"], r["pass_end_location_y"]),
            xytext=(r["location_x"], r["location_y"]),
            arrowprops=dict(arrowstyle="->", color=cmap(norm(r["progress"])),
                            lw=1.2, alpha=0.55),
            zorder=2
        )

    ax.set_title("England – Progressive Passes", color="white", fontsize=14, fontweight="bold", pad=10)
    fig.patch.set_facecolor(PITCH_BG)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Metres Progressed", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    st.pyplot(fig)

    # Top progressive passers
    st.subheader("Top Progressive Passers")
    top = prog.groupby("player").agg(
        count=("id", "size"),
        avg_progress=("progress", "mean")
    ).sort_values("count", ascending=False).head(10).reset_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.barh(top["player"], top["count"], color=ENG_BLUE)
    ax2.invert_yaxis()
    ax2.set_xlabel("Progressive Passes")
    ax2.set_facecolor(PITCH_BG)
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    fig2.patch.set_facecolor(PITCH_BG)
    for spine in ax2.spines.values():
        spine.set_color(LINE_COL)
    # annotate avg metres
    for bar, val in zip(bars, top["avg_progress"]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"avg {val:.0f}m", va="center", color="white", fontsize=8)
    st.pyplot(fig2)

# ── VIEW 3: Carries into Final Third ──────────────────────────────────────
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

    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color=LINE_COL)
    pitch.draw(ax=ax)
    ax.axvspan(FINAL_THIRD_X, 120, alpha=0.10, color=ENG_BLUE, zorder=1)

    pitch.arrows(
        ft_carries["location_x"], ft_carries["location_y"],
        ft_carries["carry_end_location_x"], ft_carries["carry_end_location_y"],
        ax=ax, color=ENG_BLUE, width=1.5, headwidth=6, headlength=4, alpha=0.55, zorder=2
    )

    ax.set_title("England – Carries into the Final Third", color="white", fontsize=14, fontweight="bold", pad=10)
    fig.patch.set_facecolor(PITCH_BG)
    st.pyplot(fig)

    # Heatmap of carry end-points in final third
    st.subheader("Carry End-Point Heatmap (Final Third)")
    fig3, ax3 = plt.subplots(figsize=(6, 8))
    pitch_v = VerticalPitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color=LINE_COL,
                             half=True)
    pitch_v.draw(ax=ax3)
    if len(ft_carries):
        pitch_v.kdeplot(
            ft_carries["carry_end_location_x"], ft_carries["carry_end_location_y"],
            ax=ax3, cmap="Reds", fill=True, levels=50, alpha=0.7, zorder=2
        )
    ax3.set_title("Where England carried the ball into the final third",
                   color="white", fontsize=11, fontweight="bold", pad=8)
    fig3.patch.set_facecolor(PITCH_BG)
    st.pyplot(fig3)

    # Top carriers
    st.subheader("Top Carriers into Final Third")
    top = ft_carries.groupby("player").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(top["player"], top["count"], color=ENG_BLUE)
    ax2.invert_yaxis()
    ax2.set_xlabel("Carries into Final Third")
    ax2.set_facecolor(PITCH_BG)
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    fig2.patch.set_facecolor(PITCH_BG)
    for spine in ax2.spines.values():
        spine.set_color(LINE_COL)
    st.pyplot(fig2)

# ── VIEW 4: Pass Network ──────────────────────────────────────────────────
elif view.startswith("4"):
    st.title("Build-Up Pass Network")
    st.caption("Average positions & passing connections for completed passes in the build-up (own half → opponent half)")

    completed = f_passes[f_passes["pass_outcome"].isna()].copy()

    # Average position per player
    avg_pos = completed.groupby("player").agg(
        x=("location_x", "mean"),
        y=("location_y", "mean"),
        count=("id", "size")
    ).reset_index()
    avg_pos = avg_pos[avg_pos["count"] >= 5]  # minimum touches filter

    # Pair counts
    pairs = completed[completed["pass_recipient"].notna()].copy()
    pairs["pair"] = pairs.apply(lambda r: tuple(sorted([r["player"], r["pass_recipient"]])), axis=1)
    pair_counts = pairs.groupby("pair").size().reset_index(name="count")
    pair_counts = pair_counts[pair_counts["count"] >= 3]

    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color=LINE_COL)
    pitch.draw(ax=ax)

    # Draw edges
    if len(pair_counts):
        max_w = pair_counts["count"].max()
        for _, r in pair_counts.iterrows():
            p1, p2 = r["pair"]
            pos1 = avg_pos[avg_pos["player"] == p1]
            pos2 = avg_pos[avg_pos["player"] == p2]
            if pos1.empty or pos2.empty:
                continue
            lw = 0.5 + 4 * (r["count"] / max_w)
            ax.plot(
                [pos1["x"].values[0], pos2["x"].values[0]],
                [pos1["y"].values[0], pos2["y"].values[0]],
                color=ENG_WHITE, lw=lw, alpha=0.5, zorder=2
            )

    # Draw nodes
    max_c = avg_pos["count"].max() if len(avg_pos) else 1
    for _, r in avg_pos.iterrows():
        size = 150 + 600 * (r["count"] / max_c)
        ax.scatter(r["x"], r["y"], s=size, color=ENG_RED, ec="white", lw=1.2, zorder=3)
        # Shorten name to last name
        name = r["player"].split()[-1] if " " in r["player"] else r["player"]
        ax.annotate(name, (r["x"], r["y"]-3.5), ha="center", va="top",
                     fontsize=7, color="white", fontweight="bold", zorder=4)

    ax.set_title("England – Pass Network (Build-Up Play)", color="white", fontsize=14, fontweight="bold", pad=10)
    fig.patch.set_facecolor(PITCH_BG)
    st.pyplot(fig)

    # Pass direction breakdown
    st.subheader("Pass Direction Breakdown")
    completed["direction"] = "Sideways"
    completed.loc[completed["pass_end_location_x"] - completed["location_x"] > 5, "direction"] = "Forward"
    completed.loc[completed["location_x"] - completed["pass_end_location_x"] > 5, "direction"] = "Backward"
    dir_counts = completed["direction"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    colors_pie = [ENG_RED, ENG_BLUE, LINE_COL]
    ax2.pie(dir_counts, labels=dir_counts.index, autopct="%1.1f%%", colors=colors_pie,
            textprops={"color": "white"})
    ax2.set_title("Pass Directions", color="white", fontsize=11, fontweight="bold")
    fig2.patch.set_facecolor(PITCH_BG)
    st.pyplot(fig2)

# ── VIEW 5: Shot Map & xG ────────────────────────────────────────────────
elif view.startswith("5"):
    st.title("Shot Map & Expected Goals")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shots", len(f_shots))
    col2.metric("Total xG", f"{f_shots['shot_statsbomb_xg'].sum():.2f}")
    goals = f_shots[f_shots["shot_outcome"] == "Goal"]
    col3.metric("Goals", len(goals))
    col4.metric("xG per Shot", f"{f_shots['shot_statsbomb_xg'].mean():.3f}" if len(f_shots) else "0")

    # ── Interactive Plotly shot map on a half-pitch ──
    plotly_fig = go.Figure()

    # Draw pitch lines (half-pitch: x 60→120, y 0→80)
    pitch_shapes = [
        # Outer boundary (half pitch)
        dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color=LINE_COL, width=2)),
        # Penalty area
        dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color=LINE_COL, width=1.5)),
        # 6-yard box
        dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color=LINE_COL, width=1.5)),
        # Centre line
        dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color=LINE_COL, width=1.5)),
        # Penalty spot
        dict(type="circle", x0=39, y0=107, x1=41, y1=109,
             fillcolor=LINE_COL, line=dict(color=LINE_COL, width=0)),
        # Goal line posts
        dict(type="rect", x0=36, y0=120, x1=44, y1=122,
             fillcolor="white", line=dict(color="white", width=1)),
    ]
    # Penalty arc (approximation with a circle shape)
    pitch_shapes.append(
        dict(type="circle", x0=28, y0=96, x1=52, y1=120,
             line=dict(color=LINE_COL, width=1.5, dash="dot"),
             fillcolor="rgba(0,0,0,0)")
    )

    # Separate goals vs non-goals
    shot_data = f_shots.copy()
    shot_data["is_goal"] = shot_data["shot_outcome"] == "Goal"
    shot_data["surname"] = shot_data["player"].apply(
        lambda p: p.split()[-1] if isinstance(p, str) and " " in p else p
    )
    shot_data["marker_size"] = shot_data["shot_statsbomb_xg"] * 40 + 8

    for is_goal, label, color, symbol in [
        (True, "Goal ★", ENG_RED, "star"),
        (False, "No Goal", ENG_BLUE, "circle"),
    ]:
        subset = shot_data[shot_data["is_goal"] == is_goal]
        if subset.empty:
            continue
        plotly_fig.add_trace(go.Scatter(
            x=subset["location_y"],
            y=subset["location_x"],
            mode="markers",
            name=label,
            marker=dict(
                size=subset["marker_size"],
                color=color,
                symbol=symbol,
                line=dict(width=1, color="white"),
                opacity=0.9 if is_goal else 0.65,
            ),
            text=subset.apply(
                lambda r: (
                    f"<b>{r['player']}</b><br>"
                    f"xG: {r['shot_statsbomb_xg']:.3f}<br>"
                    f"Outcome: {r['shot_outcome']}<br>"
                    f"Minute: {int(r['minute'])}'<br>"
                    f"Body Part: {r['shot_body_part']}<br>"
                    f"Technique: {r['shot_technique']}<br>"
                    f"Match: {r['match_label']}"
                ), axis=1
            ),
            hoverinfo="text",
            hoverlabel=dict(bgcolor=PITCH_BG, font_size=13, font_color="white"),
        ))

    plotly_fig.update_layout(
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 82], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
        yaxis=dict(range=[58, 124], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        height=650,
        title=dict(text="England – Shot Map (hover for details, size = xG)",
                   font=dict(color="white", size=15)),
        legend=dict(font=dict(color="white", size=12), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=50, b=10),
        dragmode="pan",
    )

    st.plotly_chart(plotly_fig, use_container_width=True, config={"scrollZoom": True})

    # Detailed shot log table
    st.subheader("Shot Log")
    shot_table = f_shots[["player", "minute", "shot_statsbomb_xg", "shot_outcome",
                           "shot_body_part", "shot_technique", "match_label"]].copy()
    shot_table.columns = ["Player", "Minute", "xG", "Outcome", "Body Part", "Technique", "Match"]
    shot_table = shot_table.sort_values(["Match", "Minute"]).reset_index(drop=True)
    shot_table.index += 1
    st.dataframe(shot_table.style.format({"xG": "{:.3f}"}), use_container_width=True)

    # xG timeline – also interactive with Plotly
    st.subheader("Cumulative xG Timeline (per match)")
    xg_fig = go.Figure()
    for label in selected_matches:
        m = f_shots[f_shots["match_label"] == label].sort_values("minute")
        if m.empty:
            continue
        m = m.copy()
        m["cum_xg"] = m["shot_statsbomb_xg"].cumsum()
        xg_fig.add_trace(go.Scatter(
            x=m["minute"], y=m["cum_xg"],
            mode="lines+markers", name=label,
            line=dict(shape="hv", width=2),
            marker=dict(size=6),
            text=m.apply(
                lambda r: f"<b>{r['player']}</b><br>xG: {r['shot_statsbomb_xg']:.3f}<br>Cum xG: {r['cum_xg']:.3f}<br>Min {int(r['minute'])}'",
                axis=1
            ),
            hoverinfo="text",
        ))
    xg_fig.update_layout(
        xaxis=dict(title="Minute", color="white", gridcolor="#333"),
        yaxis=dict(title="Cumulative xG", color="white", gridcolor="#333"),
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        legend=dict(font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.3)"),
        height=350,
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(xg_fig, use_container_width=True)

# ── VIEW 6: Final Third Entries by Player ──────────────────────────────────
elif view.startswith("6"):
    st.title("Final Third Entries by Player")
    st.caption("Combined pass + carry entries into the final third, broken down by player")

    # Pass entries
    p_mask = (
        (f_passes["location_x"] < FINAL_THIRD_X) &
        (f_passes["pass_end_location_x"] >= FINAL_THIRD_X) &
        (f_passes["pass_outcome"].isna())
    )
    pass_entries = f_passes[p_mask].groupby("player").size().reset_index(name="pass_entries")

    # Carry entries
    c_mask = (
        (f_carries["location_x"] < FINAL_THIRD_X) &
        (f_carries["carry_end_location_x"] >= FINAL_THIRD_X)
    )
    carry_entries = f_carries[c_mask].groupby("player").size().reset_index(name="carry_entries")

    combined = pass_entries.merge(carry_entries, on="player", how="outer").fillna(0)
    combined["total"] = combined["pass_entries"] + combined["carry_entries"]
    combined = combined.sort_values("total", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(combined))
    ax.barh(
        [r["player"] for _, r in combined.iterrows()],
        combined["pass_entries"], color=ENG_RED, label="Pass entries"
    )
    ax.barh(
        [r["player"] for _, r in combined.iterrows()],
        combined["carry_entries"], left=combined["pass_entries"], color=ENG_BLUE, label="Carry entries"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Final Third Entries")
    ax.legend(facecolor=PITCH_BG, edgecolor=LINE_COL, labelcolor="white")
    ax.set_facecolor(PITCH_BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    fig.patch.set_facecolor(PITCH_BG)
    for spine in ax.spines.values():
        spine.set_color(LINE_COL)
    ax.set_title("England – Final Third Entries by Player", color="white",
                  fontsize=13, fontweight="bold", pad=10)
    st.pyplot(fig)

    # Zone breakdown – left / centre / right
    st.subheader("Entry Zones (Left / Centre / Right)")
    ft_p = f_passes[p_mask].copy()
    ft_c = f_carries[c_mask].copy()

    def zone(y):
        if y < 26.67:
            return "Right"
        elif y < 53.33:
            return "Centre"
        else:
            return "Left"

    zones_p = ft_p["pass_end_location_y"].apply(zone).value_counts()
    zones_c = ft_c["carry_end_location_y"].apply(zone).value_counts()
    zone_df = pd.DataFrame({"Pass": zones_p, "Carry": zones_c}).fillna(0)
    zone_df = zone_df.reindex(["Left", "Centre", "Right"])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    zone_df.plot(kind="bar", ax=ax2, color=[ENG_RED, ENG_BLUE])
    ax2.set_ylabel("Entries")
    ax2.set_title("Final Third Entries by Zone", color="white", fontsize=11, fontweight="bold")
    ax2.set_facecolor(PITCH_BG)
    ax2.tick_params(colors="white")
    ax2.yaxis.label.set_color("white")
    ax2.legend(facecolor=PITCH_BG, edgecolor=LINE_COL, labelcolor="white")
    fig2.patch.set_facecolor(PITCH_BG)
    for spine in ax2.spines.values():
        spine.set_color(LINE_COL)
    plt.xticks(rotation=0)
    st.pyplot(fig2)

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Data: StatsBomb Open Data · Euro 2024")
