# ⚽ Forensics xG: The Intelligence Unit

Interactive football analytics and scouting tool for Europe's top 7 leagues, with
**custom-built xG, post-shot xG and expected-assists models** computed from raw
Opta event data — plus role-based player grading, six seasons of history, market
values and more.

**Live app:** https://forensics-xg-intelligence-unit.streamlit.app/

## Highlights

- **Custom expected-goals models** (not borrowed — fit in-house from ~100k shots):
  - **xG** (pre-shot) and **npxG** for every shooter, with **Finishing (npG − xG)** to separate finishing skill from chance quality.
  - **xA** (expected assists) — the xG of the chances a player creates — plus **npxG + xA** for total goal involvement.
  - **PSxG** (post-shot xG) for goalkeepers, with **PSxG+/- (goals prevented)** and **Saveable Goals** (soft goals conceded) to expose error-proneness.
  - **xT** (expected threat) per team — values the *build-up* rather than the shot: every successful open-play pass and carry scored by how much it raised the chance the possession ends in a goal. Solved by value iteration on a 12×8 grid, one surface across all leagues and seasons so teams compare directly.
- **Pivot Index** — does a squad actually have a *true* deep-lying playmaker? Scores every midfielder on three axes (**Control**, **Progression**, **Anchor**) and combines them *conjunctively*, so a high-volume recycler and an advanced creator can't be mistaken for a regista. Rendered as a midfield-archetype quadrant on Team Profile.
- **Player grading** — role-aware grades (S+ → F) across position-specific categories, shown as pizza charts and attribute breakdowns. PSxG+/- is the core of the goalkeeper grade; xG/xA drive attacking grades, with regression-to-mean shrinkage so small samples don't swing results.
- **Automatic role classification** — each player is typed into a role (e.g. Shot-Stopper, Ball-Playing GK, Prolific Striker, Advanced Playmaker) and graded against the right peers and KPIs.
- **Six seasons** (2020-21 → 2025-26) with cross-season comparison and a trajectory-based **Potential Grading** that tracks each player across seasons by stable Opta id.
- **Possession adjustment (Padj)** toggle and **Per-90** mode throughout.
- **Market values & salaries, preferred foot, and player photos**, plus accent-insensitive search ("Sesko" → "Šeško").

## App sections

| Tab | What it does |
|-----|--------------|
| 🔬 **Player Lab** | Sortable, gradable table of all players with role/foot/strength filters |
| 🪪 **Player Profile** | Full per-player report: grades, pizza chart, xG/xA & PSxG statlines, scouting report, market value, trajectory |
| 🧤 **GK Analysis** | Goalkeeper shot-stopping rankings powered by the PSxG model |
| 🌟 **Potential Grading** | Trajectory-driven potential, blending form momentum with the age curve |
| ⚔️ **Player Comparison** | Find Similar Players · head-to-head Compare · Cross-Season Compare |
| 🏟️ **Team Profile / Comparison** | Team-level aggregates, matchups, **xT generated** — league ranking plus a pitch map of where each team creates threat — and the **Pivot Index** quadrant: whether the squad actually contains a deep-lying playmaker |
| 🔍 **Data Explorer** | Browse, filter, search and download the raw data |

## Leagues & seasons

| League | Country |
|--------|---------|
| Premier League | England |
| LaLiga | Spain |
| Bundesliga | Germany |
| Ligue 1 | France |
| Serie A | Italy |
| Primeira Liga | Portugal |
| Eredivisie | Netherlands |

Seasons **2020-21 → 2025-26** (financials, footedness and photos apply to the current season).

## Setup

```bash
pip install -r requirements.txt   # streamlit, pandas, numpy, plotly, requests, beautifulsoup4
streamlit run app.py              # opens at http://localhost:8501
```

## Data & models

The app reads `jugadores_seasonstats.csv` (current season) and `jugadores_historical.csv`
(past seasons) from each league folder — Opta season stats, 130+ columns per player.

The xG/PSxG/xA models and enrichment data are **pre-computed locally and committed as
CSVs**, because the raw Opta event JSONs they're built from are too large for the repo.
The app just reads the CSVs; re-run a builder (locally, with the event JSONs present) to
refresh:

| Builder | Produces | Purpose |
|---------|----------|---------|
| `build_player_xg.py` | `player_xg.csv` | pre-shot xG, npxG, finishing, **xA** per player-season |
| `build_gk_psxg.py` | `gk_psxg.csv` | post-shot xG, PSxG+/-, saveable goals per keeper-season |
| `build_team_xt.py` | `team_xt.csv`, `team_xt_grid.csv`, `xt_surface.csv` | open-play **xT** per team-season, its per-zone breakdown, and the fitted surface |
| `build_footedness.py` | `player_footedness.csv` | preferred foot (Transfermarkt) |
| `build_financials_csv.py` | `Player Financials/player_financials.csv` | market value & salary |
| `build_photos_csv.py` | `player_photos.csv` | player cutout photos (`refresh "Name1;Name2"` to re-fetch) |
| `build_bundesliga_historical.py` | `Bundesliga/jugadores_historical.csv` | rebuilt top-flight history from event JSONs |
| `build_eredivisie.py` | `Eredivisie/jugadores_*.csv` | assemble the Dutch league CSVs from per-team Opta files |
| `build_team_xt.py` | `team_xt*.csv`, `xt_surface.csv` | fit the xT surface + team xT *generated* by zone |
| `build_spatial.py` | `player_xt.csv`, `team_xt_prevented*.csv` | player xT (gen & prevented) + team xT *prevented*, reusing the surface |
| `build_maps.py` | `team_/player_heatmap.csv`, `team_/player_sonar.csv` | current-season touch heat maps + pass sonars |
| `build_pivot_index.py` | `player_pivot.csv` | **Pivot Index** — deep-lying playmaker scoring per midfielder-season (CSV-only, seconds to run) |

Both xG models are logistic fits over every shot (calibrated so Σ xG ≈ Σ goals); see each
script's docstring for features and validation (`python build_player_xg.py validate`).
