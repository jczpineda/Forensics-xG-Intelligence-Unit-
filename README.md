# ⚽ Forensics xG: The Intelligence Unit

Interactive football analytics tool for Europe's top 6 leagues (2025-2026 season).

## Features

- **Pre-built Analyses** — 9 ready-made analysis types (top scorers, radar comparisons, etc.)
- **Data Explorer** — Browse, filter, search, and download raw player data
- **Per-90 Stats** — Automatic per-90-minute calculations for key metrics

## Leagues Covered

| League | Country |
|--------|---------|
| Premier League | England |
| LaLiga | Spain |
| Bundesliga | Germany |
| Ligue 1 | France |
| Serie A | Italy |
| Primeira Liga | Portugal |

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.
Official Link: 'https://forensics-xg-intelligence-unit.streamlit.app/'

## Data Structure

The app reads `jugadores_seasonstats.csv` files from each league folder in the parent directory.
Each file contains 130+ statistical columns per player per season.
