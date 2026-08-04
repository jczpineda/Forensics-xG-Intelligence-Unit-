"""
build_eredivisie.py — assemble the repo-format league CSVs for the Eredivisie
(Dutch top flight) from the per-team Opta season-stats files.

The Eredivisie drop ships as raw Opta data (season → equipos/<Team>/
<Team>_jugadores_seasonstats.csv + partidos/*.json), like the other leagues'
source folders — but the app reads ONE league-level file per season group.
This concatenates the per-team files into:

    Eredivisie/jugadores_seasonstats.csv   (current season)
    Eredivisie/jugadores_historical.csv    (past seasons, one 'temporada' each)

Only the seasons the rest of the app covers are built (2020-21 → 2025-26) so the
season selector stays aligned across leagues.  Run locally; commit the two CSVs.

    python build_eredivisie.py
"""

import os
import glob

import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
# Raw Eredivisie drop — sibling of the repo's parent (…/Downloads/…).
_SRC = os.path.join(_THIS, "..", "..", "Netherlands_Eredivisie")
_OUT_DIR = os.path.join(_THIS, "Eredivisie")

CURRENT_SEASON = "2025-2026"
PAST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def _load_season(season):
    """Concatenate every team's season-stats file for one season."""
    pattern = os.path.join(_SRC, season, "equipos", "*", "*_jugadores_seasonstats.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ! no per-team files for {season} ({pattern})")
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, encoding="utf-8-sig", low_memory=False))
        except Exception as ex:
            print(f"    ! skip {os.path.basename(f)}: {ex}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # Normalise the league label to match the app's folder-based display name.
    if "liga" in df.columns:
        df["liga"] = "Eredivisie"
    print(f"  {season}: {len(files)} teams → {len(df)} players")
    return df


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)

    print("Current season…")
    cur = _load_season(CURRENT_SEASON)
    if cur is not None:
        cur.to_csv(os.path.join(_OUT_DIR, "jugadores_seasonstats.csv"),
                   index=False, encoding="utf-8-sig")
        print(f"  wrote jugadores_seasonstats.csv ({len(cur)} rows)")

    print("\nPast seasons…")
    frames = []
    for sea in PAST_SEASONS:
        df = _load_season(sea)
        if df is not None:
            frames.append(df)
    if frames:
        hist = pd.concat(frames, ignore_index=True)
        hist.to_csv(os.path.join(_OUT_DIR, "jugadores_historical.csv"),
                    index=False, encoding="utf-8-sig")
        print(f"\nWrote jugadores_historical.csv "
              f"({len(hist)} rows, {hist['temporada'].nunique()} seasons)")
    print(f"\nOutput → {_OUT_DIR}")


if __name__ == "__main__":
    main()
