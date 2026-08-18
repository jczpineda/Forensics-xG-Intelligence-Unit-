"""
build_pivot_index.py — the Pivot Index: does a squad have a true deep-lying
playmaker (regista / pivote), and who is it?

"Deep-lying playmaker" is not one number.  A midfielder who tops any single
passing metric is usually a *different* archetype, and the whole point of this
index is to keep those apart:

    high volume, no progression          -> recycler / destroyer  (Casemiro-ish)
    high progression, plays high up      -> advanced creator      (a No. 10)
    high volume + progression, from deep -> the pivot playmaker   (Kroos-ish)

So the index scores three axes and combines them CONJUNCTIVELY:

    CONTROL      does the ball go *through* him, and does it survive?
                 own-half passing volume, total pass volume, completion %,
                 losses of possession per 100 touches

    PROGRESSION  when he touches it, does the ball gain ground *and value*?
                 xT generated per 90 and per move (from player_xt.csv),
                 successful long passes, through balls, forward-pass share

    ANCHOR       is he doing it from the pivot position, not the final third?
                 share of his successful passes played in his own half

    PIVOT = sqrt(CONTROL x PROGRESSION) x anchor_gate(ANCHOR)

The geometric mean is deliberate.  A weighted *sum* lets a pure destroyer with
huge volume and no progression score like a playmaker — Tchouameni 2024-25 is
CONTROL 98 / PROGRESSION 37, which a sum flatters and the geometric mean does
not.  ANCHOR is only a gate (never additive) because a holding midfielder who
never leaves his own half tops the anchor metric without being a playmaker; it
can pull a score down for playing too high, but it can never lift one.

Every component is percentile-ranked inside that season's midfielder pool across
all seven leagues, so scores compare across leagues and across seasons.

Unlike the other builders this one reads no event JSONs — it needs only the
committed `jugadores_*.csv` season stats plus `player_xt.csv`, so it runs in
seconds and covers all six seasons.  (An event-level version could add
line-breaking / packing passes and reception-under-pressure; the season-stat
columns cannot express those.)

Output (committed):
    player_pivot.csv   one row per midfielder-season above the minutes floor

Usage:
    python build_pivot_index.py validate   # current season only, prints rankings
    python build_pivot_index.py build      # all seasons, writes player_pivot.csv
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_THIS, "player_pivot.csv")
_PLAYER_XT = os.path.join(_THIS, "player_xt.csv")

LEAGUE_DIRS = ["Bundesliga", "English Premier League", "LaLiga", "Ligue 1",
               "Primeira Liga", "Serie A", "Eredivisie"]
SEASONS = ["2020-2021", "2021-2022", "2022-2023",
           "2023-2024", "2024-2025", "2025-2026"]
CURRENT = "2025-2026"

# Regular-starter floor.  Percentiles are computed over exactly this pool, so
# the app must not re-filter on minutes and expect the ranks to still hold.
MIN_MINUTES = 600

# Columns the index needs.  Coverage is NOT uniform: `Final Third Touches`,
# `Carries` and `Progressive Carries` exist only in jugadores_seasonstats.csv
# (Opta added them later), and Bundesliga/jugadores_historical.csv — rebuilt
# from event JSONs by build_bundesliga_historical.py — has no own-half split at
# all.  Everything below is present in every file that can be scored; rows from
# a file missing one are dropped rather than silently read as zero.
_REQUIRED = [
    "Time Played", "Total Passes", "Total Successful Passes ( Excl Crosses & Corners )",
    "Successful Passes Own Half", "Successful Long Passes", "Forward Passes",
    "Touches", "Total Losses Of Possession",
]


def _read(path):
    """Season-stat CSV reader.  utf-8-sig first — Bundesliga's historical file
    is BOM-prefixed, and reading it as latin-1 silently yields a column named
    'i>>?temporada' instead of 'temporada'."""
    if not os.path.exists(path):
        return None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "temporada" in df.columns:
            return df
    return None


def load_season_stats(seasons):
    """Every league x season of Opta season stats, concatenated."""
    frames, skipped = [], []
    for d in LEAGUE_DIRS:
        for fn in ("jugadores_seasonstats.csv", "jugadores_historical.csv"):
            df = _read(os.path.join(_THIS, d, fn))
            if df is None or df.empty:
                continue
            df["temporada"] = df["temporada"].astype(str).str.strip()
            df = df[df["temporada"].isin(seasons)]
            if df.empty:
                continue
            missing = [c for c in _REQUIRED if c not in df.columns]
            if missing:
                skipped.append((d, fn, len(df), missing[0]))
                continue
            frames.append(df)
    for d, fn, n, col in skipped:
        print(f"  ! skipped {d}/{fn} ({n} rows) - no '{col}' column")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute(seasons):
    """Score every midfielder-season; returns the output frame."""
    df = load_season_stats(seasons)
    if df.empty:
        return df

    xt = pd.read_csv(_PLAYER_XT, encoding="utf-8-sig", low_memory=False)
    xt.columns = [str(c).replace("﻿", "").strip() for c in xt.columns]
    for c in ("id", "temporada"):
        xt[c] = xt[c].astype(str).str.strip()
    df["id"] = df["id"].astype(str).str.strip()
    df = df.merge(xt[["id", "temporada", "xt_gen", "moves"]],
                  on=["id", "temporada"], how="left")

    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    mins = pd.to_numeric(df["Time Played"], errors="coerce")
    p90 = mins / 90.0
    passes = num("Total Passes").replace(0, np.nan)
    succ = num("Total Successful Passes ( Excl Crosses & Corners )").replace(0, np.nan)
    touches = num("Touches").replace(0, np.nan)
    # Through balls only exists in the richer files; absent is genuinely 0 here.
    through = pd.to_numeric(df.get("Through balls", 0), errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "id": df["id"], "nombre": df["nombre"], "equipo": df["equipo"],
        "liga": df["liga"], "temporada": df["temporada"],
        "posicion": df["posicion"], "minutes": mins,
        # CONTROL inputs
        "deep_pass90": num("Successful Passes Own Half") / p90,
        "pass90": num("Total Passes") / p90,
        "cmp_pct": succ / passes,
        "loss_p100": num("Total Losses Of Possession") / touches * 100.0,
        # PROGRESSION inputs
        "xt90": df["xt_gen"] / p90,
        "xt_per_move": df["xt_gen"] / df["moves"].replace(0, np.nan),
        "long90": num("Successful Long Passes") / p90,
        "through90": through / p90,
        "fwd_share": num("Forward Passes") / passes,
        # ANCHOR input
        "ownhalf_share": num("Successful Passes Own Half") / succ,
    })
    out = out[(out["posicion"] == "Midfielder") & (out["minutes"] >= MIN_MINUTES)]
    out = out[out["xt90"].notna()].copy()      # no xT row => nothing to progress with

    # -- percentiles, within season, across all leagues -------------------
    grp = out.groupby("temporada")
    pct = lambda col: grp[col].rank(pct=True) * 100.0
    for c in ("deep_pass90", "pass90", "cmp_pct", "xt90", "xt_per_move",
              "long90", "through90", "fwd_share", "ownhalf_share"):
        out["p_" + c] = pct(c)
    out["p_loss"] = 100.0 - pct("loss_p100")   # fewer losses is better

    out["CONTROL"] = (0.35 * out.p_deep_pass90 + 0.25 * out.p_pass90
                      + 0.20 * out.p_cmp_pct + 0.20 * out.p_loss)
    out["PROGRESSION"] = (0.40 * out.p_xt90 + 0.20 * out.p_xt_per_move
                          + 0.20 * out.p_long90 + 0.10 * out.p_through90
                          + 0.10 * out.p_fwd_share)
    out["ANCHOR"] = out.p_ownhalf_share

    # Gate, not a term: full credit from the 40th percentile of own-half share
    # upward, tapering to 0.70 for a midfielder who lives in the final third.
    # It can only ever pull a score down, and never by more than 30%.
    gate = np.clip(0.70 + 0.30 * out["ANCHOR"] / 40.0, 0.70, 1.0)
    out["PIVOT"] = np.sqrt(out.CONTROL * out.PROGRESSION) * gate

    out["pivot_rank"] = out.groupby("temporada")["PIVOT"].rank(
        ascending=False, method="min").astype(int)
    return out.sort_values(["temporada", "PIVOT"], ascending=[True, False])


def _fmt(rows):
    for _, r in rows.iterrows():
        print(f"  {str(r['nombre'])[:20]:20s} {str(r['equipo'])[:24]:24s} "
              f"PIVOT {r['PIVOT']:5.1f}   C {r['CONTROL']:5.1f}  "
              f"P {r['PROGRESSION']:5.1f}  A {r['ANCHOR']:5.1f}   "
              f"long/90 {r['long90']:4.1f}  xT/90 {r['xt90']:5.3f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    seasons = [CURRENT] if mode == "validate" else SEASONS
    print(f"Mode: {mode}  |  seasons: {', '.join(seasons)}")

    out = compute(seasons)
    if out.empty:
        print("No rows scored - are the season-stat CSVs and player_xt.csv present?")
        return
    print(f"Scored {len(out)} midfielder-seasons "
          f"(>= {MIN_MINUTES} min, {out['temporada'].nunique()} season(s))")

    cur = out[out["temporada"] == out["temporada"].max()]
    print(f"\nTop 15 Pivot Index, {cur['temporada'].iloc[0]}:")
    _fmt(cur.head(15))

    print("\nArchetype check - these must NOT rank as pivots")
    print("(high control but no progression, or progression from too high up):")
    for name in ("Tchouam", "ler", "Valverde", "Brahim"):
        hit = cur[cur["nombre"].str.contains(name, na=False, regex=False)
                  & cur["equipo"].str.contains("Real Madrid", na=False)]
        if not hit.empty:
            _fmt(hit.head(1))

    if mode == "build":
        out.to_csv(_OUT, index=False, encoding="utf-8-sig")
        print(f"\nWrote {_OUT} ({len(out)} rows)")
    else:
        print("\n(validate mode - nothing written)")


if __name__ == "__main__":
    main()
