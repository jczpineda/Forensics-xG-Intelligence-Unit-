"""
build_xt_conceded.py — open-play xT CONCEDED, per team-season and by zone.

The third side of the xT triangle, and the one that was missing:

    xT generated  (build_team_xt.py)  threat WE create with the ball
    xT prevented  (build_spatial.py)  threat we destroy with a defensive action
    xT conceded   (this file)         threat the OPPONENT creates against us

Generated and prevented both measure things we *do*.  Conceded measures what is
done *to* us, which is the only one of the three that answers "where do we get
hurt?".  A side can prevent a lot of threat simply by defending constantly —
build_spatial's own caption warns about that — so xT prevented is a workload
signal as much as a quality one.  xT conceded has no such ambiguity: lower is
better, always.

Method.  Every opponent open-play move (completed pass or carry) is valued on
the SAME fitted surface as everything else — `xt_surface.csv`, never re-fit
here — as V(end) - V(start), and charged to the team it was played against.
The move's starting zone is then MIRRORED into the conceding team's own
attacking orientation, so every map in the app reads the same way: your own
goal on the left, the opponent's on the right.  A hot cell on the left of your
conceded map means opponents are building dangerous attacks from deep inside
your half.

Values are net, exactly as in build_team_xt: an opponent pass that plays the
ball backwards carries negative xT and is netted off.  So a cell can be blue
(opponents lose threat there — you push them backwards) as well as red.

Set pieces, penalties and direct free-kick shots are excluded throughout, so
this is strictly open-play xT and directly comparable to the other two.

Outputs (committed; the JSONs are not):
    team_xt_conceded.csv       one row per (liga, temporada, equipo)
    team_xt_conceded_grid.csv  long form, xT conceded per mirrored start zone

Usage:
    python build_xt_conceded.py validate   # 2025-2026 only, prints sanity checks
    python build_xt_conceded.py build      # all leagues x seasons, writes the CSVs
"""

import os
import sys
import glob
import collections

import numpy as np
import pandas as pd

import build_gk_psxg as bg          # _partidos_dir
import build_team_xt as bx          # surface constants, parse_match, team resolution

_THIS = os.path.dirname(os.path.abspath(__file__))
_SURF = os.path.join(_THIS, "xt_surface.csv")
_OUT_TEAM = os.path.join(_THIS, "team_xt_conceded.csv")
_OUT_GRID = os.path.join(_THIS, "team_xt_conceded_grid.csv")

LEAGUES, SEASONS = bx.LEAGUES, bx.SEASONS
NX, NY, NZ = bx.NX, bx.NY, bx.NZ
CURRENT = "2025-2026"


def load_surface():
    """The fitted 12x8 surface.  This builder never re-fits — one surface across
    all leagues and seasons is what makes the numbers comparable."""
    if not os.path.exists(_SURF):
        raise SystemExit("xt_surface.csv not found - run `python build_team_xt.py build` first.")
    df = pd.read_csv(_SURF)
    V = np.zeros(NZ)
    for _, r in df.iterrows():
        V[int(r["zx"]) * NY + int(r["zy"])] = float(r["value"])
    return V


def _mirror_z(z):
    """Flip a zone index into the other team's attacking orientation.

    Opta coordinates are always oriented so the team in possession attacks
    towards x=100, so an opponent action at zone (zx, zy) sits at
    (NX-1-zx, NY-1-zy) when read from the defending team's point of view.
    """
    zx, zy = z // NY, z % NY
    return (NX - 1 - zx) * NY + (NY - 1 - zy)


def collect(seasons, V):
    """One pass over the JSONs, accumulating conceded xT per team and zone."""
    conceded = {}                                   # key -> NZ vector
    conceded_total = collections.Counter()
    team_matches = collections.Counter()

    id_map_all = bx.load_player_team_map()
    n_files = 0

    for folder in LEAGUES:
        for season in seasons:
            pdir = bg._partidos_dir(folder, season)
            files = sorted(glob.glob(os.path.join(pdir, "*.json")))
            if not files:
                continue
            id_map = id_map_all.get((folder, season), {})
            for fpath in files:
                try:
                    moves, _shots, _losses, contestants, player_team = bx.parse_match(fpath)
                except Exception as ex:
                    print(f"   ! skip {os.path.basename(fpath)}: {ex}")
                    continue
                n_files += 1

                key_of = {}
                for cid, cname in contestants.items():
                    liga, equipo = bx.resolve_team(cid, player_team, id_map, cname, folder)
                    key = (liga, season, equipo)
                    key_of[cid] = key
                    team_matches[key] += 1
                    conceded.setdefault(key, np.zeros(NZ))

                # Who each side was playing against.  Anything other than a
                # clean two-contestant match is unusable for "conceded".
                sides = list(contestants.keys())
                if len(sides) != 2:
                    continue
                opponent_of = {sides[0]: sides[1], sides[1]: sides[0]}

                for sz, ez, cid in moves:
                    ocid = opponent_of.get(cid)
                    if ocid is None:
                        continue
                    k = key_of.get(ocid)             # charged to the DEFENDING team
                    if k is None:
                        continue
                    xt = V[ez] - V[sz]
                    conceded[k][_mirror_z(sz)] += xt
                    conceded_total[k] += xt

            print(f"  {folder} {season}: {len(files)} matches", flush=True)

    print(f"Parsed {n_files} matches -> {len(conceded)} team-seasons")
    return conceded, conceded_total, team_matches


def build_frames(conceded, conceded_total, team_matches):
    trows, grows = [], []
    for key, zones in conceded.items():
        liga, temporada, equipo = key
        matches = team_matches[key]
        if matches == 0:
            continue
        total = float(conceded_total[key])
        trows.append({
            "liga": liga, "temporada": temporada, "equipo": equipo,
            "matches": int(matches),
            "xtc_total": round(total, 4),
            "xtc_per_match": round(total / matches, 4),
        })
        for z in range(NZ):
            if zones[z] != 0:
                grows.append({
                    "liga": liga, "temporada": temporada, "equipo": equipo,
                    "zx": z // NY, "zy": z % NY,
                    "xtc": round(float(zones[z]) / matches, 5),
                })
    return pd.DataFrame(trows), pd.DataFrame(grows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    seasons = [CURRENT] if mode == "validate" else SEASONS
    print(f"Mode: {mode}  |  seasons: {', '.join(seasons)}")

    V = load_surface()
    print(f"Loaded xt_surface.csv  (range {V.min():.5f} - {V.max():.5f})")

    conceded, conceded_total, team_matches = collect(seasons, V)
    team_df, grid_df = build_frames(conceded, conceded_total, team_matches)
    if team_df.empty:
        print("Nothing collected - are the event JSONs present?")
        return

    print("\nBest 12 defences (least open-play xT conceded per match):")
    for _, r in team_df.nsmallest(12, "xtc_per_match").iterrows():
        print(f"  {str(r['equipo'])[:26]:26s} {r['temporada']}  {r['xtc_per_match']:6.3f}")
    print("\nWorst 12 (most conceded per match):")
    for _, r in team_df.nlargest(12, "xtc_per_match").iterrows():
        print(f"  {str(r['equipo'])[:26]:26s} {r['temporada']}  {r['xtc_per_match']:6.3f}")

    # Sanity: conceded should mirror generated in aggregate — every move is
    # created by someone and conceded by someone else, so the league totals of
    # the two must agree to within rounding.
    gen_path = os.path.join(_THIS, "team_xt.csv")
    if os.path.exists(gen_path):
        gen = pd.read_csv(gen_path)
        gen = gen[gen["temporada"].isin(seasons)]
        if not gen.empty:
            g_tot, c_tot = float(gen["xt_total"].sum()), float(team_df["xtc_total"].sum())
            print(f"\nCross-check  sum(xT generated) = {g_tot:.1f}   "
                  f"sum(xT conceded) = {c_tot:.1f}   "
                  f"delta = {abs(g_tot - c_tot) / max(g_tot, 1) * 100:.2f}%")
            print("  (these must agree — every move created is a move conceded)")

    if mode == "build":
        team_df.to_csv(_OUT_TEAM, index=False, encoding="utf-8")
        grid_df.to_csv(_OUT_GRID, index=False, encoding="utf-8")
        print(f"\nWrote {_OUT_TEAM} ({len(team_df)} rows)")
        print(f"Wrote {_OUT_GRID} ({len(grid_df)} rows)")
    else:
        print("\n(validate mode - nothing written)")


if __name__ == "__main__":
    main()
