"""
build_spatial.py — event-level spatial features that REUSE the fitted xT surface.

Companion to build_team_xt.py (which fits the surface and does team xT *generated*).
This builder loads `xt_surface.csv` and never re-fits; it does a single pass over
the Opta event JSONs and writes the pieces the Team Profile and Player Profile
need for the defensive / spatial views.

Phase 1 (this file) — Expected Threat, the other side of it:
  • player xT GENERATED  — each open-play move's V(end)-V(start), by the player
  • player xT PREVENTED   — each ball-winning defensive action valued at the xT
                            the opponent held there, V(mirror(x,y))
  • team xT PREVENTED, by pitch zone (where on our own half we snuff out threat)

Outputs (committed; the JSONs are not):
  player_xt.csv               (id, temporada, liga, xt_gen, xt_prevented, moves, def_actions)
  team_xt_prevented.csv       (liga, temporada, equipo, matches, xtp_total, xtp_per_match)
  team_xt_prevented_grid.csv  long form, xT prevented per zone → powers the team map

Join keys match the rest of the app: (id, temporada) for players, and the
player-id-vote team resolution from build_team_xt for teams.

Usage:
    python build_spatial.py validate   # 2025-2026 only, prints sanity checks
    python build_spatial.py build       # all leagues × seasons, writes the CSVs
"""

import os
import sys
import json
import glob
import collections

import numpy as np
import pandas as pd

import build_gk_psxg as bg          # _partidos_dir
import build_team_xt as bx          # surface constants, zone, parse helpers

_THIS = os.path.dirname(os.path.abspath(__file__))
_SURF = os.path.join(_THIS, "xt_surface.csv")
_OUT_PLAYER = os.path.join(_THIS, "player_xt.csv")
_OUT_TEAMP = os.path.join(_THIS, "team_xt_prevented.csv")
_OUT_TEAMP_GRID = os.path.join(_THIS, "team_xt_prevented_grid.csv")

LEAGUES, SEASONS = bx.LEAGUES, bx.SEASONS
NX, NY, NZ = bx.NX, bx.NY, bx.NZ

# Ball-winning / threat-denying defensive actions.  Tackle (7) requires a won
# outcome; the rest are possession regains or a cleared danger.
_DEF_TYPES = {8, 12, 49, 74}        # interception, clearance, recovery, blocked pass
_TACKLE = 7


def load_surface():
    if not os.path.exists(_SURF):
        raise SystemExit("xt_surface.csv not found — run `python build_team_xt.py build` first.")
    df = pd.read_csv(_SURF)
    V = np.zeros(NZ)
    for _, r in df.iterrows():
        V[int(r["zx"]) * NY + int(r["zy"])] = float(r["value"])
    return V


def _mirror_zone(x, y):
    """Zone of a point read in the OPPONENT's attacking orientation, so a
    defensive action deep in our own third maps to a high-threat attacking zone."""
    return bx._zone(100.0 - x, 100.0 - y)


def parse_match(path):
    """One pass over a match → the player moves and defensive actions we value.

    Returns (moves, def_acts, contestants, player_team):
      moves      : (start_zone, end_zone, playerId, contestantId) open-play moves
      def_acts   : (zone, mirror_zone, playerId, contestantId) ball-winning actions
      contestants: {contestantId: name}
      player_team: {playerId: contestantId}
    """
    d = json.load(open(path, encoding="utf-8"))
    events = d.get("liveData", {}).get("event", [])
    contestants = {c.get("id"): c.get("name")
                   for c in d.get("matchInfo", {}).get("contestant", [])}
    player_team, moves, def_acts = {}, [], []

    for i, e in enumerate(events):
        t = e.get("typeId")
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue
        cid, pid = e.get("contestantId"), e.get("playerId")
        if pid and cid:
            player_team.setdefault(pid, cid)
        x, y = float(x), float(y)
        z = bx._zone(x, y)
        q = bx._qids(e)

        # ── Open-play moves (completed pass + carry), attributed to the player ──
        if t == 1 and pid and not q.intersection(bx.SETPIECE_Q) and e.get("outcome") == 1:
            ex, ey = bx._qget(e, 140), bx._qget(e, 141)
            if ex is not None and ey is not None:
                moves.append((z, bx._zone(float(ex), float(ey)), pid, cid))
        # Carry: same player's next on-ball touch starts elsewhere (mirrors build_team_xt).
        if t in bx.TOUCH_TYPES and pid and i + 1 < len(events):
            n = events[i + 1]
            if (n.get("playerId") == pid and n.get("typeId") in bx.TOUCH_TYPES
                    and n.get("x") is not None and n.get("y") is not None
                    and n.get("periodId") == e.get("periodId")
                    and 0 <= bx._secs(n) - bx._secs(e) <= bx.MAX_CARRY_SEC):
                sx, sy = x, y
                if t == 1 and e.get("outcome") == 1:
                    ex, ey = bx._qget(e, 140), bx._qget(e, 141)
                    if ex is not None and ey is not None:
                        sx, sy = float(ex), float(ey)
                nx_, ny_ = float(n["x"]), float(n["y"])
                if abs(nx_ - sx) + abs(ny_ - sy) > 2.0:
                    moves.append((bx._zone(sx, sy), bx._zone(nx_, ny_), pid, cid))

        # ── Ball-winning defensive actions ──
        if pid and (t in _DEF_TYPES or (t == _TACKLE and e.get("outcome") == 1)):
            def_acts.append((z, _mirror_zone(x, y), pid, cid))

    return moves, def_acts, contestants, player_team


def collect(seasons, V):
    """Single pass → per-player and per-team xT (generated & prevented)."""
    delta = V[None, :] - V[:, None]          # delta[sz, ez] = V(ez) - V(sz)

    # players: pid -> {temporada -> counters}
    player = collections.defaultdict(lambda: {"gen": 0.0, "prev": 0.0,
                                              "moves": 0, "def": 0,
                                              "liga": None, "equipo": None})
    team_prev = {}                           # (liga, temporada, equipo) -> 96 zone sums
    team_prev_total = collections.Counter()
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
                    moves, def_acts, contestants, player_team = parse_match(fpath)
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
                    team_prev.setdefault(key, np.zeros(NZ))

                for sz, ez, pid, cid in moves:
                    p = player[(pid, season)]
                    p["gen"] += delta[sz, ez]
                    p["moves"] += 1
                    if p["liga"] is None and pid in id_map:
                        p["liga"], p["equipo"] = id_map[pid]

                for z, mz, pid, cid in def_acts:
                    val = V[mz]
                    p = player[(pid, season)]
                    p["prev"] += val
                    p["def"] += 1
                    if p["liga"] is None and pid in id_map:
                        p["liga"], p["equipo"] = id_map[pid]
                    k = key_of.get(cid)
                    if k is not None:
                        team_prev[k][z] += val
                        team_prev_total[k] += val

            print(f"  {folder} {season}: {len(files)} matches", flush=True)

    print(f"Parsed {n_files} matches -> {len(player)} player-seasons, "
          f"{len(team_prev)} team-seasons")
    return player, team_prev, team_prev_total, team_matches, n_files


def build_frames(player, team_prev, team_prev_total, team_matches):
    prows = []
    for (pid, season), p in player.items():
        prows.append({
            "id": pid, "temporada": season,
            "liga": p["liga"], "equipo": p["equipo"],
            "xt_gen": round(p["gen"], 4),
            "xt_prevented": round(p["prev"], 4),
            "moves": p["moves"], "def_actions": p["def"],
        })
    player_df = pd.DataFrame(prows)

    trows, grows = [], []
    for key, zones in team_prev.items():
        liga, temporada, equipo = key
        matches = team_matches[key]
        if matches == 0:
            continue
        total = float(team_prev_total[key])
        trows.append({
            "liga": liga, "temporada": temporada, "equipo": equipo,
            "matches": int(matches),
            "xtp_total": round(total, 4),
            "xtp_per_match": round(total / matches, 4),
        })
        for z in range(NZ):
            if zones[z] != 0:
                grows.append({
                    "liga": liga, "temporada": temporada, "equipo": equipo,
                    "zx": z // NY, "zy": z % NY,
                    "xtp": round(float(zones[z]) / matches, 5),
                })
    return player_df, pd.DataFrame(trows), pd.DataFrame(grows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    seasons = ["2025-2026"] if mode == "validate" else SEASONS
    print(f"Mode: {mode}  |  seasons: {', '.join(seasons)}")

    V = load_surface()
    print(f"Loaded xt_surface.csv  (range {V.min():.5f} - {V.max():.5f})")

    player, team_prev, team_prev_total, team_matches, _ = collect(seasons, V)
    player_df, teamp_df, grid_df = build_frames(player, team_prev, team_prev_total, team_matches)

    print("\nTop 12 xT-preventing team-seasons (per match):")
    for _, r in teamp_df.sort_values("xtp_per_match", ascending=False).head(12).iterrows():
        print(f"  {r['equipo'][:26]:26s} {r['temporada']}  {r['xtp_per_match']:6.3f}")
    print("\nTop 12 players by xT generated:")
    for _, r in player_df.sort_values("xt_gen", ascending=False).head(12).iterrows():
        print(f"  {str(r['equipo'])[:22]:22s} {r['temporada']}  gen={r['xt_gen']:6.2f}  prev={r['xt_prevented']:6.2f}")
    print("\nTop 12 players by xT prevented:")
    for _, r in player_df.sort_values("xt_prevented", ascending=False).head(12).iterrows():
        print(f"  {str(r['equipo'])[:22]:22s} {r['temporada']}  prev={r['xt_prevented']:6.2f}  ({r['def_actions']} actions)")

    if mode == "build":
        player_df.to_csv(_OUT_PLAYER, index=False, encoding="utf-8-sig")
        teamp_df.to_csv(_OUT_TEAMP, index=False, encoding="utf-8")
        grid_df.to_csv(_OUT_TEAMP_GRID, index=False, encoding="utf-8")
        print(f"\nWrote {_OUT_PLAYER} ({len(player_df)} rows)")
        print(f"Wrote {_OUT_TEAMP} ({len(teamp_df)} rows)")
        print(f"Wrote {_OUT_TEAMP_GRID} ({len(grid_df)} rows)")
    else:
        print("\n(validate mode — nothing written)")


if __name__ == "__main__":
    main()
