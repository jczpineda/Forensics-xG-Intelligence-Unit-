"""
build_maps.py — touch heat maps + pass sonars, CURRENT SEASON only, for teams
and players.  Kept to the current season so the per-player CSVs stay small.

A fast single pass over the 2025-2026 Opta JSONs accumulates, per team and per
player:
  • touch heat map — count of on-ball actions in each 12×8 pitch zone
  • pass sonar     — open-play passes binned into 16 compass sectors, with the
                     average length and completion rate of each sector

Outputs (committed; the JSONs are not):
  team_heatmap.csv    (liga, temporada, equipo, zx, zy, touches_per_match)
  player_heatmap.csv  (id, temporada, zx, zy, touches)
  team_sonar.csv      (liga, temporada, equipo, sector, passes, avg_dist, completion)
  player_sonar.csv    (id, temporada, sector, passes, avg_dist, completion)

Usage:
    python build_maps.py            # build the current-season CSVs
"""

import os
import json
import glob
import math
import collections

import numpy as np
import pandas as pd

import build_gk_psxg as bg          # _partidos_dir
import build_team_xt as bx          # zone, qualifiers, team resolution

_THIS = os.path.dirname(os.path.abspath(__file__))
SEASON = "2025-2026"
LEAGUES = bx.LEAGUES
NX, NY, NZ = bx.NX, bx.NY, bx.NZ
N_SECTORS = 16                       # compass sectors for the pass sonar
_PITCH_L, _PITCH_W = 105.0, 68.0

# On-ball actions that mark where a player operated (for the heat map).
TOUCH_TYPES = {1, 2, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16,
               41, 42, 44, 49, 50, 61, 74}

_OUT_TEAM_HEAT = os.path.join(_THIS, "team_heatmap.csv")
_OUT_PLAYER_HEAT = os.path.join(_THIS, "player_heatmap.csv")
_OUT_TEAM_SONAR = os.path.join(_THIS, "team_sonar.csv")
_OUT_PLAYER_SONAR = os.path.join(_THIS, "player_sonar.csv")


def parse_match(path):
    """Return (touches, passes, contestants, player_team).
      touches : (zone, playerId, contestantId)
      passes  : (sector, dist_m, completed, playerId, contestantId) open-play only
    """
    d = json.load(open(path, encoding="utf-8"))
    events = d.get("liveData", {}).get("event", [])
    contestants = {c.get("id"): c.get("name")
                   for c in d.get("matchInfo", {}).get("contestant", [])}
    player_team, touches, passes = {}, [], []

    for e in events:
        t = e.get("typeId")
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue
        cid, pid = e.get("contestantId"), e.get("playerId")
        if pid and cid:
            player_team.setdefault(pid, cid)
        if not pid:
            continue
        x, y = float(x), float(y)

        if t in TOUCH_TYPES:
            touches.append((bx._zone(x, y), pid, cid))

        if t == 1:                                   # pass
            if bx._qids(e).intersection(bx.SETPIECE_Q):
                continue                             # open-play sonar only
            ex, ey = bx._qget(e, 140), bx._qget(e, 141)
            if ex is None or ey is None:
                continue
            ex, ey = float(ex), float(ey)
            dx = (ex - x) / 100.0 * _PITCH_L         # metres, +x = toward opp. goal
            dy = (ey - y) / 100.0 * _PITCH_W
            dist = math.hypot(dx, dy)
            if dist < 1.0:
                continue
            ang = math.degrees(math.atan2(dy, dx)) % 360.0
            sector = int(ang // (360.0 / N_SECTORS)) % N_SECTORS
            passes.append((sector, dist, 1 if e.get("outcome") == 1 else 0, pid, cid))

    return touches, passes, contestants, player_team


def collect():
    id_map_all = bx.load_player_team_map()
    team_heat = collections.defaultdict(lambda: np.zeros(NZ))
    player_heat = collections.defaultdict(lambda: np.zeros(NZ))
    team_sonar = collections.defaultdict(lambda: np.zeros((N_SECTORS, 3)))   # count, dist, comp
    player_sonar = collections.defaultdict(lambda: np.zeros((N_SECTORS, 3)))
    team_matches = collections.Counter()
    player_liga = {}                                 # pid -> (liga, equipo)
    n_files = 0

    for folder in LEAGUES:
        pdir = bg._partidos_dir(folder, SEASON)
        files = sorted(glob.glob(os.path.join(pdir, "*.json")))
        if not files:
            continue
        id_map = id_map_all.get((folder, SEASON), {})
        for fpath in files:
            try:
                touches, passes, contestants, player_team = parse_match(fpath)
            except Exception as ex:
                print(f"   ! skip {os.path.basename(fpath)}: {ex}")
                continue
            n_files += 1
            key_of = {}
            for cid, cname in contestants.items():
                liga, equipo = bx.resolve_team(cid, player_team, id_map, cname, folder)
                key = (liga, equipo)
                key_of[cid] = key
                team_matches[key] += 1
            for z, pid, cid in touches:
                player_heat[pid][z] += 1
                k = key_of.get(cid)
                if k:
                    team_heat[k][z] += 1
                if pid in id_map:
                    player_liga.setdefault(pid, id_map[pid])
            for sector, dist, comp, pid, cid in passes:
                ps = player_sonar[pid]
                ps[sector] += (1, dist, comp)
                k = key_of.get(cid)
                if k:
                    team_sonar[k][sector] += (1, dist, comp)
                if pid in id_map:
                    player_liga.setdefault(pid, id_map[pid])
        print(f"  {folder} {SEASON}: {len(files)} matches", flush=True)

    print(f"Parsed {n_files} matches -> {len(player_heat)} players, {len(team_heat)} teams")
    return dict(team_heat=team_heat, player_heat=player_heat, team_sonar=team_sonar,
                player_sonar=player_sonar, team_matches=team_matches, player_liga=player_liga)


def build_frames(c):
    # ── team heat map (per match) ──
    th_rows = []
    for (liga, equipo), zones in c["team_heat"].items():
        m = c["team_matches"][(liga, equipo)] or 1
        for z in range(NZ):
            if zones[z]:
                th_rows.append({"liga": liga, "temporada": SEASON, "equipo": equipo,
                                "zx": z // NY, "zy": z % NY,
                                "touches": round(float(zones[z]) / m, 4)})
    # ── player heat map (season totals) ──
    ph_rows = []
    for pid, zones in c["player_heat"].items():
        liga, equipo = c["player_liga"].get(pid, (None, None))
        for z in range(NZ):
            if zones[z]:
                ph_rows.append({"id": pid, "temporada": SEASON,
                                "zx": z // NY, "zy": z % NY, "touches": int(zones[z])})

    def _sonar_rows(store, key_cols, key_fn):
        rows = []
        for key, arr in store.items():
            base = key_fn(key)
            for s in range(N_SECTORS):
                cnt = arr[s, 0]
                if cnt <= 0:
                    continue
                rows.append({**base, "sector": s, "passes": int(cnt),
                             "avg_dist": round(float(arr[s, 1] / cnt), 2),
                             "completion": round(float(arr[s, 2] / cnt) * 100, 1)})
        return pd.DataFrame(rows)

    team_sonar = _sonar_rows(
        c["team_sonar"], None,
        lambda k: {"liga": k[0], "temporada": SEASON, "equipo": k[1]})
    player_sonar = _sonar_rows(
        c["player_sonar"], None,
        lambda pid: {"id": pid, "temporada": SEASON})

    return (pd.DataFrame(th_rows), pd.DataFrame(ph_rows), team_sonar, player_sonar)


def main():
    print(f"Building current-season ({SEASON}) heat maps + pass sonars…")
    c = collect()
    th, ph, ts, ps = build_frames(c)
    th.to_csv(_OUT_TEAM_HEAT, index=False, encoding="utf-8")
    ph.to_csv(_OUT_PLAYER_HEAT, index=False, encoding="utf-8-sig")
    ts.to_csv(_OUT_TEAM_SONAR, index=False, encoding="utf-8")
    ps.to_csv(_OUT_PLAYER_SONAR, index=False, encoding="utf-8-sig")
    print(f"\nWrote team_heatmap.csv ({len(th)}), player_heatmap.csv ({len(ph)}), "
          f"team_sonar.csv ({len(ts)}), player_sonar.csv ({len(ps)})")


if __name__ == "__main__":
    main()
