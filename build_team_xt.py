"""
build_team_xt.py — Expected Threat (xT) generated, per team-season.

Companion to build_player_xg.py / build_gk_psxg.py.  Where xG values the *shot*,
xT values the *ball movement that manufactured it*: every successful open-play
pass and carry is scored by how much it raised the chance of the possession
ending in a goal.  A sideways pass in your own half is worth ~nothing; a ball
into the central edge-of-box is worth a lot.

Model — Karun Singh style value iteration on a 12x8 grid, fit in-house from the
same Opta event JSONs the xG models use (no borrowed surface):

    V(z) = p_shot(z)*p_goal(z) + p_move(z) * SUM_z' T(z->z') V(z')

Every action from a zone is one of three things: a SHOT, a successful MOVE, or a
LOSS of possession.  The LOSS branch is what makes this work — it is the
absorbing state that drains value.  Omit it and p_shot+p_move==1, value cycles
forever and the surface converges to a flat constant (~0.09 everywhere) with no
pitch structure at all.

A single surface is fit over ALL leagues and seasons pooled, so that xT numbers
are directly comparable across leagues, across seasons and across teams.

An action's xT is V(end zone) - V(start zone).  A team's xT is the sum over its
successful open-play moves; we report the per-match average.  Set pieces
(corners, free kicks, throw-ins, goal kicks), penalties and direct free-kick
shots are excluded throughout, so this is strictly OPEN-PLAY xT.

Outputs (committed; the JSONs are not):
    team_xt.csv       one row per (liga, temporada, equipo) — xt_total, matches, xt_per_match
    team_xt_grid.csv  long form, xT generated per start zone — powers the team heatmap
    xt_surface.csv    the fitted surface itself (96 cells), for reference/validation

Usage:
    python build_team_xt.py validate   # fit on 2025-2026 only, print surface + rankings
    python build_team_xt.py build      # fit on all leagues x seasons, write the CSVs
"""

import os
import sys
import json
import glob
import collections

import numpy as np
import pandas as pd

import build_gk_psxg as bg   # LEAGUES, SEASONS, _partidos_dir, _qval

_THIS = os.path.dirname(os.path.abspath(__file__))
_OUT_TEAM = os.path.join(_THIS, "team_xt.csv")
_OUT_GRID = os.path.join(_THIS, "team_xt_grid.csv")
_OUT_SURF = os.path.join(_THIS, "xt_surface.csv")

LEAGUES, SEASONS = bg.LEAGUES, bg.SEASONS

# Opta event coordinates are 0-100 x 0-100 and ALWAYS oriented so the team in
# possession attacks towards x=100 — no per-team flipping needed.
NX, NY = 12, 8
NZ = NX * NY

SHOT_TYPES = {13, 14, 15, 16}          # miss, woodwork, attempt saved, goal
GOAL_TYPE = 16
# On-ball control events, used to chain a carry between two touches.
TOUCH_TYPES = {1, 3, 7, 12, 13, 14, 15, 16, 42, 49, 50, 61}
# Actions that hand possession straight to the opposition.
LOSS_TYPES = {2, 50, 51}               # offside pass, dispossessed, error

# Qualifiers marking a set-piece delivery (excluded from moves).
SETPIECE_Q = (5, 6, 107, 124)          # free kick, corner, throw-in, goal kick
Q_PENALTY, Q_DIRECT_FK = 9, 26

MAX_CARRY_SEC = 15.0                   # ignore "carries" spanning a stoppage


def _zone(x, y):
    ix = min(max(int(x / 100.0 * NX), 0), NX - 1)
    iy = min(max(int(y / 100.0 * NY), 0), NY - 1)
    return ix * NY + iy                 # flat index 0..95


def _qids(e):
    return {q.get("qualifierId") for q in e.get("qualifier", [])}


def _qget(e, qid):
    for q in e.get("qualifier", []):
        if q.get("qualifierId") == qid:
            return q.get("value")
    return None


def _secs(e):
    return float(e.get("timeMin", 0) or 0) * 60.0 + float(e.get("timeSec", 0) or 0)


def parse_match(path):
    """Parse one match into the pieces the fit and the per-team totals need.

    Returns (moves, shots, losses, contestants, player_team) where
      moves  : list of (start_zone, end_zone, contestantId)
      shots  : list of (zone, is_goal)
      losses : list of zone
      contestants  : {contestantId: name}
      player_team  : {playerId: contestantId}   (for name resolution)
    """
    d = json.load(open(path, encoding="utf-8"))
    mi, ld = d.get("matchInfo", {}), d.get("liveData", {})
    events = ld.get("event", [])

    contestants = {c.get("id"): c.get("name") for c in mi.get("contestant", [])}
    player_team, moves, shots, losses = {}, [], [], []

    for i, e in enumerate(events):
        t = e.get("typeId")
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue
        team, pid = e.get("contestantId"), e.get("playerId")
        if pid and team:
            player_team.setdefault(pid, team)
        z = _zone(float(x), float(y))

        if t in SHOT_TYPES:
            q = _qids(e)
            if Q_PENALTY in q or Q_DIRECT_FK in q:
                continue               # dead-ball shots distort their own zone
            shots.append((z, t == GOAL_TYPE))
            continue

        if t == 1:                      # pass
            q = _qids(e)
            if not q.intersection(SETPIECE_Q):
                if e.get("outcome") == 1:
                    ex, ey = _qget(e, 140), _qget(e, 141)
                    if ex is not None and ey is not None:
                        moves.append((z, _zone(float(ex), float(ey)), team))
                else:
                    losses.append(z)
        elif t == 3:                    # take on
            if e.get("outcome") == 0:
                losses.append(z)
        elif t in LOSS_TYPES:
            losses.append(z)
        elif t == 61 and e.get("outcome") == 0:      # miscontrol
            losses.append(z)

        # ── Carry: the same player's next on-ball event starts somewhere else ──
        if t in TOUCH_TYPES and pid and i + 1 < len(events):
            n = events[i + 1]
            if (n.get("playerId") == pid and n.get("typeId") in TOUCH_TYPES
                    and n.get("x") is not None and n.get("y") is not None
                    and n.get("periodId") == e.get("periodId")
                    and 0 <= _secs(n) - _secs(e) <= MAX_CARRY_SEC):
                sx, sy = float(x), float(y)
                if t == 1 and e.get("outcome") == 1:      # carry starts where the pass ended
                    ex, ey = _qget(e, 140), _qget(e, 141)
                    if ex is not None and ey is not None:
                        sx, sy = float(ex), float(ey)
                nx_, ny_ = float(n["x"]), float(n["y"])
                if abs(nx_ - sx) + abs(ny_ - sy) > 2.0:
                    moves.append((_zone(sx, sy), _zone(nx_, ny_), team))

    return moves, shots, losses, contestants, player_team


# ── Team-name resolution ─────────────────────────────────────────────────────
# Event JSONs say "Arsenal"; the season CSVs say "Arsenal FC".  Rather than
# match on names, resolve through the stable Opta player id: the squad's players
# vote for the CSV `equipo` they belong to.  Falls back to the JSON name.

def load_player_team_map():
    """(liga_folder, temporada) -> {player_id: (liga_csv, equipo)}"""
    out = collections.defaultdict(dict)
    for folder in LEAGUES:
        base = os.path.join(_THIS, folder)
        for fname in ("jugadores_seasonstats.csv", "jugadores_historical.csv"):
            path = os.path.join(base, fname)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                                 usecols=lambda c: c.strip() in ("liga", "temporada", "equipo", "id"))
            except Exception as ex:
                print(f"   ! {folder}/{fname}: {ex}")
                continue
            df.columns = [c.strip() for c in df.columns]
            if not {"temporada", "equipo", "id"}.issubset(df.columns):
                continue
            for temporada, sub in df.groupby(df["temporada"].astype(str)):
                m = out[(folder, temporada)]
                liga_vals = sub["liga"] if "liga" in sub.columns else None
                for i, (pid, equipo) in enumerate(zip(sub["id"], sub["equipo"])):
                    if isinstance(pid, str) and isinstance(equipo, str):
                        liga = liga_vals.iloc[i] if liga_vals is not None else folder
                        m[pid] = (liga, equipo)
    return out


def resolve_team(contestant_id, player_team, id_map, fallback_name, folder):
    """Majority-vote the CSV (liga, equipo) for one contestant."""
    votes = collections.Counter()
    for pid, cid in player_team.items():
        if cid == contestant_id and pid in id_map:
            votes[id_map[pid]] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return (folder, fallback_name)


# ── Collect ──────────────────────────────────────────────────────────────────

def collect(seasons):
    """Single pass over the JSONs, accumulating everything the fit needs."""
    shot_c = np.zeros(NZ); goal_c = np.zeros(NZ)
    move_c = np.zeros(NZ); loss_c = np.zeros(NZ)
    T = np.zeros((NZ, NZ))

    # (liga, temporada, equipo) -> 96x96 move counts; and match counts
    team_T = {}
    team_matches = collections.Counter()

    id_map_all = load_player_team_map()
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
                    moves, shots, losses, contestants, player_team = parse_match(fpath)
                except Exception as ex:
                    print(f"   ! skip {os.path.basename(fpath)}: {ex}")
                    continue
                n_files += 1

                for z, is_goal in shots:
                    shot_c[z] += 1
                    if is_goal:
                        goal_c[z] += 1
                for z in losses:
                    loss_c[z] += 1

                key_of = {}
                for cid, cname in contestants.items():
                    liga, equipo = resolve_team(cid, player_team, id_map, cname, folder)
                    key = (liga, season, equipo)
                    key_of[cid] = key
                    team_matches[key] += 1
                    if key not in team_T:
                        team_T[key] = np.zeros((NZ, NZ), dtype=np.int32)

                for sz, ez, cid in moves:
                    move_c[sz] += 1
                    T[sz, ez] += 1
                    k = key_of.get(cid)
                    if k is not None:
                        team_T[k][sz, ez] += 1

            print(f"  {folder} {season}: {len(files)} matches", flush=True)

    print(f"Parsed {n_files} matches -> {int(move_c.sum())} moves, "
          f"{int(shot_c.sum())} shots, {int(loss_c.sum())} losses")
    return dict(shot_c=shot_c, goal_c=goal_c, move_c=move_c, loss_c=loss_c,
                T=T, team_T=team_T, team_matches=team_matches, n_files=n_files)


# ── Fit the surface ──────────────────────────────────────────────────────────

def fit_surface(c, max_iter=500, tol=1e-10):
    tot = c["shot_c"] + c["move_c"] + c["loss_c"]
    p_shot = np.divide(c["shot_c"], tot, out=np.zeros(NZ), where=tot > 0)
    p_move = np.divide(c["move_c"], tot, out=np.zeros(NZ), where=tot > 0)
    p_goal = np.divide(c["goal_c"], c["shot_c"], out=np.zeros(NZ), where=c["shot_c"] > 0)

    T = c["T"]
    rows = T.sum(axis=1, keepdims=True)
    Tn = np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)

    V = np.zeros(NZ)
    for it in range(max_iter):
        V_new = p_shot * p_goal + p_move * (Tn @ V)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break
    print(f"Surface converged in {it} iterations (delta {delta:.2e}); "
          f"range {V.min():.5f} - {V.max():.5f}")
    return V


def print_surface(V):
    g = V.reshape(NX, NY)
    np.set_printoptions(suppress=True, linewidth=220)
    print("\nxT surface (rows = y bands, cols = x bands own goal -> opponent goal):")
    print(np.round(g.T[::-1], 4))


# ── Aggregate ────────────────────────────────────────────────────────────────

def aggregate(c, V):
    """Per team-season xT totals and the per-start-zone breakdown."""
    delta = V[None, :] - V[:, None]          # delta[sz, ez] = V(ez) - V(sz)
    team_rows, grid_rows = [], []

    for key, M in c["team_T"].items():
        liga, temporada, equipo = key
        matches = c["team_matches"][key]
        if matches == 0:
            continue
        contrib = M * delta                   # xT contributed by each (sz -> ez)
        by_start = contrib.sum(axis=1)        # attribute to where the move started
        total = float(by_start.sum())
        team_rows.append({
            "liga": liga, "temporada": temporada, "equipo": equipo,
            "matches": int(matches),
            "xt_total": round(total, 4),
            "xt_per_match": round(total / matches, 4),
            "moves": int(M.sum()),
        })
        for z in range(NZ):
            if by_start[z] != 0:
                grid_rows.append({
                    "liga": liga, "temporada": temporada, "equipo": equipo,
                    "zx": z // NY, "zy": z % NY,
                    "xt": round(float(by_start[z]) / matches, 5),
                })

    return pd.DataFrame(team_rows), pd.DataFrame(grid_rows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    seasons = ["2025-2026"] if mode == "validate" else SEASONS

    print(f"Mode: {mode}  |  seasons: {', '.join(seasons)}")
    c = collect(seasons)
    V = fit_surface(c)
    print_surface(V)

    team_df, grid_df = aggregate(c, V)
    team_df = team_df.sort_values(["liga", "temporada", "xt_per_match"],
                                  ascending=[True, True, False])

    print(f"\n{len(team_df)} team-seasons.  Top 20 by xT per match:")
    top = team_df.sort_values("xt_per_match", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r['equipo'][:28]:28s} {r['temporada']}  "
              f"{r['xt_per_match']:6.3f}  ({r['matches']} matches)")

    unresolved = [e for e in team_df["equipo"].unique() if e in LEAGUES]
    if unresolved:
        print(f"\n! {len(unresolved)} teams fell back to the JSON name: {unresolved[:5]}")

    if mode == "build":
        team_df.to_csv(_OUT_TEAM, index=False, encoding="utf-8")
        grid_df.to_csv(_OUT_GRID, index=False, encoding="utf-8")
        pd.DataFrame({"zx": [z // NY for z in range(NZ)],
                      "zy": [z % NY for z in range(NZ)],
                      "value": np.round(V, 6)}).to_csv(_OUT_SURF, index=False, encoding="utf-8")
        print(f"\nWrote {_OUT_TEAM}  ({len(team_df)} rows)")
        print(f"Wrote {_OUT_GRID}  ({len(grid_df)} rows)")
        print(f"Wrote {_OUT_SURF}  ({NZ} rows)")
    else:
        print("\n(validate mode — nothing written; run `build` to write the CSVs)")


if __name__ == "__main__":
    main()
