"""
build_player_xg.py — PRE-shot Expected Goals (xG) per player, for finishing.

Companion to build_gk_psxg.py.  Where PSxG conditions on *where the ball was
placed* (post-shot, for keepers), xG measures the *quality of the chance* from
pre-shot context only — deliberately excluding goalmouth placement so that
npG − xG (finishing) reflects the striker's finishing skill rather than leaking
the finish into the baseline.

Model: a logistic regression over every non-penalty shot ATTEMPT (on or off
target, including blocked) across all leagues/seasons, predicting goal vs
no-goal from distance, shooting angle, header, big-chance, fast-break and
from-corner.  No q102/q103 placement.  Penalties handled at their empirical
rate.  Calibrated so Σ xG = Σ goals.  Each shot's predicted probability is its
xG; we attribute it to the SHOOTER.

Reuses the shot-parsing helpers and numpy logistic fit from build_gk_psxg.py.

Output: player_xg.csv at the repo root (committed; JSONs are not), one row per
(id, temporada, liga) — joined in-app by the stable Opta player id.

Usage:
    python build_player_xg.py validate   # fit + calibration + top/bottom finishers (25-26)
    python build_player_xg.py build        # fit + write player_xg.csv (all 6 seasons)
"""

import os
import sys
import json
import glob
import math
import collections

import numpy as np
import pandas as pd

import build_gk_psxg as bg   # _qmap, _evt_time, fit_logistic, predict_proba, constants

_THIS = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_THIS, "player_xg.csv")

PITCH_L, PITCH_W, GOAL_HALF_W = bg.PITCH_L, bg.PITCH_W, bg.GOAL_HALF_W
SHOT_TYPES = {13, 14, 15, 16}   # miss, woodwork, attempt-saved, goal — all are attempts

FEATURE_NAMES = ["dist", "angle", "header", "big_chance", "fast_break", "from_corner"]


def _preshot_features(x, y, is_header, is_big, fast_break, from_corner):
    """Pre-shot feature vector — NO goalmouth placement (that's the finish)."""
    sx, sy = x / 100.0 * PITCH_L, y / 100.0 * PITCH_W
    dx, dy = PITCH_L - sx, PITCH_W / 2.0 - sy
    dist = math.hypot(dx, dy)

    p1y = PITCH_W / 2.0 - GOAL_HALF_W
    p2y = PITCH_W / 2.0 + GOAL_HALF_W
    a = math.hypot(sx - PITCH_L, sy - p1y)
    b = math.hypot(sx - PITCH_L, sy - p2y)
    c = 2.0 * GOAL_HALF_W
    denom = 2.0 * a * b
    cos_ang = (a * a + b * b - c * c) / denom if denom > 0 else 1.0
    angle = math.acos(max(-1.0, min(1.0, cos_ang)))

    return [dist, angle,
            1.0 if is_header else 0.0,
            1.0 if is_big else 0.0,
            1.0 if fast_break else 0.0,
            1.0 if from_corner else 0.0]


def parse_match(path):
    """Return a list of shot records attributed to the shooter."""
    d = json.load(open(path, encoding="utf-8"))
    events = d.get("liveData", {}).get("event", [])
    cname = {c.get("id"): c.get("name")
             for c in d.get("matchInfo", {}).get("contestant", [])}

    shots = []
    for e in events:
        t = e.get("typeId")
        if t not in SHOT_TYPES:
            continue
        ql = bg._qmap(e)
        if 28 in ql:                          # own goal — not the shooter's attempt
            continue
        shooter = e.get("playerId")
        if not shooter:
            continue
        is_pen = 9 in ql
        is_goal = (t == 16)
        rec = {"shooter": shooter, "name": e.get("playerName"),
               "team": cname.get(e.get("contestantId")),
               "is_pen": is_pen, "is_goal": is_goal}
        if not is_pen:
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            rec["feats"] = _preshot_features(
                float(x), float(y),
                is_header=(15 in ql), is_big=(214 in ql),
                fast_break=(23 in ql), from_corner=(25 in ql))
        shots.append(rec)
    return shots


def collect(seasons):
    all_shots, meta, n_files = [], {}, 0
    for liga in bg.LEAGUES:
        for season in seasons:
            pdir = os.path.join(bg._SRC_ROOT, liga, season, "partidos")
            files = sorted(glob.glob(os.path.join(pdir, "*.json")))
            if not files:
                continue
            for fpath in files:
                try:
                    shots = parse_match(fpath)
                except Exception as ex:
                    print(f"   ! skip {os.path.basename(fpath)}: {ex}")
                    continue
                n_files += 1
                for s in shots:
                    s["season"] = season
                    s["liga"] = liga
                    if s["name"] and s["shooter"] not in meta:
                        meta[s["shooter"]] = (s["name"], s["team"])
                all_shots.extend(shots)
            print(f"  {liga} {season}: {len(files)} matches")
    print(f"Parsed {n_files} matches -> {len(all_shots)} shot attempts")
    return all_shots, meta


def build_model(all_shots):
    np_shots = [s for s in all_shots if not s["is_pen"] and "feats" in s]
    X = np.array([s["feats"] for s in np_shots], dtype=float)
    y = np.array([1.0 if s["is_goal"] else 0.0 for s in np_shots])
    beta, mean, std = bg.fit_logistic(X, y)

    pen_shots = [s for s in all_shots if s["is_pen"]]
    pen_goals = sum(1 for s in pen_shots if s["is_goal"])
    pen_rate = (pen_goals / len(pen_shots)) if pen_shots else 0.79
    return {"beta": beta, "mean": mean, "std": std, "pen_rate": pen_rate,
            "X": X, "y": y, "n_pen": len(pen_shots), "pen_goals": pen_goals}


def shot_xg(model, s):
    if s["is_pen"]:
        return model["pen_rate"]
    return float(bg.predict_proba(model["beta"], model["mean"], model["std"],
                                  np.array(s["feats"]))[0])


def aggregate(all_shots, model):
    agg = collections.defaultdict(
        lambda: {"xg": 0.0, "npxg": 0.0, "np_goals": 0, "shots": 0, "np_shots": 0})
    for s in all_shots:
        sh = s["shooter"]
        if not sh:
            continue
        a = agg[(sh, s["season"], s["liga"])]
        p = shot_xg(model, s)
        a["xg"] += p
        a["shots"] += 1
        if not s["is_pen"]:
            a["npxg"] += p
            a["np_shots"] += 1
            a["np_goals"] += 1 if s["is_goal"] else 0
    return agg


def _report_calibration(model):
    beta, mean, std, X, y = (model["beta"], model["mean"], model["std"],
                             model["X"], model["y"])
    p = bg.predict_proba(beta, mean, std, X)
    eps = 1e-12
    logloss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    print("\n-- xG model fit (non-penalty shot attempts) --")
    print(f"  shots={len(y)}  goals={int(y.sum())}  conv={y.mean():.3f}")
    print(f"  sum xG={p.sum():.1f}  vs  sum goals={y.sum():.0f}  "
          f"(ratio {p.sum()/max(y.sum(),1):.3f})  log-loss={logloss:.4f}")
    print(f"  intercept={beta[0]:+.3f}")
    for nm, b in zip(FEATURE_NAMES, beta[1:]):
        print(f"    {nm:11s} = {b:+.3f}")
    print(f"  penalties: {model['n_pen']} taken, {model['pen_goals']} scored "
          f"-> rate {model['pen_rate']:.3f}")
    print("\n  calibration by predicted decile (pred vs actual):")
    order = np.argsort(p)
    for d in range(10):
        lo, hi = d * len(p) // 10, (d + 1) * len(p) // 10
        idx = order[lo:hi]
        print(f"    decile {d+1:2d}: pred={p[idx].mean():.3f}  "
              f"actual={y[idx].mean():.3f}  n={len(idx)}")


def validate():
    seasons = ["2025-2026"]
    print(f"Validating on {seasons[0]}...")
    shots, meta = collect(seasons)
    model = build_model(shots)
    _report_calibration(model)
    agg = aggregate(shots, model)
    rows = []
    for (sh, season, liga), a in agg.items():
        if a["np_shots"] < 25:
            continue
        nm, team = meta.get(sh, (sh, "?"))
        rows.append((nm, team, liga, a["npxg"], a["np_goals"],
                     a["np_goals"] - a["npxg"], a["np_shots"]))
    df = pd.DataFrame(rows, columns=["name", "team", "liga", "npxG", "npG",
                                     "npG-xG", "np_shots"]).sort_values(
        "npG-xG", ascending=False)
    pd.set_option("display.width", 170)
    fmt = {"npxG": "{:.1f}".format, "npG-xG": "{:+.1f}".format}
    print("\n-- Best finishers 2025-26 (>=25 non-pen shots) --")
    print(df.head(15).to_string(index=False, formatters=fmt))
    print("\n-- Worst (overperformers vs wasteful) --")
    print(df.tail(10).to_string(index=False, formatters=fmt))


def build():
    shots, meta = collect(bg.SEASONS)
    model = build_model(shots)
    _report_calibration(model)
    agg = aggregate(shots, model)
    rows = []
    for (sh, season, liga), a in agg.items():
        rows.append({
            "id": sh, "temporada": season, "liga": liga,
            "xG": round(a["xg"], 2),
            "npxG": round(a["npxg"], 2),
            "npG-xG": round(a["np_goals"] - a["npxg"], 2),
            "npxg_goals": a["np_goals"],
            "shots": a["shots"],
            "np_shots": a["np_shots"],
        })
    out = pd.DataFrame(rows).sort_values(["temporada", "liga", "npxG"],
                                         ascending=[False, True, False])
    out.to_csv(_OUT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(out)} player-seasons -> {_OUT} "
          f"({out['temporada'].nunique()} seasons, {out['liga'].nunique()} leagues)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "build":
        build()
    else:
        validate()
