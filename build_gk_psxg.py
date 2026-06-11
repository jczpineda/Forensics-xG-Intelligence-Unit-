"""
build_gk_psxg.py — TRUE post-shot expected goals (PSxG) for goalkeepers.

The app's old PSxG was a season-aggregate approximation: it counted on-target
shots by inside/outside box and applied four fixed weights.  It never saw the
individual shots, so it could not account for *where in the goal* a shot was
placed — the defining input of a post-shot model.

This script fits a real shot-level post-shot xG model from the raw Opta F24
match-event JSONs (the same `partidos/` feeds used by build_bundesliga_historical.py,
living in the sibling `../../Forensics xG Opta Data/<League>/<season>/partidos/`).

Model: pooling every on-target non-penalty shot across all leagues/seasons, we fit
a logistic regression (pure-numpy IRLS — sklearn/scipy are not installed and the
app must never import them) of goal-vs-no-goal on:
    distance to goal, shooting angle, goalmouth placement (lateral offset + height
    + a top-corner interaction), header, big chance.
The predicted goal probability of each shot IS its post-shot xG.  Penalties are
handled separately as their empirical conversion rate.

Each on-target shot is attributed to the goalkeeper who faced it (the conceding
team's GK on the pitch at that minute, tracking GK substitutions).  We sum per-shot
PSxG per keeper and subtract the goals they actually conceded from those shots to
get PSxG+/- ("goals prevented" vs expectation).

Output: gk_psxg.csv at the repo root (committed; the JSONs are not), one row per
(id, temporada, liga) — joined in-app by the stable Opta player id.

Usage:
    python build_gk_psxg.py validate   # fit + report calibration/log-loss, no write
    python build_gk_psxg.py build        # fit + write gk_psxg.csv
"""

import os
import sys
import json
import glob
import math
import collections

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.join(_THIS, "..", "..", "Forensics xG Opta Data")
_OUT = os.path.join(_THIS, "gk_psxg.csv")

LEAGUES = ["Bundesliga", "English Premier League", "LaLiga",
           "Ligue 1", "Primeira Liga", "Serie A"]
SEASONS = ["2020-2021", "2021-2022", "2022-2023",
           "2023-2024", "2024-2025", "2025-2026"]

# Opta event-coordinate pitch is 0-100 × 0-100; goal centre sits at (100, 50).
PITCH_L = 105.0
PITCH_W = 68.0
GOAL_HALF_W = 7.32 / 2.0      # 3.66 m — half the goal width
SHOT_TYPES = {13, 14, 15, 16}  # miss, post(woodwork), attempt-saved, goal


def _qmap(e):
    return {q.get("qualifierId"): q.get("value") for q in e.get("qualifier", [])}


def _qval(e, qid):
    for x in e.get("qualifier", []):
        if x.get("qualifierId") == qid:
            return x.get("value")
    return None


def _evt_time(e):
    """Match minute (float) for ordering — minutes + seconds."""
    return float(e.get("timeMin", 0) or 0) + float(e.get("timeSec", 0) or 0) / 60.0


def _shot_features(x, y, gm_y, gm_z, is_header, is_big):
    """Feature vector for the post-shot model (non-penalty on-target shot).

    x, y      : shot origin in Opta 0-100 coords
    gm_y, gm_z: goalmouth crossing (q102 lateral ~[45,55], q103 height, crossbar ~38)
    """
    sx, sy = x / 100.0 * PITCH_L, y / 100.0 * PITCH_W
    dx, dy = PITCH_L - sx, PITCH_W / 2.0 - sy
    dist = math.hypot(dx, dy)

    # Shooting angle subtended by the goal mouth (radians; wider = easier).
    p1y = PITCH_W / 2.0 - GOAL_HALF_W
    p2y = PITCH_W / 2.0 + GOAL_HALF_W
    a = math.hypot(sx - PITCH_L, sy - p1y)
    b = math.hypot(sx - PITCH_L, sy - p2y)
    c = 2.0 * GOAL_HALF_W
    denom = 2.0 * a * b
    cos_ang = (a * a + b * b - c * c) / denom if denom > 0 else 1.0
    angle = math.acos(max(-1.0, min(1.0, cos_ang)))

    # Post-shot placement: lateral offset from centre and height of the crossing.
    ydev = abs((gm_y if gm_y is not None else 50.0) - 50.0)
    z = gm_z if gm_z is not None else 0.0
    corner = ydev * z / 100.0      # top-corner interaction (placed wide AND high)

    return [dist, angle, ydev, z, corner,
            1.0 if is_header else 0.0,
            1.0 if is_big else 0.0]


FEATURE_NAMES = ["dist", "angle", "gm_ydev", "gm_z", "corner", "header", "big_chance"]


def parse_match(path, season, liga):
    """Parse one match → (list of on-target shot records, dict gk_meta).

    Each shot record: dict with feats (or pen flag), is_goal, and the id of the
    goalkeeper that faced it.  gk_meta maps gk_id -> (name, team).
    """
    d = json.load(open(path, encoding="utf-8"))
    mi, ld = d.get("matchInfo", {}), d.get("liveData", {})
    events = ld.get("event", [])
    match_len = float(ld.get("matchDetails", {}).get("matchLengthMin", 98) or 98)
    cname = {c.get("id"): c.get("name") for c in mi.get("contestant", [])}
    teams = list(cname.keys())

    # ── Identify goalkeepers ──
    # Reliable set: starting-XI players with formation code '1' (q44) ∪ anyone who
    # records a save (typeId 10).  Saves catch substitute keepers too.
    starter_gk = {}                  # team -> starting GK pid
    gk_name = {}                     # pid -> name
    gk_team = {}                     # pid -> team
    for e in events:
        if e.get("typeId") != 34:
            continue
        team = e.get("contestantId")
        ids = (_qval(e, 30) or "").split(", ")
        forms = (_qval(e, 44) or "").split(", ")
        for i, pid in enumerate(ids):
            if pid and i < len(forms) and forms[i] == "1":
                starter_gk[team] = pid
                gk_team[pid] = team
    ever_gk = set(starter_gk.values())
    for e in events:
        if e.get("typeId") == 10 and e.get("playerId"):   # a save → that player is a GK
            ever_gk.add(e["playerId"])
            if e.get("contestantId"):
                gk_team.setdefault(e["playerId"], e["contestantId"])
    for e in events:
        pid = e.get("playerId")
        if pid in ever_gk and e.get("playerName") and pid not in gk_name:
            gk_name[pid] = e.get("playerName")

    # ── GK on-pitch timeline per team (track substitutions / red cards) ──
    # current_gk[team] starts as the starter; a sub group that takes the current
    # GK off promotes the keeper who comes on in that group.
    subs_by_key = collections.defaultdict(lambda: {"on": [], "off": []})
    for e in events:
        t = e.get("typeId")
        if t not in (18, 19) or not e.get("playerId"):
            continue
        key = (e.get("contestantId"), round(_evt_time(e), 2))
        subs_by_key[key]["on" if t == 19 else "off"].append(e["playerId"])

    # Build per-team ordered list of (start_min, gk_pid).
    timelines = {tm: [(0.0, starter_gk.get(tm))] for tm in teams}
    for (team, tmin), grp in sorted(subs_by_key.items(), key=lambda kv: kv[0][1]):
        cur = timelines.get(team, [(0.0, None)])[-1][1]
        if cur in grp["off"] and grp["on"]:
            # incoming GK = an on-player in this group; prefer a known keeper
            incoming = next((p for p in grp["on"] if p in ever_gk), grp["on"][0])
            timelines[team].append((tmin, incoming))
            ever_gk.add(incoming)
            gk_team.setdefault(incoming, team)

    def gk_on_pitch(team, tmin):
        cur = starter_gk.get(team)
        for start_min, pid in timelines.get(team, []):
            if tmin + 1e-6 >= start_min:
                cur = pid
        return cur

    # ── Walk shots ──
    shots = []
    for e in events:
        t = e.get("typeId")
        if t not in SHOT_TYPES:
            continue
        ql = _qmap(e)
        scoring_team = e.get("contestantId")
        defending_team = next((tm for tm in teams if tm != scoring_team), None)
        if defending_team is None:
            continue
        gk = gk_on_pitch(defending_team, _evt_time(e))

        is_own = 28 in ql
        is_goal = (t == 16 and not is_own)
        if is_own:
            continue                                  # own goal: not a shot faced
        is_pen = 9 in ql
        blocked = (t == 15 and 82 in ql)              # blocked by a defender
        # On target = saved (15, unblocked) or goal (16).  Woodwork(14)/miss(13) are off target.
        on_target = (t == 16) or (t == 15 and not blocked)
        if not on_target:
            continue

        rec = {"gk": gk, "team": defending_team, "is_goal": is_goal, "is_pen": is_pen}
        if not is_pen:
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            is_header = 15 in ql
            is_big = 214 in ql
            rec["feats"] = _shot_features(float(x), float(y),
                                          _f(ql.get(102)), _f(ql.get(103)),
                                          is_header, is_big)
        shots.append(rec)

    meta = {pid: (gk_name.get(pid, pid), cname.get(gk_team.get(pid), gk_team.get(pid)))
            for pid in ever_gk}
    return shots, meta


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Pure-numpy logistic regression (IRLS / Newton-Raphson, ridge-regularized) ──

def fit_logistic(X, y, l2=1.0, iters=100, tol=1e-9):
    """X: (n,p) raw features (no intercept).  Returns (beta, mean, std).

    Features are z-scored for conditioning; beta includes an intercept at index 0
    and applies to the *standardized* features (use predict_proba with mean/std).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xs = np.column_stack([np.ones(len(Xs)), Xs])      # intercept
    n, p = Xs.shape
    beta = np.zeros(p)
    ridge = l2 * np.eye(p)
    ridge[0, 0] = 0.0                                  # don't penalize intercept
    for _ in range(iters):
        eta = np.clip(Xs @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1.0 - mu), 1e-9, None)
        grad = Xs.T @ (y - mu) - ridge @ beta
        H = (Xs.T * W) @ Xs + ridge
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta, mean, std


def predict_proba(beta, mean, std, X):
    Xs = (np.atleast_2d(X) - mean) / std
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    return 1.0 / (1.0 + np.exp(-np.clip(Xs @ beta, -30, 30)))


# ── Collect every shot across the requested seasons ───────────────────────────

def collect(seasons):
    all_shots = []          # shot records (with season/liga added)
    meta = {}               # gk_id -> (name, team)
    n_files = 0
    for liga in LEAGUES:
        for season in seasons:
            pdir = os.path.join(_SRC_ROOT, liga, season, "partidos")
            files = sorted(glob.glob(os.path.join(pdir, "*.json")))
            if not files:
                continue
            for fpath in files:
                try:
                    shots, mmeta = parse_match(fpath, season, liga)
                except Exception as ex:
                    print(f"   ! skip {os.path.basename(fpath)}: {ex}")
                    continue
                n_files += 1
                for s in shots:
                    s["season"] = season
                    s["liga"] = liga
                all_shots.extend(shots)
                meta.update(mmeta)
            print(f"  {liga} {season}: {len(files)} matches")
    print(f"Parsed {n_files} matches → {len(all_shots)} on-target shots faced")
    return all_shots, meta


def build_model(all_shots):
    """Fit the non-penalty post-shot model + empirical penalty rate."""
    np_shots = [s for s in all_shots if not s["is_pen"] and "feats" in s]
    X = np.array([s["feats"] for s in np_shots], dtype=float)
    y = np.array([1.0 if s["is_goal"] else 0.0 for s in np_shots])
    beta, mean, std = fit_logistic(X, y)

    pen_shots = [s for s in all_shots if s["is_pen"]]
    pen_goals = sum(1 for s in pen_shots if s["is_goal"])
    pen_rate = (pen_goals / len(pen_shots)) if pen_shots else 0.79

    return {"beta": beta, "mean": mean, "std": std, "pen_rate": pen_rate,
            "X": X, "y": y, "n_pen": len(pen_shots), "pen_goals": pen_goals}


def shot_psxg(model, s):
    if s["is_pen"]:
        return model["pen_rate"]
    return float(predict_proba(model["beta"], model["mean"], model["std"],
                               np.array(s["feats"]))[0])


# ── Aggregate per keeper ──────────────────────────────────────────────────────

# A "soft" (saveable) goal is one conceded from an open-play shot the model
# rates below this post-shot xG — i.e. an average keeper saves it 80%+ of the
# time.  Counting these surfaces error-proneness that PSxG+/- nets away against
# saves on hard shots (e.g. a busy keeper behind a leaky defence).
SOFT_GOAL_PSXG_MAX = 0.20

def aggregate(all_shots, model):
    agg = collections.defaultdict(
        lambda: {"psxg": 0.0, "goals": 0, "shots": 0, "soft": 0})
    for s in all_shots:
        gk = s["gk"]
        if not gk:
            continue
        a = agg[(gk, s["season"], s["liga"])]
        p = shot_psxg(model, s)
        a["psxg"] += p
        a["goals"] += 1 if s["is_goal"] else 0
        a["shots"] += 1
        if s["is_goal"] and not s["is_pen"] and p < SOFT_GOAL_PSXG_MAX:
            a["soft"] += 1
    return agg


# ── Commands ──────────────────────────────────────────────────────────────────

def _report_calibration(model):
    beta, mean, std = model["beta"], model["mean"], model["std"]
    X, y = model["X"], model["y"]
    p = predict_proba(beta, mean, std, X)
    eps = 1e-12
    logloss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    print("\n── Model fit (non-penalty on-target shots) ──")
    print(f"  shots={len(y)}  goals={int(y.sum())}  conv={y.mean():.3f}")
    print(f"  Σ PSxG={p.sum():.1f}  vs  Σ goals={y.sum():.0f}  "
          f"(ratio {p.sum()/max(y.sum(),1):.3f})")
    print(f"  log-loss={logloss:.4f}")
    print(f"  intercept + coefs (standardized):")
    print(f"    intercept = {beta[0]:+.3f}")
    for nm, b in zip(FEATURE_NAMES, beta[1:]):
        print(f"    {nm:10s} = {b:+.3f}")
    print(f"  penalties: {model['n_pen']} faced, {model['pen_goals']} scored "
          f"→ rate {model['pen_rate']:.3f}")
    print("\n  calibration by predicted decile (pred vs actual):")
    order = np.argsort(p)
    for d in range(10):
        lo, hi = d * len(p) // 10, (d + 1) * len(p) // 10
        idx = order[lo:hi]
        print(f"    decile {d+1:2d}: pred={p[idx].mean():.3f}  "
              f"actual={y[idx].mean():.3f}  n={len(idx)}")


def validate():
    seasons = ["2025-2026"]
    print(f"Validating on {seasons[0]} only (fast)…")
    shots, meta = collect(seasons)
    model = build_model(shots)
    _report_calibration(model)
    agg = aggregate(shots, model)
    rows = []
    for (gk, season, liga), a in agg.items():
        if a["shots"] < 20:
            continue
        nm, team = meta.get(gk, (gk, "?"))
        rows.append((nm, team, liga, a["psxg"], a["psxg"] - a["goals"],
                     a["goals"], a["shots"], a["soft"]))
    df = pd.DataFrame(rows, columns=["name", "team", "liga", "PSxG", "PSxG+/-",
                                     "goals", "shots_OT", "soft"]).sort_values(
        "PSxG+/-", ascending=False)
    pd.set_option("display.width", 160)
    print("\n── Best shot-stoppers 2025-26 (≥20 on-target faced) ──")
    print(df.head(15).to_string(index=False,
          formatters={"PSxG": "{:.1f}".format, "PSxG+/-": "{:+.1f}".format}))
    print("\n── Worst ──")
    print(df.tail(8).to_string(index=False,
          formatters={"PSxG": "{:.1f}".format, "PSxG+/-": "{:+.1f}".format}))


def build():
    shots, meta = collect(SEASONS)
    model = build_model(shots)
    _report_calibration(model)
    agg = aggregate(shots, model)
    rows = []
    for (gk, season, liga), a in agg.items():
        rows.append({
            "id": gk, "temporada": season, "liga": liga,
            "PSxG": round(a["psxg"], 2),
            "PSxG+/-": round(a["psxg"] - a["goals"], 2),
            "psxg_goals_faced": a["goals"],
            "shots_on_target_faced": a["shots"],
            "soft_goals_conceded": a["soft"],
        })
    out = pd.DataFrame(rows).sort_values(["temporada", "liga", "PSxG"],
                                         ascending=[False, True, False])
    out.to_csv(_OUT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(out)} keeper-seasons → {_OUT} "
          f"({out['temporada'].nunique()} seasons, {out['liga'].nunique()} leagues)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "build":
        build()
    else:
        validate()
