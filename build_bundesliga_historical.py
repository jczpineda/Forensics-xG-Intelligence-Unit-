"""
Rebuild correct TOP-FLIGHT Bundesliga player season stats by aggregating the
raw Opta match-event JSONs in each season's `partidos/` folder.

Why: the Bundesliga `jugadores_historical.csv` was built from the wrong
division (2. Bundesliga). The match event feeds in `partidos/` are the correct
1. Bundesliga, so we aggregate them into season player totals here.

Strategy: this maps Opta F24 event type IDs + qualifiers to the seasonstats
column names. It is calibrated against 2025-26, where we have BOTH the partidos
and the official jugadores_seasonstats.csv, so accuracy can be proven before
regenerating the past seasons.

Usage:
    python build_bundesliga_historical.py validate   # check vs official 25-26
    python build_bundesliga_historical.py build       # write corrected historical CSV
"""

import os
import json
import glob
import sys
import collections

import pandas as pd

# Source Opta data (per league/season/partidos) — sibling of the repo.
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS, "..", "..", "Forensics xG Opta Data", "Bundesliga")
_OUT = os.path.join(_THIS, "Bundesliga", "jugadores_historical.csv")

PAST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def _q(ev):
    """Set of qualifierIds on an event."""
    return {x.get("qualifierId") for x in ev.get("qualifier", [])}


def _qval(ev, qid):
    for x in ev.get("qualifier", []):
        if x.get("qualifierId") == qid:
            return x.get("value")
    return None


def _blank_stats():
    return collections.defaultdict(float)


def parse_match(path):
    """Return {playerId: {meta + stat counters}} for one match."""
    d = json.load(open(path, encoding="utf-8"))
    mi, ld = d.get("matchInfo", {}), d.get("liveData", {})
    events = ld.get("event", [])
    match_len = float(ld.get("matchDetails", {}).get("matchLengthMin", 90) or 90)

    # contestant id -> name
    cname = {c.get("id"): c.get("name") for c in mi.get("contestant", [])}

    players = {}  # pid -> dict

    def P(pid):
        if pid not in players:
            players[pid] = {"id": pid, "name": None, "team": None, "dorsal": None,
                            "started": 0, "on": None, "off": None, "pos_code": None,
                            "s": _blank_stats()}
        return players[pid]

    # ── Lineups / minutes from team set-up (typeId 34) ──
    for e in events:
        if e.get("typeId") != 34:
            continue
        team = cname.get(e.get("contestantId"))
        ids = (_qval(e, 30) or "").split(", ")
        slots = (_qval(e, 131) or "").split(", ")
        shirts = (_qval(e, 59) or "").split(", ")
        forms = (_qval(e, 44) or "").split(", ")  # formation position code (1=GK,2=DEF,3=MID,4/5=FWD)
        for i, pid in enumerate(ids):
            if not pid:
                continue
            p = P(pid)
            p["team"] = team
            if i < len(shirts):
                p["dorsal"] = shirts[i]
            slot = slots[i] if i < len(slots) else "0"
            started = slot not in ("", "0")
            p["started"] = 1 if started else 0
            p["on"] = 0.0 if started else None
            p["pos_code"] = forms[i] if i < len(forms) else None

    # ── Substitutions (player on/off) ──
    for e in events:
        t = e.get("typeId")
        if t not in (18, 19):
            continue
        pid = e.get("playerId")
        if not pid:
            continue
        tmin = float(e.get("timeMin", 0) or 0)
        p = P(pid)
        if t == 19:      # player on
            p["on"] = tmin
        else:            # player off
            p["off"] = tmin

    # ── Per-event stat accumulation ──
    for i, e in enumerate(events):
        pid = e.get("playerId")
        if not pid:
            continue
        p = P(pid)
        if p["name"] is None:
            p["name"] = e.get("playerName")
        if p["team"] is None:
            p["team"] = cname.get(e.get("contestantId"))
        s = p["s"]
        t = e.get("typeId")
        out = e.get("outcome")
        ql = _q(e)
        x = e.get("x")
        is_gk = (p.get("pos_code") == "1")

        if t == 1:  # Pass
            s["Total Passes"] += 1
            succ = (out == 1)
            is_cross = 2 in ql
            is_corner = 6 in ql
            is_long = 1 in ql
            if succ:
                if not (is_cross or is_corner):
                    s["Total Successful Passes ( Excl Crosses & Corners ) "] += 1
                    s["Successful Open Play Passes"] += 1
                s["Successful Long Passes" if is_long else "Successful Short Passes"] += 1
                if is_cross or is_corner:
                    s["Successful Crosses & Corners"] += 1
                    if is_cross and not is_corner:
                        s["Successful Crosses open play"] += 1
                # Forward = successful pass that ends meaningfully upfield
                endx = _qval(e, 140)
                try:
                    if endx is not None and x is not None and float(endx) > float(x) + 5:
                        s["Forward Passes"] += 1
                    elif endx is not None and x is not None and float(endx) < float(x):
                        s["Backward Passes"] += 1
                except (TypeError, ValueError):
                    pass
            else:
                if not (is_cross or is_corner):
                    s["Total Unsuccessful Passes ( Excl Crosses & Corners )"] += 1
                s["Unsuccessful Long Passes" if is_long else "Unsuccessful Short Passes"] += 1
                if is_cross or is_corner:
                    s["Unsuccessful Crosses & Corners"] += 1
                    if is_cross and not is_corner:
                        s["Unsuccessful Crosses open play"] += 1
            if 4 in ql:                        # through ball
                s["Through balls"] += 1
            if is_corner:
                s["Corners Taken (incl short corners)"] += 1
        elif t == 2:
            s["Total Passes"] += 1             # offside pass counts as attempted
        elif t in (13, 14, 15, 16):           # shots
            blocked = (t == 15 and 82 in ql)
            if blocked:
                s["Blocked Shots"] += 1        # blocked by a defender — not a shot on target
            else:
                s["Total Shots"] += 1
                if t in (15, 16):
                    s["Shots On Target ( inc goals )"] += 1
                else:
                    s["Shots Off Target (inc woodwork)"] += 1
            if t == 16 and 28 not in ql:       # goal (not own goal)
                s["Goals"] += 1
                if 15 in ql:
                    s["Headed Goals"] += 1
                if 9 in ql or 26 in ql:        # penalty
                    s["Penalty Goals"] += 1
                else:
                    s["Goals Openplay"] += 1
            # Key pass / assist: credit the most recent successful same-team pass
            # before the shot. A key pass must be close (≤3 events); an assist
            # (on a goal) is credited a bit wider since crosses/set-pieces sit
            # further back. An assist always implies a key pass.
            cid = e.get("contestantId")
            is_goal = (t == 16 and 28 not in ql)
            for j in range(i - 1, max(i - 7, -1), -1):
                pe = events[j]
                pt = pe.get("typeId")
                if pe.get("contestantId") and pe.get("contestantId") != cid and pt in (1, 2, 3, 13, 14, 15, 16):
                    break  # possession changed hands
                if pt == 1 and pe.get("outcome") == 1 and pe.get("playerId") and pe.get("playerId") != pid:
                    kp = P(pe["playerId"])["s"]
                    dist = i - j
                    if dist <= 2:
                        kp["Key Passes (Attempt Assists)"] += 1
                    if is_goal:
                        kp["Goal Assists"] += 1
                        if dist > 2:
                            kp["Key Passes (Attempt Assists)"] += 1
                    break
                if pt in (13, 14, 15, 16):
                    break
        elif t == 3:  # take on (dribble)
            s["Successful Dribbles" if out == 1 else "Unsuccessful Dribbles"] += 1
        elif t == 7:  # tackle
            s["Total Tackles"] += 1
            s["Tackles Won" if out == 1 else "Tackles Lost"] += 1
        elif t == 8:
            s["Interceptions"] += 1
        elif t == 12:
            s["Total Clearances"] += 1
        elif t == 49:
            s["Recoveries"] += 1
        elif t == 44:  # aerial
            s["Aerial Duels"] += 1
            s["Aerial Duels won" if out == 1 else "Aerial Duels lost"] += 1
        elif t == 4:   # foul
            s["Total Fouls Won" if out == 1 else "Total Fouls Conceded"] += 1
        elif t == 17:  # card
            if 31 in ql:
                s["Yellow Cards"] += 1
            elif 32 in ql:
                s["Red Cards - 2nd Yellow"] += 1
                s["Total Red Cards"] += 1
            elif 33 in ql:
                s["Straight Red Cards"] += 1
                s["Total Red Cards"] += 1
        elif t == 50:
            s["Total Losses Of Possession"] += 1
        elif t == 10:  # save / block — GK only counts as a save; outfield = block
            if is_gk:
                s["Saves Made"] += 1
            else:
                s["Blocked Shots"] += 1
        elif t == 11:  # claim / catch (GK)
            if is_gk:
                s["Catches"] += 1
        elif t == 41:  # punch (GK)
            s["Punches"] += 1
        elif t == 74:  # blocked pass
            s["Blocks"] += 1
        # touches
        if t in (1, 2, 3, 7, 8, 12, 13, 14, 15, 16, 41, 42, 44, 49, 50, 61, 74):
            s["Touches"] += 1
            try:
                if x is not None and float(x) >= 66.7:
                    s["Final Third Touches"] += 1
                if x is not None and float(x) >= 83.0:
                    s["Total Touches In Opposition Box"] += 1
            except (TypeError, ValueError):
                pass

    # Ground duels (derived): tackles + take-ons faced are complex; approximate
    # ground duels won = tackles won + successful dribbles for a rough Duel set.
    for p in players.values():
        s = p["s"]
        s["Ground Duels won"] = s.get("Tackles Won", 0) + s.get("Successful Dribbles", 0)
        s["Ground Duels"] = (s.get("Total Tackles", 0) + s.get("Successful Dribbles", 0)
                             + s.get("Unsuccessful Dribbles", 0))
        s["Duels won"] = s["Ground Duels won"] + s.get("Aerial Duels won", 0)
        s["Duels"] = s["Ground Duels"] + s.get("Aerial Duels", 0)

    # ── Finalize minutes ──
    for p in players.values():
        on = p["on"] if p["on"] is not None else (0.0 if p["started"] else None)
        off = p["off"] if p["off"] is not None else match_len
        if on is None:
            p["minutes"] = 0.0
            p["appeared"] = 0
        else:
            p["minutes"] = max(0.0, off - on)
            p["appeared"] = 1
        p["sub_on"] = 1 if (p["on"] is not None and not p["started"]) else 0
        p["sub_off"] = 1 if p["off"] is not None else 0

    # ── Goals conceded / clean sheets → each team's primary GK ──
    teams = list(cname.values())
    conceded = collections.Counter()
    for e in events:
        if e.get("typeId") != 16:
            continue
        scoring = cname.get(e.get("contestantId"))
        other = next((tn for tn in teams if tn != scoring), None)
        if 28 in _q(e):                       # own goal → conceded by scorer's team
            conceded[scoring] += 1
        elif other:
            conceded[other] += 1
    for team in teams:
        gks = [p for p in players.values()
               if p.get("team") == team and p.get("pos_code") == "1" and p.get("appeared")]
        if not gks:
            continue
        primary = max(gks, key=lambda p: p["minutes"])
        c = conceded.get(team, 0)
        primary["s"]["Goals Conceded"] += c
        primary["s"]["Clean Sheets"] += 1 if c == 0 else 0
        primary["s"]["Games Played"] += 1
    return players


def aggregate_season(season):
    """Aggregate every match in a season's partidos/ → per-player totals DataFrame."""
    pdir = os.path.join(_SRC, season, "partidos")
    files = sorted(glob.glob(os.path.join(pdir, "*.json")))
    agg = {}  # pid -> totals
    for f in files:
        try:
            match = parse_match(f)
        except Exception as ex:
            print(f"   ! skip {os.path.basename(f)}: {ex}")
            continue
        for pid, p in match.items():
            a = agg.get(pid)
            if a is None:
                a = {"id": pid, "name": p["name"], "team": p["team"], "dorsal": p["dorsal"],
                     "Appearances": 0, "Starts": 0, "Substitute On": 0, "Substitute Off": 0,
                     "Time Played": 0.0, "pos": collections.Counter(), "s": collections.defaultdict(float)}
                agg[pid] = a
            if p["name"] and not a["name"]:
                a["name"] = p["name"]
            if p["team"]:
                a["team"] = p["team"]          # last team seen
            if p["dorsal"]:
                a["dorsal"] = p["dorsal"]
            if p.get("pos_code"):
                a["pos"][p["pos_code"]] += 1
            a["Appearances"] += p.get("appeared", 0)
            a["Starts"] += p.get("started", 0)
            a["Substitute On"] += p.get("sub_on", 0)
            a["Substitute Off"] += p.get("sub_off", 0)
            a["Time Played"] += p.get("minutes", 0.0)
            for k, v in p["s"].items():
                a["s"][k] += v

    _POS = {"1": "Goalkeeper", "2": "Defender", "3": "Midfielder", "4": "Forward", "5": "Forward"}
    rows = []
    for pid, a in agg.items():
        pos_code = a["pos"].most_common(1)[0][0] if a["pos"] else None
        row = {"temporada": season, "liga": "Bundesliga", "id": pid, "nombre": a["name"],
               "equipo": a["team"], "dorsal": a["dorsal"],
               "posicion": _POS.get(pos_code, "Midfielder"),
               "Appearances": a["Appearances"], "Starts": a["Starts"],
               "Substitute On": a["Substitute On"], "Substitute Off": a["Substitute Off"],
               "Time Played": round(a["Time Played"])}
        for k, v in a["s"].items():
            if not k.startswith("_"):
                row[k] = round(v, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Validation against the official 2025-26 seasonstats ───────────────────────

def validate():
    season = "2025-2026"
    print(f"Aggregating {season} partidos…")
    mine = aggregate_season(season)
    official = pd.read_csv(os.path.join(_THIS, "Bundesliga", "jugadores_seasonstats.csv"),
                           encoding="utf-8-sig", low_memory=False)
    print(f"  mine: {len(mine)} players | official: {len(official)} players")
    m = mine.merge(official, on="id", suffixes=("_mine", "_off"))
    print(f"  matched by id: {len(m)}")
    checks = ["Time Played", "Appearances", "Goals", "Goal Assists", "Total Shots",
              "Total Passes", "Total Successful Passes ( Excl Crosses & Corners ) ",
              "Total Tackles", "Interceptions", "Total Clearances", "Recoveries",
              "Aerial Duels won", "Successful Dribbles", "Saves Made",
              "Key Passes (Attempt Assists)", "Forward Passes"]
    print(f"\n  {'metric':45s} {'mine_sum':>10s} {'off_sum':>10s} {'ratio':>6s} {'corr':>6s}")
    for c in checks:
        cm, co = c + "_mine", c + "_off"
        if cm not in m.columns or co not in m.columns:
            print(f"  {c:45s}  (missing)")
            continue
        a = pd.to_numeric(m[cm], errors="coerce").fillna(0)
        b = pd.to_numeric(m[co], errors="coerce").fillna(0)
        ratio = a.sum() / b.sum() if b.sum() else float("nan")
        corr = a.corr(b)
        print(f"  {c:45s} {a.sum():10.0f} {b.sum():10.0f} {ratio:6.2f} {corr:6.3f}")


def build():
    frames = []
    for sea in PAST_SEASONS:
        print(f"Aggregating {sea}…")
        df = aggregate_season(sea)
        frames.append(df)
        print(f"  {len(df)} players")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(_OUT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(out)} rows → {_OUT}  ({out['temporada'].nunique()} seasons)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "build":
        build()
    else:
        validate()
