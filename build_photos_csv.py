"""
Build / gap-fill player_photos.csv — player cutout photos from TheSportsDB
for every player in the Opta database.

Output format (exactly what app.py consumes):
    short_name, photo_url

GAP-FILL BY DEFAULT: existing photos are kept and never re-fetched; only
players without a photo are looked up.  Safe to interrupt and re-run —
it checkpoints to disk and resumes from whatever is already in the CSV.

ANTI-COLLISION: the original CSV had 385 players sharing one face because
TheSportsDB short-name search returns the wrong player for common surnames.
This builder (a) only accepts a team-matched or unique result — it never
guesses players[0] — and (b) refuses any photo URL already assigned to a
different player, so a wrong match becomes an avatar instead of a duplicate.

Run locally (needs internet):
    python build_photos_csv.py
"""

import os
import csv
import json
import time
import urllib.parse
import urllib.request
import unicodedata

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # repo root (app.py lives here)
_LOCAL_DIR = os.path.join(_THIS_DIR, "..", "..", "Forensics xG Opta Data")
OPTA_DIR = _THIS_DIR if os.path.isdir(os.path.join(_THIS_DIR, "Bundesliga")) else _LOCAL_DIR

OUTPUT_CSV = os.path.join(_THIS_DIR, "player_photos.csv")
FIELDNAMES = ["short_name", "photo_url"]

LEAGUE_FOLDERS = {
    "Premier League": "English Premier League",
    "LaLiga": "LaLiga",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Serie A": "Serie A",
    "Primeira Liga": "Primeira Liga",
}

DELAY = 1.0             # seconds between API calls (TheSportsDB free tier)
CHECKPOINT_EVERY = 40   # rewrite the CSV after this many newly-fetched players

_EMPTY_TOKENS = {"", "nan", "none", "n/a", "na"}


def _present(v):
    if v is None:
        return False
    try:
        if isinstance(v, float) and pd.isna(v):
            return False
    except Exception:
        pass
    return str(v).strip().lower() not in _EMPTY_TOKENS


def _normalize_name(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


# ── Photo fetch (hardened, mirrors app.py) ────────────────────────────────────

def _fetch_photo(short_name, team, used_urls):
    """Return a confident photo URL for *short_name*, or None.

    Only a team-matched or single-result soccer player is accepted, and a URL
    already used by another player is rejected as a wrong match.
    """
    try:
        q = urllib.parse.quote(short_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    players = [p for p in (data.get("player") or [])
               if (p.get("strSport") or "").lower() == "soccer"]
    if not players:
        return None

    def _photo(p):
        return p.get("strCutout") or p.get("strThumb") or p.get("strRender")

    chosen = None
    if team:
        tl = _normalize_name(team).replace(" fc", "").replace("fc ", "").strip()
        for p in players:
            pt = _normalize_name(p.get("strTeam") or "")
            if tl and pt and (tl in pt or pt in tl):
                chosen = _photo(p)
                if chosen:
                    break
    if not chosen and len(players) == 1:
        chosen = _photo(players[0])

    if not chosen or chosen in used_urls:   # ambiguous, or a wrong/duplicate match
        return None
    return chosen


# ── Data plumbing ─────────────────────────────────────────────────────────────

def collect_players():
    """Deduplicated [{short_name, team}] from Opta data (keyed by short_name)."""
    seen, players = set(), []
    for display_name, folder_name in LEAGUE_FOLDERS.items():
        csv_path = os.path.join(OPTA_DIR, folder_name, "jugadores_seasonstats.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP {display_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        if df.empty or "nombre" not in df.columns:
            continue
        if "Time Played" in df.columns:
            df = df[pd.to_numeric(df["Time Played"], errors="coerce").fillna(0) > 0]
        for _, row in df.iterrows():
            name = str(row.get("nombre", "")).strip()
            team = str(row.get("equipo", "")).strip()
            if not name or name.lower() == "nan" or name in seen:
                continue
            seen.add(name)
            players.append({"short_name": name, "team": team})
    return players


def load_existing():
    """Read existing CSV as {short_name: photo_url} (present values only)."""
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        return {}
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        sn = (r.get("short_name") or "").strip()
        url = (r.get("photo_url") or "").strip()
        if sn and _present(url):
            out[sn] = url
    return out


def write_csv(rows, order):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for sn in order:
            url = rows.get(sn, "")
            if _present(url):
                w.writerow({"short_name": sn, "photo_url": url})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting player list from Opta data…")
    players = collect_players()
    order = [p["short_name"] for p in players]
    print(f"Found {len(players)} unique players.")
    if not players:
        print(f"  (No Opta data found under {OPTA_DIR}.)")
        return

    rows = load_existing()
    used_urls = set(rows.values())
    print(f"Existing CSV: {len(rows)} photos.")

    todo = [p for p in players if not _present(rows.get(p["short_name"]))]
    print(f"Players missing a photo: {len(todo)}\n")
    if not todo:
        print("Nothing to fill — every player already has a photo.")
        write_csv(rows, order)
        return

    fetched = added = 0
    for i, p in enumerate(players):
        sn, team = p["short_name"], p["team"]
        if _present(rows.get(sn)):
            continue
        url = _fetch_photo(sn, team, used_urls)
        time.sleep(DELAY)
        fetched += 1
        if url:
            rows[sn] = url
            used_urls.add(url)
            added += 1
            print(f"  [{i + 1}/{len(players)}] {sn} -> OK")
        else:
            print(f"  [{i + 1}/{len(players)}] {sn} -> no confident match (avatar)")

        if added and added % CHECKPOINT_EVERY == 0:
            write_csv(rows, order)
            print(f"    …checkpoint saved ({added} new photos)")

    write_csv(rows, order)
    print(f"\nDone. Tried {fetched}, added {added} new photos. "
          f"Total {sum(1 for v in rows.values() if _present(v))} across {len(order)} players.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
