"""
Scrape each player's preferred foot (left / right / both) from Transfermarkt
and save to player_footedness.csv, keyed by the stable Opta player id (so it
joins cleanly to any player/season and avoids same-name collisions).

GAP-FILL by default: existing values are kept; only players still missing a
foot are looked up. Safe to interrupt and re-run — it checkpoints to disk.

Must run from a normal (residential) internet connection — Transfermarkt blocks
cloud/datacenter IPs. Expect a few hours for a fresh fill (two requests per
player: quick-search + profile page), with rate-limiting.

Usage:
    python build_footedness.py
"""

import os
import re
import csv
import time
import urllib.parse
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

_THIS = os.path.dirname(os.path.abspath(__file__))
_LEAGUE_FOLDERS = ["English Premier League", "LaLiga", "Bundesliga",
                   "Ligue 1", "Serie A", "Primeira Liga"]
OUTPUT_CSV = os.path.join(_THIS, "player_footedness.csv")
FIELDNAMES = ["id", "nombre", "foot"]

_TM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 2.5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 25
_VALID = {"left", "right", "both"}
_EMPTY = {"", "nan", "none", "n/a"}


def _present(v):
    return v is not None and str(v).strip().lower() not in _EMPTY


def _normalize_name(name):
    s = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _http_get(url):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=_TM_HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            time.sleep(DELAY * (attempt + 2))
        except Exception:
            time.sleep(DELAY * (attempt + 2))
    return None


def _fetch_foot(short_name, team=None):
    """Quick-search short_name, pick the team-matched player, read 'Foot' from
    their Transfermarkt profile. Returns 'left'/'right'/'both' or None."""
    for query in _query_variants(short_name):
        r = _http_get("https://www.transfermarkt.com/schnellsuche/ergebnis/"
                      "schnellsuche?query=" + urllib.parse.quote(query))
        if r is None:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table.items tr.odd, table.items tr.even")
        prof = None
        for row in rows:
            link = row.select_one('a[href*="/profil/spieler/"]')
            if not link:
                continue
            if team:
                row_text = _normalize_name(row.get_text(" ", strip=True))
                if _normalize_name(team) and _normalize_name(team).split()[0] in row_text:
                    prof = link["href"]
                    break
            if prof is None:
                prof = link["href"]
        if not prof:
            continue
        pr = _http_get("https://www.transfermarkt.com" + prof)
        time.sleep(DELAY)
        if pr is None:
            continue
        m = re.search(r'Foot:?\s*</span>\s*<span[^>]*>\s*([a-zA-Z]+)', pr.text)
        if m and m.group(1).lower() in _VALID:
            return m.group(1).lower()
    return None


def _query_variants(name):
    variants, parts = [name], name.replace(".", " ").split()
    if len(parts) >= 2:
        variants.append(parts[-1])
    seen, out = set(), []
    for v in variants:
        if v.strip() and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v.strip())
    return out


def collect_players():
    """[{id, nombre, team}] across all leagues' current season (dedup by id)."""
    seen, players = set(), []
    for folder in _LEAGUE_FOLDERS:
        path = os.path.join(_THIS, folder, "jugadores_seasonstats.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if "Time Played" in df.columns:
            df = df[pd.to_numeric(df["Time Played"], errors="coerce").fillna(0) > 0]
        for _, r in df.iterrows():
            pid = str(r.get("id", "")).strip()
            if pid and pid not in seen:
                seen.add(pid)
                players.append({"id": pid, "nombre": str(r.get("nombre", "")).strip(),
                                "team": str(r.get("equipo", "")).strip()})
    return players


def load_existing():
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        return {}
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return {r["id"]: {"nombre": r.get("nombre", ""), "foot": r.get("foot", "")}
            for _, r in df.iterrows() if r.get("id")}


def write_csv(rows, order):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for pid in order:
            rec = rows.get(pid, {})
            w.writerow({"id": pid, "nombre": rec.get("nombre", ""), "foot": rec.get("foot", "")})


def main():
    players = collect_players()
    order = [p["id"] for p in players]
    print(f"Players (current season, all leagues): {len(players)}")
    rows = load_existing()
    have = sum(1 for v in rows.values() if _present(v.get("foot")))
    print(f"Existing: {len(rows)} rows, {have} with foot.")
    todo = [p for p in players if not _present(rows.get(p["id"], {}).get("foot"))]
    print(f"To fetch: {len(todo)}\n")

    fetched = 0
    for i, p in enumerate(players):
        pid = p["id"]
        cur = rows.get(pid, {"nombre": p["nombre"], "foot": ""})
        if _present(cur.get("foot")):
            rows[pid] = cur
            continue
        foot = _fetch_foot(p["nombre"], team=p["team"]) or ""
        rows[pid] = {"nombre": p["nombre"], "foot": foot}
        fetched += 1
        print(f"  [{i + 1}/{len(players)}] {p['nombre']} ({p['team'][:18]}) -> {foot or 'N/A'}")
        if fetched % CHECKPOINT_EVERY == 0:
            write_csv(rows, order)
            print(f"    …checkpoint ({fetched} fetched)")
        time.sleep(0.2)

    write_csv(rows, order)
    final = sum(1 for v in rows.values() if _present(v.get("foot")))
    print(f"\nDone. {final}/{len(order)} players have a foot. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
