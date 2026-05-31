"""
build_player_data.py — Run this LOCALLY to populate player_photos.csv
and Player Financials/player_financials.csv from live sources.

Usage:
    python build_player_data.py

Output:
    - player_photos.csv  (in repo root)
    - Player Financials/player_financials.csv  (in repo)

After running, commit and push both CSVs to GitHub so Streamlit Cloud
uses them instead of attempting (blocked) live scraping.
"""

import os
import re
import json
import time
import unicodedata
import urllib.request
import urllib.parse
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIR  = os.path.join(_REPO_DIR, "..", "..", "Forensics xG Opta Data")
OPTA_DIR    = _REPO_DIR if os.path.isdir(os.path.join(_REPO_DIR, "Bundesliga")) else _LOCAL_DIR

LEAGUE_FOLDERS = {
    "Premier League":  "English Premier League",
    "LaLiga":          "LaLiga",
    "Bundesliga":      "Bundesliga",
    "Ligue 1":         "Ligue 1",
    "Serie A":         "Serie A",
    "Primeira Liga":   "Primeira Liga",
}

PHOTOS_CSV       = os.path.join(_REPO_DIR, "player_photos.csv")
FINANCIALS_CSV   = os.path.join(_REPO_DIR, "Player Financials", "player_financials.csv")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _load_existing(path, key_col):
    """Load an existing CSV into a dict keyed by key_col, so we skip already-fetched rows."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", comment="#")
    result = {}
    for _, r in df.iterrows():
        k = str(r.get(key_col, "")).strip()
        if k:
            result[k] = r.to_dict()
    return result


# ── Load all players from Opta CSVs ──────────────────────────────────────────

def load_players():
    players = []
    seen = set()
    for league, folder in LEAGUE_FOLDERS.items():
        path = os.path.join(OPTA_DIR, folder, "jugadores_seasonstats.csv")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        for _, row in df.iterrows():
            name = str(row.get("nombre", "")).strip()
            team = str(row.get("equipo", "")).strip()
            if name and name not in seen:
                seen.add(name)
                players.append({"nombre": name, "equipo": team, "league": league})
    print(f"Loaded {len(players)} unique players from Opta data.")
    return players


# ── Photo fetching via TheSportsDB ────────────────────────────────────────────

def fetch_photo(player_name, team=None):
    try:
        q = urllib.parse.quote(player_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        players = data.get("player") or []

        def _photo(p):
            return p.get("strCutout") or p.get("strThumb") or p.get("strRender")

        if players and team:
            team_lower = team.lower().replace(" fc", "").replace("fc ", "").strip()
            for p in players:
                p_team = (p.get("strTeam") or "").lower().replace(" fc", "").replace("fc ", "").strip()
                if (p.get("strSport") or "").lower() == "soccer" and (
                    team_lower in p_team or p_team in team_lower
                ):
                    photo = _photo(p)
                    if photo:
                        return photo
        if players:
            for p in players:
                if (p.get("strSport") or "").lower() == "soccer":
                    photo = _photo(p)
                    if photo:
                        return photo
    except Exception:
        pass
    return None


# ── Market value fetching via Transfermarkt ───────────────────────────────────

def fetch_transfermarkt(player_name, team=None):
    try:
        query = urllib.parse.quote(player_name)
        url = f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query={query}"
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table", class_="items")
        if not tables:
            return None
        rows = tables[0].find_all("tr", class_=["odd", "even"])
        best = None
        for row in rows:
            value_cell = row.find("td", class_=lambda c: c and "rechts" in c and "hauptlink" in c)
            if not value_cell:
                continue
            mv = value_cell.get_text(strip=True)
            if not mv or mv == "-":
                continue
            if team:
                cells = row.find_all("td")
                row_text = " ".join(c.get_text(strip=True).lower() for c in cells)
                team_norm = _normalize(team)
                if team_norm and team_norm in row_text:
                    return mv
            if best is None:
                best = mv
        return best
    except Exception:
        return None


# ── Salary fetching via Capology ─────────────────────────────────────────────

def _fetch_capology_index():
    try:
        resp = requests.get(
            "https://www.capology.com/static/files/search_players.json",
            headers=_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


_CAPOLOGY_INDEX = None

def fetch_capology(player_name, team=None):
    global _CAPOLOGY_INDEX
    if _CAPOLOGY_INDEX is None:
        print("  Downloading Capology index…")
        _CAPOLOGY_INDEX = _fetch_capology_index()
        print(f"  Capology index: {len(_CAPOLOGY_INDEX)} entries")

    target = _normalize(player_name)
    candidates = [e for e in _CAPOLOGY_INDEX if _normalize(e.get("name", "")) == target]
    if not candidates:
        parts = target.split()
        if len(parts) >= 2:
            surname, initial = parts[-1], parts[0].rstrip(".")
            candidates = [
                e for e in _CAPOLOGY_INDEX
                if (nm := _normalize(e.get("name", "")).split())
                and len(nm) >= 2
                and nm[-1] == surname
                and nm[0].startswith(initial)
            ]
    if not candidates:
        return None
    slug = candidates[0].get("link")
    if len(candidates) > 1 and team:
        team_norm = _normalize(team)
        for c in candidates:
            club_url = (c.get("club") or "").lower()
            if team_norm.replace(" ", "-") in club_url or team_norm.replace(" ", "") in club_url:
                slug = c.get("link")
                break

    try:
        url = f"https://www.capology.com{slug}/"
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        m_base = re.search(r'"annual_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
        if m_base:
            raw = int(m_base.group(1))
            m_bonus = re.search(r'"bonus_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
            if m_bonus:
                raw += int(m_bonus.group(1))
            if raw >= 1_000_000:
                return f"€{raw / 1_000_000:,.2f}M"
            elif raw > 0:
                return f"€{raw / 1_000:,.0f}K"
    except Exception:
        pass
    return None


# ── Main build ────────────────────────────────────────────────────────────────

def main():
    players = load_players()

    # Load existing data so we don't re-fetch
    existing_photos     = _load_existing(PHOTOS_CSV, "short_name")
    existing_financials = _load_existing(FINANCIALS_CSV, "short_name")

    photo_rows      = dict(existing_photos)
    financial_rows  = dict(existing_financials)

    total = len(players)
    for i, p in enumerate(players, 1):
        name = p["nombre"]
        team = p["equipo"]
        print(f"[{i}/{total}] {name} ({team})", end=" ")

        # ── Photo ──────────────────────────────────────────────────────
        if name not in photo_rows:
            photo = fetch_photo(name, team)
            if photo:
                photo_rows[name] = {"short_name": name, "photo_url": photo}
                print(f"📷 ", end="")
            else:
                print(f"📷? ", end="")
            time.sleep(0.3)  # be polite to TheSportsDB

        # ── Financials ─────────────────────────────────────────────────
        needs_mv  = name not in financial_rows or not financial_rows[name].get("market_value")
        needs_sal = name not in financial_rows or not financial_rows[name].get("salary")

        mv  = financial_rows.get(name, {}).get("market_value") or None
        sal = financial_rows.get(name, {}).get("salary") or None

        if needs_mv:
            mv = fetch_transfermarkt(name, team)
            if mv:
                print(f"💰{mv} ", end="")
            time.sleep(1.2)  # Transfermarkt rate limiting

        if needs_sal:
            sal = fetch_capology(name, team)
            if sal:
                print(f"💶{sal} ", end="")
            time.sleep(0.5)

        financial_rows[name] = {
            "short_name":   name,
            "market_value": mv or "",
            "salary":       sal or "",
        }

        print()  # newline

        # ── Write CSVs every 50 players (checkpoint) ───────────────────
        if i % 50 == 0:
            _write_csvs(photo_rows, financial_rows)
            print(f"  ✅ Checkpoint saved at {i}/{total}")

    _write_csvs(photo_rows, financial_rows)
    print(f"\n✅ Done! Written {len(photo_rows)} photo entries and {len(financial_rows)} financial entries.")
    print(f"   Photos:     {PHOTOS_CSV}")
    print(f"   Financials: {FINANCIALS_CSV}")
    print("\nNext steps:")
    print("  git add player_photos.csv 'Player Financials/player_financials.csv'")
    print("  git commit -m 'Populate player photos and financials'")
    print("  git push origin main")


def _write_csvs(photo_rows, financial_rows):
    # Photos
    with open(PHOTOS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["short_name", "photo_url"])
        writer.writeheader()
        for row in photo_rows.values():
            writer.writerow({"short_name": row.get("short_name", ""), "photo_url": row.get("photo_url", "")})

    # Financials
    os.makedirs(os.path.dirname(FINANCIALS_CSV), exist_ok=True)
    with open(FINANCIALS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["short_name", "market_value", "salary"])
        writer.writeheader()
        for row in financial_rows.values():
            writer.writerow({
                "short_name":   row.get("short_name", ""),
                "market_value": row.get("market_value", ""),
                "salary":       row.get("salary", ""),
            })


if __name__ == "__main__":
    main()
