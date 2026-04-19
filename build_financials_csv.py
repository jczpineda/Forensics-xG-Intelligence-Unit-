"""
Pre-fetch Transfermarkt market values and Capology salaries for every
player in the Opta database and save to player_financials.csv.

Run once (takes a while due to rate-limiting):
    python build_financials_csv.py

The CSV is then consumed by app.py for instant lookups in Player Lab
and Player Profile.
"""

import os, re, time, json, urllib.parse, urllib.request, unicodedata, csv
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ───────────────────────────────────────────────────────────────────
OPTA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Forensics xG Opta Data")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "player_financials.csv")

LEAGUE_FOLDERS = {
    "Premier League": "English Premier League",
    "LaLiga": "LaLiga",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Serie A": "Serie A",
    "Primeira Liga": "Primeira Liga",
}

_TM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY_BETWEEN_REQUESTS = 2.5  # seconds between web requests


# ── Helpers (mirrored from app.py) ───────────────────────────────────────────

def _normalize_name(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _resolve_full_name(short_name, team=None):
    try:
        q = urllib.parse.quote(short_name)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        players = data.get("player")
        if not players:
            return short_name
        if team:
            team_lower = team.lower().replace(" fc", "").replace("fc ", "").strip()
            for p in players:
                p_team = (p.get("strTeam") or "").lower().replace(" fc", "").replace("fc ", "").strip()
                if (p.get("strSport") or "").lower() == "soccer" and (
                    team_lower in p_team or p_team in team_lower
                ):
                    return p.get("strPlayer") or short_name
        for p in players:
            if (p.get("strSport") or "").lower() == "soccer":
                return p.get("strPlayer") or short_name
        return players[0].get("strPlayer") or short_name
    except Exception:
        return short_name


def _fetch_transfermarkt_value(player_name, team=None):
    try:
        query = urllib.parse.quote(player_name)
        url = (f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/"
               f"schnellsuche?query={query}")
        resp = requests.get(url, headers=_TM_HEADERS, timeout=10)
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
                team_norm = _normalize_name(team)
                if team_norm and team_norm in row_text:
                    return mv
            if best is None:
                best = mv
        return best
    except Exception:
        return None


_capology_index = None

def _fetch_capology_search_index():
    global _capology_index
    if _capology_index is not None:
        return _capology_index
    try:
        resp = requests.get(
            "https://www.capology.com/static/files/search_players.json",
            headers=_TM_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            _capology_index = resp.json()
            return _capology_index
    except Exception:
        pass
    return []


def _capology_find_slug(player_name, team=None):
    index = _fetch_capology_search_index()
    if not index:
        return None
    target = _normalize_name(player_name)
    candidates = []
    for entry in index:
        name = _normalize_name(entry.get("name", ""))
        if name == target:
            candidates.append(entry)
    if not candidates:
        parts = target.split()
        if len(parts) >= 2:
            surname = parts[-1]
            initial = parts[0].rstrip(".")
            for entry in index:
                name = _normalize_name(entry.get("name", ""))
                name_parts = name.split()
                if len(name_parts) >= 2 and name_parts[-1] == surname:
                    if name_parts[0].startswith(initial):
                        candidates.append(entry)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].get("link")
    if team:
        team_norm = _normalize_name(team)
        for c in candidates:
            club_url = (c.get("club") or "").lower()
            if team_norm.replace(" ", "-") in club_url or team_norm.replace(" ", "") in club_url:
                return c.get("link")
    return candidates[0].get("link")


def _fetch_capology_salary(player_name, team=None):
    try:
        slug = _capology_find_slug(player_name, team)
        if not slug:
            return None
        url = f"https://www.capology.com{slug}/"
        resp = requests.get(url, headers=_TM_HEADERS, timeout=10)
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
            elif raw >= 1_000:
                return f"€{raw / 1_000:,.0f}K"
            else:
                return f"€{raw:,}"
        return None
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def collect_players():
    """Build deduplicated list of (short_name, team, league) from Opta data."""
    seen = set()
    players = []
    for display_name, folder_name in LEAGUE_FOLDERS.items():
        folder_path = os.path.join(OPTA_DIR, folder_name)
        csv_path = os.path.join(folder_path, "jugadores_seasonstats.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP {display_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        if df.empty or "nombre" not in df.columns:
            print(f"  SKIP {display_name}: empty or missing 'nombre' column")
            continue
        # Filter out players with 0 minutes
        if "Time Played" in df.columns:
            df = df[pd.to_numeric(df["Time Played"], errors="coerce").fillna(0) > 0]
        for _, row in df.iterrows():
            name = str(row.get("nombre", "")).strip()
            team = str(row.get("equipo", "")).strip()
            if not name or name == "nan":
                continue
            key = (name, display_name)
            if key not in seen:
                seen.add(key)
                players.append({"short_name": name, "team": team, "league": display_name})
    return players


def load_existing():
    """Load already-processed rows so we can resume."""
    if not os.path.exists(OUTPUT_CSV):
        return {}
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    existing = {}
    for _, r in df.iterrows():
        key = (r["short_name"], r["league"])
        existing[key] = r.to_dict()
    return existing


def main():
    print("Collecting player list from Opta data…")
    players = collect_players()
    print(f"Found {len(players)} unique players across all leagues.\n")

    # Load existing progress
    existing = load_existing()
    print(f"Already have {len(existing)} entries. Resuming…\n")

    # Pre-fetch Capology index
    print("Loading Capology search index…")
    _fetch_capology_search_index()
    print(f"Capology index: {len(_capology_index or [])} entries\n")

    done = 0
    skipped = 0
    total = len(players)

    # Open CSV in append mode
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "short_name", "team", "league", "full_name",
            "market_value", "salary",
        ])
        if write_header:
            writer.writeheader()

        for i, p in enumerate(players):
            key = (p["short_name"], p["league"])
            if key in existing:
                skipped += 1
                continue

            short = p["short_name"]
            team = p["team"]
            league = p["league"]

            # Resolve full name
            full_name = _resolve_full_name(short, team=team)
            time.sleep(0.3)

            # Transfermarkt
            mv = _fetch_transfermarkt_value(full_name, team=team)
            time.sleep(DELAY_BETWEEN_REQUESTS)

            # Capology (no delay needed for slug lookup, only for page fetch)
            salary = _fetch_capology_salary(full_name, team=team)
            time.sleep(DELAY_BETWEEN_REQUESTS)

            row = {
                "short_name": short,
                "team": team,
                "league": league,
                "full_name": full_name,
                "market_value": mv or "",
                "salary": salary or "",
            }
            writer.writerow(row)
            f.flush()
            done += 1

            status = f"[{i+1}/{total}] {short} ({league})"
            mv_str = mv or "N/A"
            sal_str = salary or "N/A"
            print(f"  {status} -> MV: {mv_str} | Salary: {sal_str}")

    print(f"\nDone! Processed {done} new, skipped {skipped} existing.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
