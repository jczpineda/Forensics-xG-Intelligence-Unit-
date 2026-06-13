"""
Build / gap-fill player_financials.csv — Transfermarkt market values and
Capology salaries for every player in the Opta database.

Output format (exactly what app.py consumes):
    short_name, market_value, salary

GAP-FILL BY DEFAULT: existing non-empty values are kept and never re-fetched;
only missing market_value / salary are looked up.  The script checkpoints to
disk every few players and is safe to interrupt and re-run — it resumes from
whatever is already in the CSV.

So to fill the ~72% of missing market values, just run it again:
    python build_financials_csv.py

It must run from a normal (residential) internet connection — Transfermarkt
blocks cloud/datacenter IPs, which is why the original bulk build came back
mostly empty.  Expect a couple of hours with rate-limiting on a fresh fill.
"""

import os
import re
import csv
import json
import time
import urllib.parse
import urllib.request
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ───────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # …/Player Financials
_REPO_DIR = os.path.dirname(_THIS_DIR)                            # …/football-analytics model
_LOCAL_DIR = os.path.join(_REPO_DIR, "..", "..", "Forensics xG Opta Data")
# Mirror app.py: league folders live in the repo root on Streamlit Cloud,
# otherwise in the local Opta data directory.
OPTA_DIR = _REPO_DIR if os.path.isdir(os.path.join(_REPO_DIR, "Bundesliga")) else _LOCAL_DIR

OUTPUT_CSV = os.path.join(_THIS_DIR, "player_financials.csv")
# `team` is stored so the app can match by name + team and never apply a famous
# player's value to an obscure namesake (e.g. two "M. Saka").
FIELDNAMES = ["short_name", "team", "market_value", "salary", "age"]

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

DELAY = 3.0              # base seconds between web requests (be gentle)
MAX_RETRIES = 3         # retries on non-200 / network error, with backoff
CHECKPOINT_EVERY = 20   # rewrite the CSV after this many newly-fetched players

# Tokens that count as "no value" when read back from an existing CSV.
_EMPTY_TOKENS = {"", "nan", "none", "n/a", "na"}


def _present(v):
    """True when *v* is a real, non-empty value (not '', NaN or 'nan')."""
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


# ── Scrapers ─────────────────────────────────────────────────────────────────
# NOTE: we deliberately do NOT resolve full names via TheSportsDB — its search
# returned wrong players for abbreviated names (e.g. "B. Saka" -> "Fabrice
# N'Sakala").  Transfermarkt and Capology both fuzzy-match the short name and we
# disambiguate by team, which is far more accurate.

def _http_get(url):
    """GET with retry + backoff; returns Response or None."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=_TM_HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            # 403/429/5xx — back off progressively (likely rate-limited).
            time.sleep(DELAY * (attempt + 2))
        except Exception:
            time.sleep(DELAY * (attempt + 2))
    return None


def _query_variants(name):
    """Search strings to try on Transfermarkt, most specific first."""
    variants = [name]
    parts = name.replace(".", " ").split()
    if len(parts) >= 2:
        variants.append(parts[-1])           # surname only (team match disambiguates)
    seen, out = set(), []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _extract_age(row):
    """Player age from a Transfermarkt quick-search row, or None.

    The 'Date of birth/Age' cell reads like 'Jun 28, 1990 (35)'; fall back to a
    plausible standalone integer in a centered cell."""
    txt = " ".join(c.get_text(" ", strip=True) for c in row.find_all("td"))
    m = re.search(r"\((\d{2})\)", txt)
    if m and 14 <= int(m.group(1)) <= 50:
        return int(m.group(1))
    for c in row.find_all("td", class_="zentriert"):
        t = c.get_text(strip=True)
        if t.isdigit() and 14 <= int(t) <= 50:
            return int(t)
    return None


def _fetch_transfermarkt(player_name, team=None):
    """(market_value, age) from Transfermarkt quick search.

    market_value like '€75.00m'; age is an int or None.  Prefers the row whose
    text contains the team, else the first row with a value."""
    for q in _query_variants(player_name):
        url = ("https://www.transfermarkt.com/schnellsuche/ergebnis/"
               f"schnellsuche?query={urllib.parse.quote(q)}")
        resp = _http_get(url)
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table", class_="items")
        if not tables:
            continue
        rows = tables[0].find_all("tr", class_=["odd", "even"])
        best = best_age = None
        for row in rows:
            value_cell = row.find("td", class_=lambda c: c and "rechts" in c and "hauptlink" in c)
            if not value_cell:
                continue
            mv = value_cell.get_text(strip=True)
            if not mv or mv == "-":
                continue
            age = _extract_age(row)
            if team:
                row_text = " ".join(c.get_text(strip=True).lower() for c in row.find_all("td"))
                if _normalize_name(team) and _normalize_name(team) in row_text:
                    return mv, age
            if best is None:
                best, best_age = mv, age
        if best:
            return best, best_age
    return None, None


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
    _capology_index = []
    return _capology_index


def _capology_find_slug(player_name, team=None):
    index = _fetch_capology_search_index()
    if not index:
        return None
    target = _normalize_name(player_name)
    candidates = [e for e in index if _normalize_name(e.get("name", "")) == target]
    if not candidates:
        parts = target.split()
        if len(parts) >= 2:
            surname, initial = parts[-1], parts[0].rstrip(".")
            for e in index:
                np_ = _normalize_name(e.get("name", "")).split()
                if len(np_) >= 2 and np_[-1] == surname and np_[0].startswith(initial):
                    candidates.append(e)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].get("link")
    if team:
        tn = _normalize_name(team)
        for c in candidates:
            club = (c.get("club") or "").lower()
            if tn.replace(" ", "-") in club or tn.replace(" ", "") in club:
                return c.get("link")
    return candidates[0].get("link")


def _fetch_capology_salary(player_name, team=None):
    """Gross annual salary incl. bonus (e.g. '€10.54M') from Capology."""
    try:
        slug = _capology_find_slug(player_name, team)
        if not slug:
            return None
        resp = _http_get(f"https://www.capology.com{slug}/")
        if resp is None:
            return None
        m_base = re.search(r'"annual_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
        if not m_base:
            return None
        raw = int(m_base.group(1))
        m_bonus = re.search(r'"bonus_gross_eur"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"', resp.text)
        if m_bonus:
            raw += int(m_bonus.group(1))
        if raw >= 1_000_000:
            return f"€{raw / 1_000_000:,.2f}M"
        if raw >= 1_000:
            return f"€{raw / 1_000:,.0f}K"
        return f"€{raw:,}"
    except Exception:
        return None


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
    """Read existing CSV as {short_name: {market_value, salary}} (strings)."""
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        return {}
    # keep_default_na=False so the literal string 'nan' isn't turned into NaN.
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        sn = (r.get("short_name") or "").strip()
        if not sn:
            continue
        mv = (r.get("market_value") or "").strip()
        sal = (r.get("salary") or "").strip()
        age = (r.get("age") or "").strip()
        team = (r.get("team") or "").strip()
        out[sn] = {
            "team": team,
            "market_value": mv if _present(mv) else "",
            "salary": sal if _present(sal) else "",
            "age": age if _present(age) else "",
        }
    return out


def write_csv(rows, order):
    """Write all rows to CSV in *order* (list of short_names)."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for sn in order:
            rec = rows.get(sn, {})
            w.writerow({
                "short_name": sn,
                "team": rec.get("team", "") or "",
                "market_value": rec.get("market_value", "") or "",
                "salary": rec.get("salary", "") or "",
                "age": rec.get("age", "") or "",
            })


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
    have_mv = sum(1 for v in rows.values() if _present(v.get("market_value")))
    have_sal = sum(1 for v in rows.values() if _present(v.get("salary")))
    print(f"Existing CSV: {len(rows)} rows — {have_mv} market values, {have_sal} salaries.")

    todo = [p for p in players
            if not _present(rows.get(p["short_name"], {}).get("market_value"))
            or not _present(rows.get(p["short_name"], {}).get("salary"))
            or not _present(rows.get(p["short_name"], {}).get("age"))]
    print(f"Need to fetch (missing MV, salary and/or age): {len(todo)} players.\n")
    if not todo:
        print("Nothing to fill — every player already has both values.")
        write_csv(rows, order)
        return

    print("Loading Capology search index…")
    _fetch_capology_search_index()
    print(f"Capology index: {len(_capology_index or [])} entries\n")

    fetched = 0
    for i, p in enumerate(players):
        sn, team = p["short_name"], p["team"]
        cur = rows.get(sn, {"team": team, "market_value": "", "salary": "", "age": ""})
        need_mv = not _present(cur.get("market_value"))
        need_sal = not _present(cur.get("salary"))
        need_age = not _present(cur.get("age"))
        if not need_mv and not need_sal and not need_age:
            rows[sn] = cur
            continue

        # Pass the short name + team straight to the scrapers.  Both
        # Transfermarkt's search and Capology's index do their own fuzzy
        # matching and we disambiguate by team — this is far more reliable
        # than TheSportsDB full-name resolution, which returned wrong players
        # (e.g. "B. Saka" -> "Fabrice N'Sakala", "A. Wharton" -> "Scott Wharton").
        mv = cur.get("market_value", "") or ""
        sal = cur.get("salary", "") or ""
        age = cur.get("age", "") or ""
        if need_mv or need_age:
            new_mv, new_age = _fetch_transfermarkt(sn, team=team)
            if need_mv and new_mv:
                mv = new_mv
            if new_age:
                age = str(new_age)
            time.sleep(DELAY)
        if need_sal:
            sal = _fetch_capology_salary(sn, team=team) or ""
            time.sleep(DELAY)

        rows[sn] = {"team": team, "market_value": mv, "salary": sal, "age": age}
        fetched += 1
        print(f"  [{i + 1}/{len(players)}] {sn} -> "
              f"MV: {mv or 'N/A'} | Salary: {sal or 'N/A'} | Age: {age or 'N/A'}")

        if fetched % CHECKPOINT_EVERY == 0:
            write_csv(rows, order)
            print(f"    …checkpoint saved ({fetched} fetched so far)")

    write_csv(rows, order)
    final_mv = sum(1 for v in rows.values() if _present(v.get("market_value")))
    final_sal = sum(1 for v in rows.values() if _present(v.get("salary")))
    print(f"\nDone. Fetched {fetched} this run. "
          f"Now {final_mv} market values, {final_sal} salaries across {len(order)} players.")
    print(f"Output: {OUTPUT_CSV}")


def refresh_values():
    """Re-fetch market value + age for EVERY player (overwriting), keeping the
    existing Capology salary.  Use when Transfermarkt values have moved — the
    default gap-fill never re-fetches a value it already has.

        python build_financials_csv.py refresh
    """
    print("Refreshing ALL Transfermarkt market values + ages…")
    players = collect_players()
    order = [p["short_name"] for p in players]
    rows = load_existing()
    changed = 0
    for i, p in enumerate(players):
        sn, team = p["short_name"], p["team"]
        cur = rows.get(sn, {"team": team, "market_value": "", "salary": "", "age": ""})
        new_mv, new_age = _fetch_transfermarkt(sn, team=team)
        time.sleep(DELAY)
        old_mv = cur.get("market_value", "")
        rows[sn] = {
            "team": team,
            "market_value": new_mv or old_mv,           # keep old if fetch failed
            "salary": cur.get("salary", ""),            # salaries unchanged
            "age": str(new_age) if new_age else cur.get("age", ""),
        }
        if new_mv and new_mv != old_mv:
            changed += 1
            print(f"  [{i + 1}/{len(players)}] {sn}: {old_mv or 'N/A'} -> {new_mv}"
                  f"  (age {rows[sn]['age'] or 'N/A'})")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            write_csv(rows, order)
            print(f"    …checkpoint ({i + 1}/{len(players)}, {changed} changed)")
    write_csv(rows, order)
    print(f"\nDone. {changed} market values changed. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        refresh_values()
    else:
        main()
