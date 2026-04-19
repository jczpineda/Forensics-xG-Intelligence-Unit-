"""Debug GK grading: Bayindir vs Lammens."""
import pandas as pd, os, re, unicodedata

BESOCCER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LEAGUE_FOLDERS = {
    "Premier League": "Premier League BeSoccer Files",
    "LaLiga": "LaLiga BeSoccer Files",
    "Bundesliga": "Bundesliga BeSoccer Files",
    "Ligue 1": "Ligue 1 BeSoccer Files",
    "Serie A": "Serie A BeSoccer Files",
    "Primeira Liga": "Primeira Liga BeSoccer Files",
}
POSITION_MAP = {
    "GK": "Goalkeeper",
    "CB": "Centre-Back", "LB": "Full-Back", "RB": "Full-Back", "RWB": "Full-Back", "LWB": "Full-Back",
    "DM": "Defensive Midfield",
    "CM": "Central Midfield", "LM": "Central Midfield", "RM": "Central Midfield",
    "CAM": "Attacking Midfield", "RAM": "Attacking Midfield", "LAM": "Attacking Midfield",
    "ST": "Striker", "CF": "Striker", "LW": "Winger", "RW": "Winger",
}

def _parse_player_col(val):
    if pd.isna(val):
        return ("Unknown", "")
    s = str(val).strip()
    m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", s)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    return (s, "")

def _load_league(folder_path, stat_mode):
    dfs = {}
    categories = ["Attacking", "Defending", "Goalkeeping", "Passing"]
    mode_str = f"({stat_mode})"
    for cat in categories:
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".csv"):
                continue
            if cat in fname and mode_str in fname:
                fpath = os.path.join(folder_path, fname)
                try:
                    dfs[cat] = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False)
                except Exception:
                    pass
                break
    if not dfs:
        return None
    base_cat = list(dfs.keys())[0]
    merged = dfs[base_cat].copy()
    for cat in list(dfs.keys())[1:]:
        other = dfs[cat]
        existing = set(merged.columns)
        new_cols = [c for c in other.columns if c not in existing or c == "Player"]
        merged = merged.merge(other[new_cols], on="Player", how="outer")
    return merged

def _norm(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()

# Load both Total and Per 90
def load_all(mode):
    frames = []
    for display_name, folder_name in LEAGUE_FOLDERS.items():
        folder_path = os.path.join(BESOCCER_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        df = _load_league(folder_path, mode)
        if df is not None and not df.empty:
            df["league_display"] = display_name
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    parsed = combined["Player"].apply(_parse_player_col)
    combined["nombre"] = parsed.apply(lambda x: x[0])
    combined["posicion_detail"] = parsed.apply(lambda x: x[1])
    combined["posicion"] = combined["posicion_detail"].map(POSITION_MAP).fillna("Unknown")
    META = {"Player", "nombre", "posicion_detail", "posicion", "league_display"}
    for col in combined.columns:
        if col not in META:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined

print("Loading Total stats...")
df_total = load_all("Total")
print("Loading Per 90 stats...")
df_per90 = load_all("Per 90")

# Find both keepers
for name_search in ["bayindir", "lammens"]:
    print(f"\n{'='*60}")
    print(f"Searching for: {name_search}")
    matches_t = df_total[df_total["nombre"].apply(lambda n: name_search in _norm(str(n)))]
    matches_p = df_per90[df_per90["nombre"].apply(lambda n: name_search in _norm(str(n)))]
    if matches_t.empty:
        print(f"  NOT FOUND in total data!")
        continue
    p = matches_t.iloc[0]
    pp = matches_p.iloc[0] if not matches_p.empty else None
    print(f"  Name: {p['nombre']}")
    print(f"  Position: {p['posicion']} | Detail: {p.get('posicion_detail','')}")
    print(f"  League: {p['league_display']}")

    position = p["posicion"]
    all_gk_total = df_total[df_total["posicion"] == "Goalkeeper"]
    all_gk_per90 = df_per90[df_per90["posicion"] == "Goalkeeper"]
    
    print(f"  GK peers (total): {len(all_gk_total)}")
    print(f"  GK peers (per90): {len(all_gk_per90)}")

    # Key GK stats - Total
    gk_stats = ["Saves", "Goals Conceded", "Goals Prevented", "% Saves / Shots on Target Faced",
                "% Clean Sheets", "High Claims", "Saved Penalties",
                "Att. Passes", "Succ. Passes", "Succ. Long Passes", "% Succ. Passes",
                "Recoveries", "Clearances", "Interceptions",
                "Aerial Duels", "Aerial Duels Won", "% Aerial Duels Won",
                "xG against (shots on target)", "Launch %",
                "Counter-Pressing Recoveries", "Forward Passes",
                "% Succ. Long Passes"]
    
    print(f"\n  --- TOTAL STATS ---")
    for s in gk_stats:
        val = p.get(s, "N/A")
        if s in all_gk_total.columns:
            med = all_gk_total[s].fillna(0).median()
            mx = all_gk_total[s].fillna(0).max()
            pctl = (all_gk_total[s].fillna(0) < (0 if pd.isna(val) else (val or 0))).sum() / max(len(all_gk_total), 1) * 100
            print(f"    {s}: {val} (pctl={pctl:.1f}%, med={med:.1f}, max={mx:.1f})")
    
    if pp is not None:
        print(f"\n  --- PER 90 STATS ---")
        for s in gk_stats:
            val = pp.get(s, "N/A")
            if s in all_gk_per90.columns:
                med = all_gk_per90[s].fillna(0).median()
                mx = all_gk_per90[s].fillna(0).max()
                pctl = (all_gk_per90[s].fillna(0) < (0 if pd.isna(val) else (val or 0))).sum() / max(len(all_gk_per90), 1) * 100
                print(f"    {s}: {val} (pctl={pctl:.1f}%, med={med:.1f}, max={mx:.1f})")

    # Role classification scores (per-metric percentile avg, skipping %)
    GOALKEEPER_ROLE_PROFILES = {
        "Line-Holding Keeper": [
            "Saves", "% Saves / Shots on Target Faced", "Goals Prevented",
            "Goals Conceded", "Saved Penalties", "% Clean Sheets",
            "xG against (shots on target)", "High Claims",
            "Aerial Duels", "Aerial Duels Won", "% Aerial Duels Won",
        ],
        "No-Nonsense GK": [
            "Clearances", "Launch %",
            "Succ. Long Passes", "% Succ. Long Passes",
            "High Claims",
            "Aerial Duels", "Aerial Duels Won", "% Aerial Duels Won",
            "Saves", "Goals Prevented",
        ],
        "Sweeper Keeper": [
            "Recoveries", "Counter-Pressing Recoveries",
            "Interceptions", "Clearances",
            "Aerial Duels", "Aerial Duels Won", "% Aerial Duels Won",
        ],
        "Ball-Playing GK": [
            "Att. Passes", "Succ. Passes", "% Succ. Passes",
            "Forward Passes",
            "Succ. Long Passes", "% Succ. Long Passes",
        ],
        "Balanced Keeper": [
            "Saves", "% Saves / Shots on Target Faced", "Goals Prevented",
            "% Clean Sheets",
            "High Claims", "Aerial Duels Won",
            "Att. Passes", "Succ. Passes", "% Succ. Passes",
            "Recoveries", "Clearances", "Interceptions",
        ],
    }
    
    # Role scores using TOTAL
    print(f"\n  --- ROLE SCORES (TOTAL, per-metric pctl avg) ---")
    for role, metrics in GOALKEEPER_ROLE_PROFILES.items():
        avail = [m for m in metrics if m in all_gk_total.columns and not m.startswith("% ")]
        if not avail:
            continue
        mpcts = []
        for m in avail:
            val = p.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            peer_vals = all_gk_total[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            mpcts.append(pct)
        avg = sum(mpcts) / len(mpcts)
        print(f"    {role}: {avg:.1f}% ({len(avail)} metrics)")
    
    # Role scores using PER 90
    if pp is not None:
        print(f"\n  --- ROLE SCORES (PER 90, per-metric pctl avg) ---")
        for role, metrics in GOALKEEPER_ROLE_PROFILES.items():
            avail = [m for m in metrics if m in all_gk_per90.columns and not m.startswith("% ")]
            if not avail:
                continue
            mpcts = []
            for m in avail:
                val = pp.get(m, 0)
                val = 0 if pd.isna(val) else (val or 0)
                peer_vals = all_gk_per90[m].fillna(0)
                pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                mpcts.append(pct)
            avg = sum(mpcts) / len(mpcts)
            print(f"    {role}: {avg:.1f}% ({len(avail)} metrics)")

    # Per-90 role grade (sum-based, single category = all role metrics)
    if pp is not None:
        # find best role per90
        best_role_p90, best_avg_p90 = None, -1
        for role, metrics in GOALKEEPER_ROLE_PROFILES.items():
            avail = [m for m in metrics if m in all_gk_per90.columns and not m.startswith("% ")]
            if not avail:
                continue
            mpcts = []
            for m in avail:
                val = pp.get(m, 0)
                val = 0 if pd.isna(val) else (val or 0)
                peer_vals = all_gk_per90[m].fillna(0)
                pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                mpcts.append(pct)
            avg = sum(mpcts) / len(mpcts)
            if avg > best_avg_p90:
                best_avg_p90 = avg
                best_role_p90 = role
        
        print(f"\n  Best P90 role: {best_role_p90} ({best_avg_p90:.1f}%)")
        
        # Sum-based grade for that role (strip % stats from sum)
        role_metrics = GOALKEEPER_ROLE_PROFILES[best_role_p90]
        avail = [m for m in role_metrics if m in all_gk_per90.columns and not m.startswith("% ")]
        player_sum = sum(0 if pd.isna(pp.get(m, 0)) else (pp.get(m, 0) or 0) for m in avail)
        peer_sums = all_gk_per90[avail].fillna(0).sum(axis=1)
        role_pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100
        print(f"  Role sum-percentile (P90): {role_pct:.1f}%")
        print(f"  (player_sum={player_sum:.2f}, peer_med={peer_sums.median():.2f}, peer_max={peer_sums.max():.2f})")
        # Show each metric's contribution
        for m in avail:
            val = pp.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            print(f"    {m}: {val:.2f}")
