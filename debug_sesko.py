"""Debug Sesko's grading and classification."""
import pandas as pd
import os, sys, importlib.util

# ── Load the app module to reuse its data loading ───────────────────────────
# We can't import app directly (Streamlit), so we'll replicate the critical
# loading logic. Let's find data via the same BESOCCER_DIR.

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

import re as _re

def _parse_player_col(val):
    if pd.isna(val):
        return ("Unknown", "")
    s = str(val).strip()
    m = _re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", s)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    return (s, "")

def _load_league(folder_path, stat_mode):
    dfs = {}
    categories = ["Attacking", "Defending", "Goalkeeping", "Passing"]
    mode_str = f"({stat_mode})"
    for cat in categories:
        # Find the file matching this category and mode
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

total_frames = []
for display_name, folder_name in LEAGUE_FOLDERS.items():
    folder_path = os.path.join(BESOCCER_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    df_total = _load_league(folder_path, "Total")
    if df_total is not None and not df_total.empty:
        df_total["league_display"] = display_name
        total_frames.append(df_total)

df = pd.concat(total_frames, ignore_index=True)
parsed = df["Player"].apply(_parse_player_col)
df["nombre"] = parsed.apply(lambda x: x[0])
df["posicion_detail"] = parsed.apply(lambda x: x[1])
df["posicion"] = df["posicion_detail"].map(POSITION_MAP).fillna("Unknown")

# Convert stat columns
META_COLS = {"Player", "nombre", "posicion_detail", "posicion", "league_display"}
for col in df.columns:
    if col not in META_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Find Sesko
import unicodedata
def _norm(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()

sesko = df[df["nombre"].apply(lambda n: "sesko" in _norm(str(n)))]

print(f"Found {len(sesko)} matching rows:")
for _, r in sesko.iterrows():
    print(f"  {r['nombre']} | pos_detail={r.get('posicion_detail','')} | posicion={r['posicion']} | league={r['league_display']} | apps={r.get('Partidos jugados', '?')}")

if sesko.empty:
    print("No Sesko found!")
    sys.exit(1)

player = sesko.iloc[0]
position = player["posicion"]
league = player["league_display"]
print(f"\nPosition: {position}")
print(f"League: {league}")

# Peer counts
all_peers = df[df["posicion"] == position]
league_peers = all_peers[all_peers["league_display"] == league]
print(f"\nPeer count (all Europe, same position '{position}'): {len(all_peers)}")
print(f"Peer count (same league '{league}'): {len(league_peers)}")

# ── PROFILE_CATEGORIES percentiles (position-level grading) ─────────────
PROFILE_CATEGORIES = {
    "Offensive": ["Goals", "Assists", "Shots", "Shots on Target",
                  "xG", "Key Passes", "Touches in the Box"],
    "Defensive": ["Tackles", "Interceptions", "Clearances",
                  "Shots Blocked (DEF)", "Recoveries"],
    "Passing": ["Att. Passes", "Succ. Passes", "Succ. Long Passes",
                "Forward Passes", "Progressive Passes"],
    "Ball Progression": ["Succ. Progressive Passes", "Succ. Progressive Runs",
                         "Passes to Final 1/3", "Passes into the Box", "Deep Passes"],
    "Possession": ["Succ. Dribbles", "Progressive Runs",
                   "Duels Won", "Ground Duels Won"],
    "Aerial": ["Aerial Duels Won", "Aerial Duels", "% Aerial Duels Won"],
    "Discipline": ["Yellow Cards", "Red Cards", "Fouls"],
}

STRIKER_PROFILE_CATEGORIES = {
    "Finishing": ["Goals", "Non-Penalty Goals", "Shots", "Shots on Target",
                 "xG", "Touches in the Box"],
    "Chance Creation": ["Assists", "Key Passes", "xA", "Shot Assists"],
    "Aerial": ["Aerial Duels Won", "Aerial Duels"],
    "Possession": ["Succ. Dribbles", "Progressive Runs",
                   "Offensive Duels Won", "Fouls Won"],
    "Pressing": ["Recoveries", "Interceptions",
                 "Counter-Pressing Recoveries", "Defensive Pressures"],
    "Discipline": ["Yellow Cards", "Red Cards", "Fouls"],
}

# Use position-specific categories
cats = STRIKER_PROFILE_CATEGORIES

print("\n=== POSITION-LEVEL PERCENTILES (NEW STRIKER CATEGORIES, vs ALL Europe) ===")
for cat, metrics in cats.items():
    avail = [m for m in metrics if m in all_peers.columns]
    if not avail:
        print(f"  {cat}: no metrics available")
        continue
    player_sum = sum(0 if pd.isna(player.get(m, 0)) else (player.get(m, 0) or 0) for m in avail)
    peer_sums = all_peers[avail].fillna(0).sum(axis=1)
    pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100
    if cat == "Discipline":
        pct = 100 - pct
    print(f"  {cat}: {pct:.1f}%  (player_sum={player_sum:.1f}, peer_median={peer_sums.median():.1f}, peer_max={peer_sums.max():.1f})")
    # Show individual metric values
    for m in avail:
        val = 0 if pd.isna(player.get(m, 0)) else (player.get(m, 0) or 0)
        peer_m = all_peers[m].fillna(0)
        m_pct = (peer_m < val).sum() / max(len(peer_m), 1) * 100
        print(f"    {m}: {val} (pctl={m_pct:.1f}%, median={peer_m.median():.1f}, max={peer_m.max():.1f})")

# Overall position percentile
all_pcts = []
for cat, metrics in cats.items():
    avail = [m for m in metrics if m in all_peers.columns]
    if not avail:
        continue
    player_sum = sum(0 if pd.isna(player.get(m, 0)) else (player.get(m, 0) or 0) for m in avail)
    peer_sums = all_peers[avail].fillna(0).sum(axis=1)
    pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100
    if cat == "Discipline":
        pct = 100 - pct
    all_pcts.append(pct)
overall = sum(all_pcts) / len(all_pcts)
print(f"\n  OVERALL POSITION PERCENTILE: {overall:.1f}%")

# ── Role classification ─────────────────────────────────────────────────
STRIKER_ROLE_PROFILES = {
    "Prolific Striker": [
        "Goals", "Non-Penalty Goals", "Header Goals",
        "Shots", "Shots on Target", "Shots in the Box",
        "xG", "NPxG", "xG Difference", "NPxG Difference",
        "Touches in the Box",
        "Offensive Duels", "Offensive Duels Won",
        "Penalties Won", "Fouls Won",
        "% Scoring Effectiveness", "N.P. Goals / Shot",
    ],
    "Target Man": [
        "Aerial Duels", "Aerial Duels Won", "% Aerial Duels Won",
        "Header Goals", "Goals", "Non-Penalty Goals",
        "Shots", "Shots on Target", "xG",
        "Duels", "Duels Won", "% Duels Won",
        "Offensive Duels", "Offensive Duels Won", "% Offensive Duels Won",
        "Fouls Won", "Cards Won",
    ],
    "False 9": [
        "Assists", "Pre-Assists", "Key Passes", "xA", "xA / Key Passes",
        "Shot Assists", "Shot Assists (CAtt)", "Shot on Target Assists",
        "Passes into the Box", "Succ. Passes into the Box", "% Succ. Passes into the Box",
        "Deep Passes", "Passes to Final 1/3", "Succ. Passes to Final 1/3",
        "% Succ. Passes to Final 1/3",
        "Succ. Dribbles", "Dribbles", "% Succ. Dribbles",
        "Progressive Runs", "Succ. Progressive Runs",
        "Goals", "Non-Penalty Goals",
    ],
    "Pressing Forward": [
        "Counter-Pressing Recoveries", "Recoveries", "Opp. Half Recoveries",
        "Final 1/3 Recoveries",
        "Tackles", "Interceptions", "Defensive Pressures",
        "Ground Duels", "Ground Duels Won", "% Ground Duels Won",
        "Defensive Duels", "Defensive Duels Won", "% Defensive Duels Won",
        "Fouls",
    ],
}

print("\n=== ROLE CLASSIFICATION SCORES ===")
pos_peers = all_peers
for role, metrics in STRIKER_ROLE_PROFILES.items():
    avail = [m for m in metrics if m in pos_peers.columns and not m.startswith("% ")]
    if not avail:
        print(f"  {role}: no metrics")
        continue
    metric_pcts = []
    for m in avail:
        val = player.get(m, 0)
        val = 0 if pd.isna(val) else (val or 0)
        peer_vals = pos_peers[m].fillna(0)
        pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
        metric_pcts.append(pct)
    avg_pct = sum(metric_pcts) / len(metric_pcts)
    print(f"  {role}: avg_pct={avg_pct:.1f}% ({len(avail)} metrics)")

# Role grade (using role metrics as a single category)
print("\n=== ROLE-LEVEL GRADE (best role vs all Europe) ===")
# Find best role
best_role = None
best_avg = -1
for role, metrics in STRIKER_ROLE_PROFILES.items():
    avail = [m for m in metrics if m in pos_peers.columns and not m.startswith("% ")]
    if not avail:
        continue
    mpcts = []
    for m in avail:
        val = player.get(m, 0)
        val = 0 if pd.isna(val) else (val or 0)
        peer_vals = pos_peers[m].fillna(0)
        pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
        mpcts.append(pct)
    avg = sum(mpcts) / len(mpcts)
    if avg > best_avg:
        best_avg = avg
        best_role = role

print(f"  Best role: {best_role} (avg={best_avg:.1f}%)")

# Now compute role grade: that role's metrics as a single sum-based percentile
role_metrics = STRIKER_ROLE_PROFILES[best_role]
avail = [m for m in role_metrics if m in pos_peers.columns]
player_sum = sum(0 if pd.isna(player.get(m, 0)) else (player.get(m, 0) or 0) for m in avail)
peer_sums = all_peers[avail].fillna(0).sum(axis=1)
role_pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100
print(f"  Role sum-percentile: {role_pct:.1f}%")
print(f"  (player_sum={player_sum:.1f}, peer_median={peer_sums.median():.1f}, peer_max={peer_sums.max():.1f})")

# Key stats
print(f"\n=== KEY STATS ===")
key_stats = ["Goals", "Non-Penalty Goals", "Shots", "Shots on Target", "xG",
             "Assists", "Key Passes", "Touches in the Box", "Aerial Duels Won",
             "Succ. Dribbles", "Progressive Runs", "Partidos jugados"]
for s in key_stats:
    val = player.get(s, "N/A")
    if s in all_peers.columns:
        med = all_peers[s].fillna(0).median()
        mx = all_peers[s].fillna(0).max()
        print(f"  {s}: {val} (median={med:.1f}, max={mx:.1f})")
    else:
        print(f"  {s}: {val}")
