"""Debug Wharton & Tonali role classification + percentile breakdown."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from app import (load_data, POSITION_ROLE_PROFILES, PROFILE_CATEGORIES,
                 _classify_percentile_role, _compute_percentiles)

data = load_data()
df_total = data["total"]
df_per90 = data["per90"]

for name in ["Adam Wharton", "Sandro Tonali"]:
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    row_t = df_total[df_total["nombre"] == name]
    if row_t.empty:
        print("  NOT FOUND in totals")
        continue
    row_t = row_t.iloc[0]
    position = row_t["posicion"]
    pos_detail = row_t["posicion_detail"]
    print(f"  Position: {pos_detail} -> {position}")
    
    # Role from totals
    role_t = _classify_percentile_role(dict(row_t), df_total, position)
    print(f"  Role (Total): {role_t}")
    
    # Show all role scores from totals
    profiles = POSITION_ROLE_PROFILES.get(position, {})
    pos_peers = df_total[df_total["posicion"] == position]
    print(f"  Peers (Total): {len(pos_peers)}")
    
    print(f"\n  --- Role Scores (TOTAL data) ---")
    for role, metrics in profiles.items():
        avail = [m for m in metrics if m in pos_peers.columns and not m.startswith("% ")]
        metric_pcts = []
        for m in avail:
            val = row_t.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            peer_vals = pos_peers[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            metric_pcts.append((m, val, round(pct, 1)))
        avg_pct = sum(p for _, _, p in metric_pcts) / len(metric_pcts) if metric_pcts else 0
        print(f"\n  {role}: avg={avg_pct:.1f}")
        for m, v, p in metric_pcts:
            print(f"    {m:40s} val={v:8.2f}  pctl={p:5.1f}")
    
    # Now per-90
    row_p = df_per90[df_per90["nombre"] == name]
    if row_p.empty:
        print(f"\n  NOT FOUND in per90")
        continue
    row_p = row_p.iloc[0]
    
    role_p = _classify_percentile_role(dict(row_p), df_per90, position)
    pos_peers_p = df_per90[df_per90["posicion"] == position]
    print(f"\n  Role (Per 90): {role_p}")
    print(f"  Peers (Per 90): {len(pos_peers_p)}")
    
    print(f"\n  --- Role Scores (PER 90 data) ---")
    for role, metrics in profiles.items():
        avail = [m for m in metrics if m in pos_peers_p.columns and not m.startswith("% ")]
        metric_pcts = []
        for m in avail:
            val = row_p.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            peer_vals = pos_peers_p[m].fillna(0)
            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
            metric_pcts.append((m, val, round(pct, 1)))
        avg_pct = sum(p for _, _, p in metric_pcts) / len(metric_pcts) if metric_pcts else 0
        print(f"\n  {role}: avg={avg_pct:.1f}")
        for m, v, p in metric_pcts:
            print(f"    {m:40s} val={v:8.2f}  pctl={p:5.1f}")
    
    # Also show PROFILE_CATEGORIES percentiles (what shows in Percentile Summary)
    print(f"\n  --- Percentile Summary Categories (TOTAL) ---")
    league = row_t.get("league_display", "")
    lg_peers = df_total[(df_total["posicion"] == position) & (df_total["league_display"] == league)]
    pcts_t = _compute_percentiles(dict(row_t), lg_peers, PROFILE_CATEGORIES)
    for cat, pct in pcts_t.items():
        print(f"    {cat:25s} {pct:5.1f}")
    
    print(f"\n  --- Percentile Summary Categories (PER 90) ---")
    lg_peers_p = df_per90[(df_per90["posicion"] == position) & (df_per90["league_display"] == league)]
    pcts_p = _compute_percentiles(dict(row_p), lg_peers_p, PROFILE_CATEGORIES)
    for cat, pct in pcts_p.items():
        print(f"    {cat:25s} {pct:5.1f}")
    
    # Show appearances/time played
    apps = row_t.get("Appearances", None)
    mins = row_t.get("Time Played", None)
    print(f"\n  Appearances: {apps}, Time Played: {mins}")
