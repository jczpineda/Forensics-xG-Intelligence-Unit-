"""Show all CM role scores for Garner and Onana on Per 90."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from app import (
    load_data, _classify_percentile_role,
    POSITION_ROLE_PROFILES, CM_ROLE_PROFILES,
)

data = load_data()
df_p90 = data["per90"]
MIN_APPS = 5

for name in ["Garner", "A. Onana"]:
    matches = df_p90[df_p90["nombre"].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"{name} NOT FOUND"); continue
    row = matches.iloc[0]
    pos = row["posicion"]
    apps = row.get("Appearances", 0)
    print(f"\n{'='*60}")
    print(f"{name} [Per 90]  pos={pos}  apps={apps}")

    peers = df_p90[(df_p90["posicion"] == pos) & (df_p90["Appearances"] >= MIN_APPS)]
    print(f"  Peers: {len(peers)}")

    # Show ALL role scores (per-metric averaging, same as _classify_percentile_role)
    for role, metrics in CM_ROLE_PROFILES.items():
        avail = [m for m in metrics if m in peers.columns and not m.startswith("% ")]
        if not avail:
            continue
        metric_pcts = []
        for m in avail:
            val = row.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            pct = (peers[m].fillna(0) < val).sum() / max(len(peers), 1) * 100
            metric_pcts.append(pct)
        avg = sum(metric_pcts) / len(metric_pcts)

        # Also compute WITH % stats included (per-metric averaging)
        avail_all = [m for m in metrics if m in peers.columns]
        metric_pcts_all = []
        for m in avail_all:
            val = row.get(m, 0)
            val = 0 if pd.isna(val) else (val or 0)
            pct = (peers[m].fillna(0) < val).sum() / max(len(peers), 1) * 100
            metric_pcts_all.append(pct)
        avg_all = sum(metric_pcts_all) / len(metric_pcts_all)

        # Old sum-based (with % stats) - what they "used to see"
        avail_sum = [m for m in metrics if m in peers.columns]
        player_sum = sum(0 if pd.isna(row.get(m, 0)) else (row.get(m, 0) or 0) for m in avail_sum)
        peer_sums = peers[avail_sum].fillna(0).sum(axis=1)
        sum_pct = (peer_sums < player_sum).sum() / max(len(peer_sums), 1) * 100

        print(f"  {role:25s}  classify(no%)={avg:.1f}%  grade(w/%)={avg_all:.1f}%  OLD_sum={sum_pct:.1f}%")
