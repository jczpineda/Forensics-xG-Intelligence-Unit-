"""Debug Wharton passing percentile: League vs Europe."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from app import load_data, PROFILE_CATEGORIES, _compute_percentiles

data = load_data()
df_total = data["total"]

name = "Adam Wharton"
row = df_total[df_total["nombre"] == name].iloc[0]
position = row["posicion"]
league = row["league_display"]

print(f"Player: {name}")
print(f"Position: {position}, League: {league}")

# Passing metrics
pass_metrics = PROFILE_CATEGORIES["Passing"]
print(f"\nPassing metrics: {pass_metrics}")

# League peers
lg_peers = df_total[(df_total["posicion"] == position) & (df_total["league_display"] == league)]
all_peers = df_total[df_total["posicion"] == position]
print(f"\nLeague DM peers: {len(lg_peers)}")
print(f"Europe DM peers: {len(all_peers)}")

# Show Wharton's raw values
avail = [m for m in pass_metrics if m in df_total.columns]
print(f"\nWharton passing values:")
for m in avail:
    val = row.get(m, 0)
    print(f"  {m:30s} = {val}")

# Sum-based percentile (how _compute_percentiles works)
player_sum = sum(0 if pd.isna(row.get(m, 0)) else (row.get(m, 0) or 0) for m in avail)
print(f"\nWharton passing SUM: {player_sum}")

# League peer sums
lg_sums = lg_peers[avail].fillna(0).sum(axis=1)
print(f"\nLeague peer sums (sorted desc, top 15):")
lg_sorted = lg_sums.sort_values(ascending=False)
for i, (idx, s) in enumerate(lg_sorted.items()):
    p_name = df_total.at[idx, "nombre"]
    marker = " <-- WHARTON" if p_name == name else ""
    print(f"  {i+1:3d}. {p_name:30s} sum={s:8.1f}{marker}")
    if i >= 14:
        break

lg_pct = (lg_sums < player_sum).sum() / max(len(lg_sums), 1) * 100
print(f"\nLeague passing percentile: {lg_pct:.1f}")

# Europe peer sums
all_sums = all_peers[avail].fillna(0).sum(axis=1)
print(f"\nEurope peer sums (sorted desc, top 15):")
all_sorted = all_sums.sort_values(ascending=False)
for i, (idx, s) in enumerate(all_sorted.items()):
    p_name = df_total.at[idx, "nombre"]
    marker = " <-- WHARTON" if p_name == name else ""
    print(f"  {i+1:3d}. {p_name:30s} sum={s:8.1f}{marker}")
    if i >= 14:
        break

all_pct = (all_sums < player_sum).sum() / max(len(all_sums), 1) * 100
print(f"\nEurope passing percentile: {all_pct:.1f}")

# Show distribution
print(f"\nLeague sum stats: mean={lg_sums.mean():.1f}, median={lg_sums.median():.1f}, max={lg_sums.max():.1f}")
print(f"Europe sum stats: mean={all_sums.mean():.1f}, median={all_sums.median():.1f}, max={all_sums.max():.1f}")

# Show where Wharton ranks
lg_rank = (lg_sums > player_sum).sum() + 1
all_rank = (all_sums > player_sum).sum() + 1
print(f"\nWharton league rank: {lg_rank}/{len(lg_sums)}")
print(f"Wharton europe rank: {all_rank}/{len(all_sums)}")

# Check appearances of top league peers
print(f"\nLeague peers with higher passing sums - appearances:")
for idx in lg_sorted.index:
    s = lg_sums[idx]
    if s <= player_sum:
        break
    p_name = df_total.at[idx, "nombre"]
    apps = df_total.at[idx, "Appearances"]
    mins = df_total.at[idx, "Time Played"]
    print(f"  {p_name:30s} sum={s:8.1f}  apps={apps}  mins={mins}")
