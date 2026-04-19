"""Debug grade changes for James Garner and A. Onana after _compute_percentiles refactor."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from app import (
    load_data, _classify_percentile_role, _compute_percentiles,
    _compute_overall_percentile, _compute_player_grades, _percentile_to_grade,
    POSITION_ROLE_PROFILES, PROFILE_CATEGORIES, GK_PROFILE_CATEGORIES,
    STRIKER_PROFILE_CATEGORIES, WINGER_PROFILE_CATEGORIES,
)
_MIN_APPS_P90 = 5

data = load_data()
df_total = data["total"]
df_p90 = data["per90"]

PLAYERS = ["Garner", "A. Onana"]

for name in PLAYERS:
    print("=" * 70)
    # Search in both total and p90
    for label, df in [("Total", df_total), ("Per 90", df_p90)]:
        matches = df[df["nombre"].str.contains(name, case=False, na=False)]
        if matches.empty:
            print(f"{name} NOT FOUND in {label}")
            continue
        row = matches.iloc[0]
        pos = row["posicion"]
        league = row.get("league_display", "")
        apps = row.get("Appearances", 0)

        print(f"\n{name} [{label}]  pos={pos}  league={league}  apps={apps}")

        if label == "Per 90" and apps < _MIN_APPS_P90:
            print(f"  SKIPPED — below {_MIN_APPS_P90} apps threshold")
            continue

        # Pick appropriate peers
        if label == "Per 90":
            peers = df[
                (df["posicion"] == pos) &
                (df["Appearances"] >= _MIN_APPS_P90)
            ]
        else:
            peers = df[df["posicion"] == pos]

        league_peers = peers[peers["league_display"] == league]
        all_peers = peers

        # Position-specific categories
        _POS_CATS = {
            "Goalkeeper": GK_PROFILE_CATEGORIES,
            "Striker": STRIKER_PROFILE_CATEGORIES,
            "Winger": WINGER_PROFILE_CATEGORIES,
        }
        pos_cats = _POS_CATS.get(pos, PROFILE_CATEGORIES)

        # Classify role
        role = _classify_percentile_role(dict(row), df if label == "Total" else df, pos)
        print(f"  Role: {role}")

        # Position grade (overall)
        pos_pcts = _compute_percentiles(dict(row), all_peers, pos_cats)
        pos_ov_pct = sum(pos_pcts.values()) / max(len(pos_pcts), 1)
        pos_grade = _percentile_to_grade(pos_ov_pct)
        print(f"  Position grade (overall): {pos_grade} ({pos_ov_pct:.1f}%)")
        for cat, pct in pos_pcts.items():
            print(f"    {cat}: {pct}%")

        # Role grade (overall)
        role_profiles = POSITION_ROLE_PROFILES.get(pos, {})
        role_metrics = role_profiles.get(role)
        if role_metrics:
            role_cats = {role: role_metrics}
            role_pcts = _compute_percentiles(dict(row), all_peers, role_cats)
            role_pct = list(role_pcts.values())[0]
            role_grade = _percentile_to_grade(role_pct)
            print(f"  Role grade (overall): {role_grade} ({role_pct:.1f}%)")
            # Show individual metric percentiles for role
            avail = [m for m in role_metrics if m in all_peers.columns]
            print(f"  Role metrics ({len(avail)} avail / {len(role_metrics)} total):")
            for m in avail:
                val = row.get(m, 0)
                val = 0 if pd.isna(val) else (val or 0)
                peer_vals = all_peers[m].fillna(0)
                pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                print(f"    {m}: val={val:.2f}  pct={pct:.1f}%")
        else:
            print(f"  No role metrics for {role}")

# Also re-check Bayindir to confirm GK fix still holds
print("\n" + "=" * 70)
print("GK CHECK: Bayindir vs Lammens (Per 90)")
for gk_name in ["Bayindir", "Lammens"]:
    matches = df_p90[df_p90["nombre"].str.contains(gk_name, case=False, na=False)]
    if matches.empty:
        print(f"  {gk_name} NOT FOUND")
        continue
    row = matches.iloc[0]
    pos = row["posicion"]
    apps = row.get("Appearances", 0)
    peers = df_p90[(df_p90["posicion"] == pos) & (df_p90["Appearances"] >= _MIN_APPS_P90)]
    role = _classify_percentile_role(dict(row), df_p90, pos)
    role_profiles = POSITION_ROLE_PROFILES.get(pos, {})
    role_metrics = role_profiles.get(role)
    if role_metrics:
        role_cats = {role: role_metrics}
        role_pcts = _compute_percentiles(dict(row), peers, role_cats)
        role_pct = list(role_pcts.values())[0]
        role_grade = _percentile_to_grade(role_pct)
        print(f"  {gk_name}: role={role} apps={apps} grade={role_grade} ({role_pct:.1f}%)")
