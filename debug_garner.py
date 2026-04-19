import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_position_roles, POSITION_ROLE_PROFILES)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

# ── Find Garner's role from total data ──
roles = _classify_position_roles(dft, "Central Midfield")
garner_total = dft[dft["nombre"] == "J. Garner"]
for gi in garner_total.index:
    print(f"Garner role (from total): {roles.get(gi, 'NOT FOUND')}")

garner_row = df90[df90["nombre"] == "J. Garner"].iloc[0]
role = roles.get(garner_total.index[0], "Box-to-Box")
print(f"\nRole assigned: {role}")
print(f"Position: {garner_row['posicion']}, League: {garner_row['liga']}")
print(f"Team: {garner_row['equipo']}")

prof = _ROLE_KPI_PROFILES.get(role, {})
print(f"\nKPI Categories: {list(prof.keys())}")

# ── Position peers ──
pos_mask = df90["posicion"] == garner_row["posicion"]
pos_df = df90[pos_mask]
print(f"Position peer pool: {len(pos_df)}")

# ── Per-metric values and percentiles ──
for cat, (w, metrics) in prof.items():
    found = sum(1 for m in metrics if m in df90.columns)
    print(f"\n  {cat} (w={w}): {found}/{len(metrics)} metrics")
    for m in metrics:
        val = garner_row.get(m, np.nan)
        vals = pd.to_numeric(pos_df[m], errors="coerce").dropna()
        pctile = (vals < val).sum() / len(vals) * 100 if len(vals) > 0 else 0
        print(f"    {m}: val={val:.2f}, pctile={pctile:.1f}")

# ── Compute grades ──
pos = garner_row["posicion"]
league_disp = garner_row.get("league_display", "")

# Across Europe (no league filter)
grades_ov = _compute_attribute_grades(garner_row, pos, df90, league=None, kpi_role=role)
print("\n  --- Per 90, Scope=Across Europe ---")
for cat, (letter, num) in grades_ov.items():
    print(f"    {cat}: {letter} ({num})")

# League only
grades_lg = _compute_attribute_grades(garner_row, pos, df90, league=league_disp, kpi_role=role)
print(f"\n  --- Per 90, Scope=League ({league_disp}) ---")
for cat, (letter, num) in grades_lg.items():
    print(f"    {cat}: {letter} ({num})")
