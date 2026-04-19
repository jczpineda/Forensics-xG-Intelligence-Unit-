import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_position_roles)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

row = df90[df90["nombre"].str.contains("Tonali", case=False, na=False)].iloc[0]
pos = row["posicion"]
league = row.get("league_display", "")
name = row["nombre"]
team = row["equipo"]
print(f"Player: {name}, Team: {team}, Pos: {pos}, League: {league}")

roles = _classify_position_roles(dft, pos)
ti = dft[dft["nombre"] == name].index
role = roles.get(ti[0], "Unknown") if len(ti) > 0 else "Unknown"
print(f"Role: {role}")

prof = _ROLE_KPI_PROFILES.get(role, {})
pos_df = df90[df90["posicion"] == pos]
print(f"Position peer pool: {len(pos_df)}")

for cat, (w, metrics) in prof.items():
    print(f"\n  {cat} (w={w}):")
    for m in metrics:
        val = row.get(m, np.nan)
        vals = pd.to_numeric(pos_df[m], errors="coerce").dropna()
        pctile = (vals < val).sum() / len(vals) * 100 if len(vals) > 0 else 0
        print(f"    {m}: val={val:.2f}, pctile={pctile:.1f}")

for scope_name, lg in [("Across Europe", None), ("League", league)]:
    grades = _compute_attribute_grades(row, pos, df90, league=lg, kpi_role=role)
    print(f"\n  --- {scope_name} ---")
    for cat, (letter, num) in grades.items():
        print(f"    {cat}: {letter} ({num})")
