import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_role, POSITION_ROLE_PROFILES)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

# Find Vinicius Jr
vini = df90[df90["nombre"].str.contains("Vinícius|Vinicius", case=False, na=False)]
print("=== Vinicius matches ===")
print(vini[["nombre", "equipo", "posicion", "posicion_detail", "liga"]].to_string())
print()

row = vini.iloc[0]
name = row["nombre"]
orig_pos = row["posicion"]
print(f"Player: {name}, Original Pos: {orig_pos}")

# Override to AMW
new_pos = "Attacking Midfielder/Winger"
new_role = _classify_role(row, new_pos, dft)
print(f"Auto-classified AMW role: {new_role}")
print(f"Available AMW roles: {list(POSITION_ROLE_PROFILES.get(new_pos, {}).keys())}")

# Check peer pool size
amw_peers = df90[df90["posicion"] == new_pos]
print(f"\nAMW peer pool in per90: {len(amw_peers)}")

# Check KPI profile
kpi = _ROLE_KPI_PROFILES.get(new_role)
print(f"KPI profile exists for '{new_role}': {kpi is not None}")
if kpi:
    for cat, (w, metrics) in kpi.items():
        print(f"  {cat}: {metrics}")

# Try computing grades
print("\n--- Grades as AMW ---")
try:
    grades = _compute_attribute_grades(dict(row), new_pos, df90, kpi_role=new_role)
    for cat, (letter, num) in grades.items():
        print(f"  {cat}: {letter} ({num})")
except Exception as e:
    print(f"  ERROR: {e}")

# Try with league filter
league = row.get("league_display", "")
print(f"\n--- Grades as AMW (League: {league}) ---")
try:
    grades_lg = _compute_attribute_grades(dict(row), new_pos, df90, league=league, kpi_role=new_role)
    for cat, (letter, num) in grades_lg.items():
        print(f"  {cat}: {letter} ({num})")
except Exception as e:
    print(f"  ERROR: {e}")

# Check if maybe the issue is estimated_90s filtering
print(f"\n--- Per90 filtering check ---")
min90 = 5
has_mins = df90["estimated_90s"].fillna(0) >= min90
p90_filtered = df90[df90["nombre"] == name] | df90[has_mins]
print(f"Vini estimated_90s: {row.get('estimated_90s', 'N/A')}")
is_sel = df90["nombre"] == name
p90f = df90[is_sel | has_mins]
amw_in_p90f = p90f[p90f["posicion"] == new_pos]
print(f"AMW peers in p90_filtered: {len(amw_in_p90f)}")
