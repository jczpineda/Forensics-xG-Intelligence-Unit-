import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_role, POSITION_ROLE_PROFILES)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

row = df90[df90["nombre"].str.contains("Vinícius Júnior", case=False, na=False)].iloc[0]
name = row["nombre"]
orig_pos = row["posicion"]
print(f"Player: {name}, Original Pos: {orig_pos}")

# Check Striker now has winger roles
st_roles = list(POSITION_ROLE_PROFILES.get("Striker", {}).keys())
print(f"\nStriker roles: {st_roles}")

# Auto-classify as Striker
role_as_striker = _classify_role(row, "Striker", dft)
print(f"Auto-classified Striker role: {role_as_striker}")

# Check KPI profile
kpi = _ROLE_KPI_PROFILES.get(role_as_striker)
print(f"KPI profile for '{role_as_striker}': {kpi is not None}")
if kpi:
    for cat, (w, metrics) in kpi.items():
        print(f"  {cat} (w={w}): {metrics}")

# Grades as Striker with auto role
print(f"\n--- Grades as Striker / {role_as_striker} ---")
grades = _compute_attribute_grades(dict(row), "Striker", df90, kpi_role=role_as_striker)
for cat, (letter, num) in grades.items():
    print(f"  {cat}: {letter} ({num})")

# Now test override to AMW
print(f"\n--- Override to AMW ---")
role_as_amw = _classify_role(row, "Attacking Midfielder/Winger", dft)
print(f"Auto-classified AMW role: {role_as_amw}")
kpi_amw = _ROLE_KPI_PROFILES.get(role_as_amw)
print(f"KPI profile for '{role_as_amw}': {kpi_amw is not None}")

grades_amw = _compute_attribute_grades(dict(row), "Attacking Midfielder/Winger", df90, kpi_role=role_as_amw)
for cat, (letter, num) in grades_amw.items():
    print(f"  {cat}: {letter} ({num})")

# Test each winger role explicitly
print("\n--- All winger roles as Striker ---")
for r in ["Inside Forward", "Classic Winger", "Creative Winger", "Pressing Winger"]:
    kpi_r = _ROLE_KPI_PROFILES.get(r)
    if kpi_r:
        g = _compute_attribute_grades(dict(row), "Striker", df90, kpi_role=r)
        cats_str = ", ".join(f"{c}: {l} ({n})" for c, (l, n) in g.items())
        print(f"  {r}: {cats_str}")
    else:
        print(f"  {r}: no KPI profile")
