import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_role, POSITION_ROLE_PROFILES)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

print("=== POSITION_ROLE_PROFILES keys ===")
for pos, profiles in POSITION_ROLE_PROFILES.items():
    roles = list(profiles.keys())
    print(f"  {pos}: {roles}")

# Test Vinicius Jr
row = df90[df90["nombre"].str.contains("Vinícius Júnior", case=False, na=False)].iloc[0]
name = row["nombre"]
print(f"\nPlayer: {name}, Opta Pos: {row['posicion']}")

# As Striker (default)
role_st = _classify_role(row, "Striker", dft)
print(f"\nStriker role: {role_st}")
grades_st = _compute_attribute_grades(dict(row), "Striker", df90, kpi_role=role_st)
for cat, (letter, num) in grades_st.items():
    print(f"  {cat}: {letter} ({num})")

# As Wingers (override)
role_w = _classify_role(row, "Wingers", dft)
print(f"\nWingers role: {role_w}")
grades_w = _compute_attribute_grades(dict(row), "Wingers", df90, kpi_role=role_w)
for cat, (letter, num) in grades_w.items():
    print(f"  {cat}: {letter} ({num})")

# Test Shaw as Full-Back still works
shaw = df90[df90["nombre"].str.contains("L. Shaw", case=False, na=False)].iloc[0]
role_fb = _classify_role(shaw, "Full-Back", dft)
print(f"\nShaw as Full-Back: role={role_fb}")
grades_shaw = _compute_attribute_grades(dict(shaw), "Full-Back", df90, kpi_role=role_fb)
for cat, (letter, num) in grades_shaw.items():
    print(f"  {cat}: {letter} ({num})")
