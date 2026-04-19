import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _compute_attribute_grades,
                 _classify_role, POSITION_ROLE_PROFILES)

data = load_data()
df90 = data["per90"]
dft  = data["total"]

# Find Shaw
shaw = df90[df90["nombre"].str.contains("Shaw", case=False, na=False)]
print("=== Shaw matches ===")
print(shaw[["nombre", "equipo", "posicion", "posicion_detail", "liga"]].to_string())
print()

row = shaw[shaw["nombre"].str.contains("L. Shaw", case=False, na=False)].iloc[0]
orig_pos = row["posicion"]
name = row["nombre"]
print(f"Player: {name}, Original Pos: {orig_pos}")

# Original role
orig_role = _classify_role(row, orig_pos, dft)
print(f"Original Role: {orig_role}")

# Override to Full-Back
new_pos = "Full-Back"
new_role = _classify_role(row, new_pos, dft)
print(f"Overridden Pos: {new_pos}, New Role: {new_role}")

# Show available roles for Full-Back
fb_roles = list(POSITION_ROLE_PROFILES.get(new_pos, {}).keys())
print(f"Available Full-Back roles: {fb_roles}")

# Grades with original position (CB)
print(f"\n--- Grades as {orig_pos} ({orig_role}) ---")
grades_orig = _compute_attribute_grades(dict(row), orig_pos, df90, kpi_role=orig_role)
for cat, (letter, num) in grades_orig.items():
    print(f"  {cat}: {letter} ({num})")

# Grades with overridden position (Full-Back)
print(f"\n--- Grades as {new_pos} ({new_role}) ---")
grades_new = _compute_attribute_grades(dict(row), new_pos, df90, kpi_role=new_role)
for cat, (letter, num) in grades_new.items():
    print(f"  {cat}: {letter} ({num})")
