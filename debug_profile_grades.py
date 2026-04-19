"""Exactly reproduce Player Profile grade computation for Stiller."""
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _KPI_INVERTED_CATS,
                 _compute_attribute_grades, _classify_role, _select_df,
                 ATTRIBUTE_GRADE_CATEGORIES, _INVERTED_GRADE_CATS)

data = load_data()
df_total = data["total"]

# Find Stiller
row = df_total[df_total["nombre"].str.contains("Stiller", case=False, na=False)].iloc[0]
position = row["posicion"]
pos_detail = row["posicion_detail"]
league = row["league_display"]
role = _classify_role(row, position, df_total)

print(f"Player: {row['nombre']}")
print(f"Position: {pos_detail} ({position})")
print(f"Role: {role}")
print(f"League: {league}")
print(f"KPI profile exists: {'Deep-Lying Playmaker' in _ROLE_KPI_PROFILES}")
print(f"_KPI_INVERTED_CATS: {_KPI_INVERTED_CATS}")
print()

# Simulate Per 90 mode
stat_mode = "Per 90"
_active_df = _select_df(data, stat_mode)
print(f"Active DF shape: {_active_df.shape}")
print(f"'Retention %' in active_df: {'Retention %' in _active_df.columns}")
print(f"'Own Half Pass %' in active_df: {'Own Half Pass %' in _active_df.columns}")

# Filter like the app does
_MIN_90S_P90 = 5
_is_sel = _active_df["nombre"] == row["nombre"]
_has_mins = _active_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
p90_filtered = _active_df[_is_sel | _has_mins]
p90_match = p90_filtered[p90_filtered["nombre"] == row["nombre"]]
row_data = dict(p90_match.iloc[0]) if not p90_match.empty else dict(row)
grade_df = p90_filtered

print(f"grade_df shape: {grade_df.shape}")
print(f"'Retention %' in grade_df: {'Retention %' in grade_df.columns}")
print(f"row_data Retention %: {row_data.get('Retention %', 'MISSING')}")
print(f"row_data Own Half Pass %: {row_data.get('Own Half Pass %', 'MISSING')}")
print(f"row_data Pass %: {row_data.get('Pass %', 'MISSING')}")
print()

# Test both scope modes
for scope_mode in ["Across Europe", "League"]:
    for basis_mode in ["Position", "Role"]:
        _scope_league = league if scope_mode == "League" else None
        _basis_role = role if basis_mode == "Role" else None
        
        attr_grades = _compute_attribute_grades(
            row_data, position, grade_df,
            league=_scope_league, role=_basis_role,
            df_role_ref=df_total, kpi_role=role
        )
        
        print(f"--- Scope={scope_mode}, Basis={basis_mode} ---")
        for attr, (grade, pct) in attr_grades.items():
            print(f"  {attr}: {grade} ({pct})")
        print()
