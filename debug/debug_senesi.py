"""Debug M. Senesi's grades — investigate Progression S vs low Ball Security/Distribution per 90."""
from app import (load_data, _compute_attribute_grades, _classify_role,
                 ATTRIBUTE_GRADE_CATEGORIES, _ROLE_KPI_PROFILES,
                 POSITION_ROLE_PROFILES, _classify_position_roles)
import pandas as pd

data = load_data()
df_total = data["total"]
df_per90 = data["per90"]

# Find Senesi
for df_label, df in [("Total", df_total), ("Per 90", df_per90)]:
    match = df[df["nombre"].str.contains("Senesi", case=False, na=False)]
    if not match.empty:
        row = match.iloc[0]
        name = row["nombre"]
        pos = row["posicion"]
        league = row["league_display"]
        team = row["equipo"]
        print(f"=== {name} ({df_label}) ===")
        print(f"  Position: {pos}, Team: {team}, League: {league}")
        print(f"  Minutes: {row.get('Time Played', 'N/A')}, Est 90s: {row.get('estimated_90s', 'N/A')}")

        # Classify role
        role = _classify_role(row, pos, df_total if df_label == "Total" else df_per90)
        print(f"  Role: {role}")

        # Attribute grades (Europe-wide, position peers)
        grades = _compute_attribute_grades(dict(row), pos, df, league=None, kpi_role=role)
        print(f"\n  Attribute grades (Europe-wide, role={role}):")
        for attr, (grade, pct) in grades.items():
            print(f"    {attr}: {grade} ({pct})")

        # Also show league-scoped
        grades_lg = _compute_attribute_grades(dict(row), pos, df, league=league, kpi_role=role)
        print(f"\n  Attribute grades (League={league}, role={role}):")
        for attr, (grade, pct) in grades_lg.items():
            print(f"    {attr}: {grade} ({pct})")

        # Show raw stats for Progression / Ball Security / Distribution KPIs
        kpi = _ROLE_KPI_PROFILES.get(role)
        if kpi:
            print(f"\n  KPI profile for {role}:")
            for cat_name, (weight, metrics) in kpi.items():
                print(f"    {cat_name} (w={weight}):")
                for m in metrics:
                    val = row.get(m, "N/A")
                    # Get peer percentile
                    peers = df[df["posicion"] == pos]
                    if m in peers.columns:
                        peer_vals = peers[m].fillna(0)
                        pval = val if pd.notna(val) else 0
                        pct = (peer_vals < pval).sum() / max(len(peer_vals), 1) * 100
                        print(f"      {m}: {val:.3f} (pct={pct:.1f})" if isinstance(val, float) else f"      {m}: {val} (pct={pct:.1f})")
                    else:
                        print(f"      {m}: {val}")
        else:
            # Generic categories
            print(f"\n  Generic ATTRIBUTE_GRADE_CATEGORIES:")
            cats = ATTRIBUTE_GRADE_CATEGORIES
            for cat_name, metrics in cats.items():
                print(f"    {cat_name}:")
                for m in metrics:
                    val = row.get(m, "N/A")
                    if isinstance(val, float) and pd.notna(val):
                        peers = df[df["posicion"] == pos]
                        if m in peers.columns:
                            peer_vals = peers[m].fillna(0)
                            pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                            print(f"      {m}: {val:.3f} (pct={pct:.1f})")
                        else:
                            print(f"      {m}: {val:.3f}")
                    else:
                        print(f"      {m}: {val}")
        print()
