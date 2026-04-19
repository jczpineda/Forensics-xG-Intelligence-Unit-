"""Debug Onana (Anchor Man) and Lammens grades in Player Profile."""
import pandas as pd, numpy as np
from app import (load_data, _ROLE_KPI_PROFILES, _KPI_INVERTED_CATS,
                 _compute_attribute_grades, _classify_role, _select_df)

data = load_data()
df_total = data["total"]

for name in ["Onana", "Lammens"]:
    matches = df_total[df_total["nombre"].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"\n{'='*60}\n{name}: NOT FOUND\n")
        continue
    for _, row in matches.iterrows():
        position = row["posicion"]
        pos_detail = row["posicion_detail"]
        league = row["league_display"]
        role = _classify_role(row, position, df_total)
        
        print(f"\n{'='*60}")
        print(f"Player: {row['nombre']}, Team: {row['equipo']}")
        print(f"Position: {pos_detail} ({position}), Role: {role}, League: {league}")
        
        # Per 90 mode
        _active_df = _select_df(data, "Per 90")
        _MIN_90S_P90 = 5
        _is_sel = _active_df["nombre"] == row["nombre"]
        _has_mins = _active_df["estimated_90s"].fillna(0) >= _MIN_90S_P90
        p90_filtered = _active_df[_is_sel | _has_mins]
        p90_match = p90_filtered[p90_filtered["nombre"] == row["nombre"]]
        row_data = dict(p90_match.iloc[0]) if not p90_match.empty else dict(row)
        grade_df = p90_filtered
        
        print(f"Per90 pool size: {len(grade_df)}")
        print(f"Position pool: {len(grade_df[grade_df['posicion'] == position])}")
        
        # Get the KPI profile
        profile = _ROLE_KPI_PROFILES.get(role, {})
        if profile:
            print(f"\nKPI Profile ({role}):")
            for cat_name, (weight, metrics) in profile.items():
                avail = [m for m in metrics if m in grade_df.columns]
                missing = [m for m in metrics if m not in grade_df.columns]
                print(f"  {cat_name} (w={weight}): {len(avail)}/{len(metrics)} metrics")
                if missing:
                    print(f"    MISSING: {missing}")
                # Show player's values + percentiles
                peers = grade_df[grade_df["posicion"] == position]
                for m in avail:
                    val = row_data.get(m, 0)
                    val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else val
                    peer_vals = peers[m].fillna(0)
                    pct = (peer_vals < val).sum() / max(len(peer_vals), 1) * 100
                    print(f"    {m}: val={val:.2f}, pctile={pct:.1f}")
        
        # Compute grades
        for scope_mode in ["Across Europe", "League"]:
            _scope_league = league if scope_mode == "League" else None
            attr_grades = _compute_attribute_grades(
                row_data, position, grade_df,
                league=_scope_league, role=None,
                df_role_ref=df_total, kpi_role=role
            )
            print(f"\n  --- Per 90, Scope={scope_mode} ---")
            for attr, (grade, pct) in attr_grades.items():
                print(f"    {attr}: {grade} ({pct})")
