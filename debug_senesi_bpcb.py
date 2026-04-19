"""Check Senesi as Ball-Playing CB in both Total and Per 90."""
from app import load_data, _compute_attribute_grades, _ROLE_KPI_PROFILES

data = load_data()

for mode, label in [("total", "Total"), ("per90", "Per 90")]:
    df = data[mode]
    match = df[df["nombre"].str.contains("Senesi", case=False, na=False)]
    if match.empty:
        continue
    row = match.iloc[0]
    pos = "Centre-Back"
    role = "Ball-Playing CB"

    grades = _compute_attribute_grades(dict(row), pos, df, league=None, kpi_role=role)
    grades_lg = _compute_attribute_grades(dict(row), pos, df, league=row["league_display"], kpi_role=role)

    print(f"=== {row['nombre']} as {role} ({label}) ===")
    print(f"  Europe-wide:")
    for attr, (g, p) in grades.items():
        print(f"    {attr}: {g} ({p})")
    print(f"  League ({row['league_display']}):")
    for attr, (g, p) in grades_lg.items():
        print(f"    {attr}: {g} ({p})")

    # Show raw KPI metrics
    kpi = _ROLE_KPI_PROFILES[role]
    print(f"  Raw stats:")
    import pandas as pd
    peers = df[df["posicion"] == pos]
    for cat, (w, metrics) in kpi.items():
        print(f"    {cat} (w={w}):")
        for m in metrics:
            val = row.get(m, "N/A")
            if m in peers.columns and pd.notna(val):
                pct = (peers[m].fillna(0) < val).sum() / max(len(peers), 1) * 100
                print(f"      {m}: {val:.3f} (pct={pct:.1f})" if isinstance(val, float) else f"      {m}: {val} (pct={pct:.1f})")
    print()
