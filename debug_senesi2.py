"""Check Senesi's Lab-visible grades (generic ATTRIBUTE_GRADE_CATEGORIES) in Per 90."""
from app import load_data, _compute_attribute_grades, ATTRIBUTE_GRADE_CATEGORIES, _INVERTED_GRADE_CATS
import pandas as pd

data = load_data()
df_per90 = data["per90"]

match = df_per90[df_per90["nombre"].str.contains("Senesi", case=False, na=False)]
if match.empty:
    print("Senesi not found in Per 90")
else:
    row = match.iloc[0]
    name = row["nombre"]
    pos = row["posicion"]
    league = row["league_display"]
    print(f"{name} — {pos}, {league}")
    print(f"Minutes: {row.get('Time Played')}, Est 90s: {row.get('estimated_90s')}")

    # Generic attribute grades (what the Lab shows)
    grades_ov = _compute_attribute_grades(dict(row), pos, df_per90, league=None, kpi_role=None)
    grades_lg = _compute_attribute_grades(dict(row), pos, df_per90, league=league, kpi_role=None)

    print(f"\nGeneric grades (Europe-wide):")
    for attr, (grade, pct) in grades_ov.items():
        inv = " [INVERTED]" if attr in _INVERTED_GRADE_CATS else ""
        print(f"  {attr}: {grade} ({pct}){inv}")

    print(f"\nGeneric grades (League={league}):")
    for attr, (grade, pct) in grades_lg.items():
        inv = " [INVERTED]" if attr in _INVERTED_GRADE_CATS else ""
        print(f"  {attr}: {grade} ({pct}){inv}")

    # Raw stats for Ball Progression
    print(f"\nBall Progression raw stats (Per 90):")
    for m in ATTRIBUTE_GRADE_CATEGORIES["Ball Progression"]:
        val = row.get(m, "N/A")
        peers = df_per90[df_per90["posicion"] == pos]
        if m in peers.columns and pd.notna(val):
            pct = (peers[m].fillna(0) < val).sum() / max(len(peers), 1) * 100
            print(f"  {m}: {val:.3f} (pct={pct:.1f})")
        else:
            print(f"  {m}: {val}")

    # Raw stats for Passing Safety (inverted)
    print(f"\nPassing Safety raw stats (Per 90) [LOWER is better]:")
    for m in ATTRIBUTE_GRADE_CATEGORIES["Passing Safety"]:
        val = row.get(m, "N/A")
        peers = df_per90[df_per90["posicion"] == pos]
        if m in peers.columns and pd.notna(val):
            pct = (peers[m].fillna(0) < val).sum() / max(len(peers), 1) * 100
            print(f"  {m}: {val:.3f} (raw pct={pct:.1f}, inverted={100-pct:.1f})")
        else:
            print(f"  {m}: {val}")
