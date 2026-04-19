"""Debug Stiller's Ball Security & Distribution grades."""
import pandas as pd
import numpy as np
from app import load_data, _ROLE_KPI_PROFILES, _KPI_INVERTED_CATS

data = load_data()

for mode_name in ["per90"]:
    df = data[mode_name]
    stiller = df[df["nombre"].str.contains("Stiller", case=False, na=False)]
    if stiller.empty:
        print(f"[{mode_name}] Stiller not found"); continue
    s = stiller.iloc[0]
    print(f"\n{'='*60}")
    print(f"MODE: {mode_name}")
    print(f"Player: {s['nombre']}, Team: {s['equipo']}, Pos: {s['posicion']}")
    
    # Check DLP profile
    profile = _ROLE_KPI_PROFILES.get("Deep-Lying Playmaker", {})
    print(f"\nKPI_INVERTED_CATS = {_KPI_INVERTED_CATS}")
    print(f"\nDLP Profile:")
    
    # Get position pool for percentile
    pos = s["posicion"]
    pos_df = df[df["posicion"] == pos].copy()
    print(f"Position pool ({pos}): {len(pos_df)} players")
    
    for cat_name, (weight, metrics) in profile.items():
        print(f"\n  {cat_name} (w={weight}):")
        pcts = []
        for m in metrics:
            if m not in df.columns:
                print(f"    {m}: COLUMN MISSING!")
                continue
            val = s.get(m, np.nan)
            vals = pd.to_numeric(pos_df[m], errors="coerce").dropna()
            if len(vals) == 0:
                print(f"    {m}: no data in pool")
                continue
            pct = (vals < val).sum() / len(vals) * 100
            # Check if inverted
            inverted = cat_name in _KPI_INVERTED_CATS
            if inverted:
                pct = 100 - pct
            pcts.append(pct)
            print(f"    {m}: val={val:.2f}, pctile={pct:.1f}{'(inv)' if inverted else ''}")
        if pcts:
            avg = np.mean(pcts)
            # Grade
            if avg >= 90: grade = "A+"
            elif avg >= 80: grade = "A"
            elif avg >= 70: grade = "B+"
            elif avg >= 60: grade = "B"
            elif avg >= 50: grade = "C+"
            elif avg >= 40: grade = "C"
            elif avg >= 30: grade = "D+"
            elif avg >= 20: grade = "D"
            elif avg >= 10: grade = "F+"
            else: grade = "F"
            print(f"    => AVG pctile: {avg:.1f} => Grade: {grade}")
