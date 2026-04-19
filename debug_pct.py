from app import load_data
d = load_data()
p90 = d['per90']
ret_col = 'Retention %'
ohp_col = 'Own Half Pass %'
print(f"per90 has '{ret_col}': {ret_col in p90.columns}")
print(f"per90 has '{ohp_col}': {ohp_col in p90.columns}")
pct_cols = [c for c in p90.columns if '%' in c]
print(f"All % cols in per90: {sorted(pct_cols)}")
s = p90[p90['nombre'].str.contains('Stiller', case=False, na=False)]
if not s.empty:
    for c in pct_cols:
        print(f"  {c}: {s.iloc[0][c]}")
