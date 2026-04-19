import pandas as pd, os, ast
BASE = r'c:\Users\kenzi\Downloads\AnalisisDeJuego_Europea_Marzo26'
EVENTS = os.path.join(BASE, 'Events')

s = pd.read_excel(os.path.join(EVENTS, 'SB_Euro24_Events_shots.xlsx'))
eng = s[s['team'] == 'England'].copy()

# Check shot_end_location structure
print('=== Sample shot_end_location values ===')
for i, row in eng.head(10).iterrows():
    print(f"  {row['player'][:20]:<20} min={row['minute']:>3}  end_loc={row['shot_end_location']}  outcome={row['shot_outcome']}")

print()

# Parse end locations
def parse_loc(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    if isinstance(val, (list, tuple)):
        return val
    return None

eng['end_parsed'] = eng['shot_end_location'].apply(parse_loc)
eng['end_y'] = eng['end_parsed'].apply(lambda x: x[1] if x and len(x) >= 2 else None)
eng['end_z'] = eng['end_parsed'].apply(lambda x: x[2] if x and len(x) >= 3 else None)
eng['start_y'] = eng['location'].apply(lambda x: parse_loc(x)[1] if parse_loc(x) else None)
eng['start_x'] = eng['location'].apply(lambda x: parse_loc(x)[0] if parse_loc(x) else None)

# StatsBomb pitch: goal posts at y=36 and y=44 (center of goal at y=40)
# x=120 is the goal line
print('=== Goal post reference: y=36 (right post), y=44 (left post), center=40 ===')
print()

# Classify shot direction
def classify_direction(end_y):
    if end_y is None:
        return 'Unknown'
    if end_y < 36:
        return 'Wide Right (missed)'
    elif end_y < 38:
        return 'Near Post (Right)'
    elif end_y <= 42:
        return 'Centre'
    elif end_y <= 44:
        return 'Far Post (Left)'
    else:
        return 'Wide Left (missed)'

eng['shot_direction'] = eng['end_y'].apply(classify_direction)

print('=== Shot Direction Distribution ===')
print(eng['shot_direction'].value_counts().to_string())
print()

print('=== Shot Direction by Outcome ===')
ct = pd.crosstab(eng['shot_direction'], eng['shot_outcome'])
print(ct.to_string())
print()

print('=== Detailed shots with direction ===')
cols = ['player', 'minute', 'shot_outcome', 'shot_direction', 'start_x', 'start_y', 'end_y', 'end_z', 'shot_statsbomb_xg']
print(eng[cols].to_string())
print()

# Check available body part info
print('=== shot_body_part values ===')
print(eng['shot_body_part'].value_counts().to_string())
print()
print('=== shot_technique values ===')
print(eng['shot_technique'].value_counts().to_string())
