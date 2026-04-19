import pandas as pd, os

BASE = r'c:\Users\kenzi\Downloads\AnalisisDeJuego_Europea_Marzo26'
EVENTS = os.path.join(BASE, 'Events')

shots = pd.read_excel(os.path.join(EVENTS, 'SB_Euro24_Events_shots.xlsx'))
passes = pd.read_excel(os.path.join(EVENTS, 'SB_Euro24_Events_passes.xlsx'))
matches = pd.read_excel(os.path.join(BASE, 'SB_Euro24_matchdetails.xlsx'))

eng_shots = shots[shots['team'] == 'England']
eng_passes = passes[passes['team'] == 'England']
key_passes = eng_passes[eng_passes['pass_assisted_shot_id'].notna()]

def match_name(mid):
    row = matches[matches['match_id'] == mid]
    if len(row):
        return '{} vs {}'.format(row.iloc[0]['home_team'], row.iloc[0]['away_team'])
    return str(mid)

# --- WITH shootout penalties ---
print('=' * 80)
print('INCLUDING penalty shootout shots')
print('=' * 80)
print('{:<30} {:>6} {:>8} {:>10} {:>8}'.format('Match', 'Shots', 'xG', 'Key Pass', 'Goals'))
print('-' * 70)
for mid in sorted(eng_shots['match_id'].unique()):
    ms = eng_shots[eng_shots['match_id'] == mid]
    mk = key_passes[key_passes['match_id'] == mid]
    goals = (ms['shot_outcome'] == 'Goal').sum()
    pens = (ms['shot_type'] == 'Penalty').sum()
    print('{:<30} {:>6} {:>8.2f} {:>10} {:>5} ({} pens)'.format(
        match_name(mid), len(ms), ms['shot_statsbomb_xg'].sum(), len(mk), goals, pens))

total = len(eng_shots)
total_xg = eng_shots['shot_statsbomb_xg'].sum()
total_kp = len(key_passes)
total_goals = (eng_shots['shot_outcome'] == 'Goal').sum()
nm = len(eng_shots['match_id'].unique())
print('-' * 70)
print('{:<30} {:>6} {:>8.2f} {:>10} {:>8}'.format('TOTAL', total, total_xg, total_kp, total_goals))
print('{:<30} {:>6.1f} {:>8.2f} {:>10.1f} {:>8.1f}'.format('PER MATCH (n={})'.format(nm), total/nm, total_xg/nm, total_kp/nm, total_goals/nm))

# --- WITHOUT shootout penalties (exclude penalties in extra time / shootout) ---
# Shootout = penalties after minute 120 in match 3942227
shootout_mask = (eng_shots['match_id'] == 3942227) & (eng_shots['shot_type'] == 'Penalty') & (eng_shots['minute'] >= 120)
eng_shots_clean = eng_shots[~shootout_mask]

print()
print('=' * 80)
print('EXCLUDING penalty shootout shots (5 shootout pens vs Switzerland removed)')
print('=' * 80)
print('{:<30} {:>6} {:>8} {:>10} {:>8}'.format('Match', 'Shots', 'xG', 'Key Pass', 'Goals'))
print('-' * 70)
for mid in sorted(eng_shots_clean['match_id'].unique()):
    ms = eng_shots_clean[eng_shots_clean['match_id'] == mid]
    mk = key_passes[key_passes['match_id'] == mid]
    goals = (ms['shot_outcome'] == 'Goal').sum()
    pens = (ms['shot_type'] == 'Penalty').sum()
    print('{:<30} {:>6} {:>8.2f} {:>10} {:>5} ({} pens)'.format(
        match_name(mid), len(ms), ms['shot_statsbomb_xg'].sum(), len(mk), goals, pens))

total2 = len(eng_shots_clean)
total_xg2 = eng_shots_clean['shot_statsbomb_xg'].sum()
total_goals2 = (eng_shots_clean['shot_outcome'] == 'Goal').sum()
print('-' * 70)
print('{:<30} {:>6} {:>8.2f} {:>10} {:>8}'.format('TOTAL', total2, total_xg2, total_kp, total_goals2))
print('{:<30} {:>6.1f} {:>8.2f} {:>10.1f} {:>8.1f}'.format('PER MATCH (n={})'.format(nm), total2/nm, total_xg2/nm, total_kp/nm, total_goals2/nm))

# --- xG per shot quality comparison ---
print()
print('=' * 80)
print('xG PER SHOT (chance quality indicator)')
print('=' * 80)
xg_per_shot_all = total_xg / total
xg_per_shot_clean = total_xg2 / total2
# Exclude ALL penalties for open-play quality
open_play = eng_shots[eng_shots['shot_type'] != 'Penalty']
xg_per_shot_open = open_play['shot_statsbomb_xg'].sum() / len(open_play) if len(open_play) else 0
print('Including everything:      {:.3f} xG/shot  ({} shots, {:.2f} total xG)'.format(xg_per_shot_all, total, total_xg))
print('Excluding shootout pens:   {:.3f} xG/shot  ({} shots, {:.2f} total xG)'.format(xg_per_shot_clean, total2, total_xg2))
print('Open play only (no pens):  {:.3f} xG/shot  ({} shots, {:.2f} total xG)'.format(xg_per_shot_open, len(open_play), open_play['shot_statsbomb_xg'].sum()))
