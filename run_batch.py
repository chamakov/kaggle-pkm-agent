import json
import random
from test_matchups import run_match, deck_aggro, deck_tank

def get_reward(fname):
    try:
        with open(fname) as f: r = json.load(f)
        return r['steps'][-1][0].get('reward', -1), len(r['steps'])
    except:
        return -1, 0

results = {'Aggro': {'wins': 0, 'turns': []}, 'Tanks': {'wins': 0, 'turns': []}}

for i in range(4):
    print(f"\n--- CYCLE {i+1} ---")
    
    run_match(f'Aggro {i}', deck_aggro, f'replay_aggro_{i}.json')
    r, t = get_reward(f'replay_aggro_{i}.json')
    if r == 1: results['Aggro']['wins'] += 1
    results['Aggro']['turns'].append(t)
    
    run_match(f'Tanks {i}', deck_tank, f'replay_tank_{i}.json')
    r, t = get_reward(f'replay_tank_{i}.json')
    if r == 1: results['Tanks']['wins'] += 1
    results['Tanks']['turns'].append(t)

print("\n=== FINAL RESULTS ===")
for opp, data in results.items():
    wins = data['wins']
    turns = data['turns']
    avg_turns = sum(turns) / len(turns) if turns else 0
    print(f"Vs {opp}: {wins}/4 Wins (Avg Turns: {avg_turns:.1f})")

