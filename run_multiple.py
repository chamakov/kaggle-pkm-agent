import json
import random
from test_matchups import run_match, deck_tank

wins = 0
for i in range(3):
    run_match(f'Tanks {i}', deck_tank, f'replay_tank_{i}.json')
    with open(f'replay_tank_{i}.json') as f: r = json.load(f)
    reward = r['steps'][-1][0].get('reward', -1)
    if reward == 1: wins += 1

print(f'\nTanks winrate: {wins}/3')
