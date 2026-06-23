import json
import subprocess
from collections import Counter

losses = 0
for i in range(5):
    subprocess.run(['python', '/app/run_simulation.py'], capture_output=True)
    with open('replay.json') as f: r = json.load(f)
    reward = r['steps'][-1][0].get('reward', -1)
    if reward == -1:
        losses += 1
        retreats = {'us': 0, 'them': 0}
        for step in r['steps']:
            for p in step:
                if not isinstance(p, dict) or 'observation' not in p: continue
                logs = p['observation'].get('logs', [])
                for log in logs:
                    if not isinstance(log, dict): continue
                    if log.get('type') == 6:
                        if log.get('fromArea') == 4 and log.get('toArea') == 5:
                            if log.get('playerIndex') == 0: retreats['us'] += 1
                            else: retreats['them'] += 1
        
        print(f'LOSS! Match Length: {len(r["steps"])}')
        print(f'Retreats: Us: {retreats["us"]}, Them: {retreats["them"]}')
