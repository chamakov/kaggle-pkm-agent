import json
import subprocess
import os
from collections import Counter

# Run a baseline match to generate a fresh replay
subprocess.run(['python', '/app/run_simulation.py'])

try:
    with open('replay.json') as f: r = json.load(f)
    our_atks = Counter()
    opp_atks = Counter()
    retreats = {'us': 0, 'them': 0}
    
    for step in r['steps']:
        for p in step:
            if not isinstance(p, dict) or 'observation' not in p: continue
            logs = p['observation'].get('logs', [])
            for log in logs:
                if not isinstance(log, dict): continue
                pi = log.get('playerIndex')
                t = log.get('type')
                if t == 15:
                    if pi == 0: our_atks[log.get('attackId', -1)] += 1
                    else: opp_atks[log.get('attackId', -1)] += 1
                elif t == 6:
                    # Move card log, could be retreat or benching
                    from_a = log.get('fromArea')
                    to_a = log.get('toArea')
                    if from_a == 4 and to_a == 5:
                        if pi == 0: retreats['us'] += 1
                        else: retreats['them'] += 1
                        
    dmg_names = {89: 'Litten(20)', 205: 'Crabo(20)', 206: 'Chewtle(120)', 458: 'XerneasEX(200)', 571: 'Houndour(150)'}
    print(f'Match Length: {len(r["steps"])}')
    
    our_atk_names = {dmg_names.get(k, f"?{k}"): v for k, v in our_atks.most_common()}
    opp_atk_names = {dmg_names.get(k, f"?{k}"): v for k, v in opp_atks.most_common()}
    
    print(f'Our Attacks: {our_atk_names}')
    print(f'Opp Attacks: {opp_atk_names}')
    print(f'Retreats (Active->Bench): Us: {retreats["us"]}, Them: {retreats["them"]}')
    
    # Check end state
    last = r['steps'][-2]
    for p in last:
        if isinstance(p, dict) and 'observation' in p:
            current = p['observation'].get('current')
            if current and current.get('players'):
                me = current['players'][0]
                them = current['players'][1]
                print(f'End state Us: deck={me.get("deckCount")}, prizes={me.get("prizeCount")}')
                print(f'End state Them: deck={them.get("deckCount")}, prizes={them.get("prizeCount")}')
                break
except Exception as e:
    print('Error:', e)
