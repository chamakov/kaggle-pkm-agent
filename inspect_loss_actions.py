import json

with open('replay_loss.json') as f: r = json.load(f)

for step_idx, step in enumerate(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            
            if log.get('playerIndex') == 0:
                t = log.get('type')
                if t in [3, 9, 10, 12, 13, 15]:
                    names = {3:'PLAY_ITEM', 9:'ATTACH_ENERGY', 10:'EVOLVE', 12:'END_TURN', 13:'RETREAT', 15:'ATTACK'}
                    print(f"Turn {step_idx}: {names[t]} - Card {log.get('cardId')}")
