import json

with open('replay_tank_0.json') as f: r = json.load(f)

for step_idx, step in enumerate(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            
            # KO log?
            if log.get('type') == 21: # type 21 might be KO or prize taken
                print(f"Log type 21: {log}")
            
            if log.get('type') == 15 and log.get('playerIndex') == 0:
                # Attack!
                print(f"Turn {step_idx}: We attacked with ID {log.get('attackId')} (Card {log.get('cardId')})")
