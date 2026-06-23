import json

with open('replay_loss.json') as f: r = json.load(f)

# Summarize the match
my_prizes = 0
their_prizes = 0
for step_idx, step in enumerate(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            
            # Print attacks to see what's happening
            if log.get('type') == 15:
                print(f"Turn {step_idx}: P{log.get('playerIndex')} used Attack {log.get('attackId')} (Card {log.get('cardId')})")
            elif log.get('type') == 21: # whatever KO is
                pass
