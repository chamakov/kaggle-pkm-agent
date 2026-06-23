import json

with open('replay_tank_0.json') as f: r = json.load(f)

for step in r['steps']:
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        actions = p['observation'].get('actions', [])
        # CABT sometimes provides actions differently (e.g., select options)
        select = p['observation'].get('select')
        options = []
        if select:
            options = select.get('option', [])
        else:
            options = actions
            
        for opt in options:
            if opt.get('type') == 15:
                print(f"Option available: Attack {opt.get('attackId')}")
