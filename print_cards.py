import json

with open('replay.json') as f: r = json.load(f)

for step in r['steps']:
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        current = p['observation'].get('current')
        if current and current.get('players'):
            for player in current['players']:
                zones = []
                if player.get('deck'): zones.extend(player['deck'])
                if player.get('hand'): zones.extend(player['hand'])
                if player.get('bench'): zones.extend(player['bench'])
                if player.get('active'): zones.extend(player['active'])
                
                for c in zones:
                    if c and c.get('id') == 331:
                        print(json.dumps(c, indent=2))
                        exit(0)
