import json

with open('replay.json') as f: r = json.load(f)

cards = set()
for step in r['steps']:
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        current = p['observation'].get('current')
        if current and current.get('players'):
            opp = current['players'][1]
            for zone in ['active', 'bench', 'discard']:
                for c in opp.get(zone, []):
                    if c and 'id' in c:
                        cards.add(c['id'])
print("Opponent cards seen:", sorted(list(cards)))
