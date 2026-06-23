import json

with open('replay.json') as f: r = json.load(f)

for step in reversed(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        current = p['observation'].get('current')
        if current and current.get('players'):
            me = current['players'][0]
            them = current['players'][1]
            print(f'End state Us: deck={me.get("deckCount")}, prizes={me.get("prizeCount")}')
            print(f'End state Them: deck={them.get("deckCount")}, prizes={them.get("prizeCount")}')
            break
    break
