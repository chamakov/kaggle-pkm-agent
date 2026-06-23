import json

with open('replay.json') as f: r = json.load(f)

attack_ids = set()
for step in r['steps']:
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            if log.get('type') == 15:
                cid = log.get('cardId')
                aid = log.get('attackId')
                attack_ids.add((cid, aid))

for cid, aid in sorted(attack_ids):
    print(f"Card {cid} used Attack {aid}")
