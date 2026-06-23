import json
import subprocess

attack_ids = set()
for i in range(10):
    subprocess.run(['python', '/app/run_simulation.py'], capture_output=True)
    with open('replay.json') as f: r = json.load(f)

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
