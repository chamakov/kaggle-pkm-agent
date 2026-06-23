import json
import subprocess

for i in range(5):
    subprocess.run(['python', '/app/run_simulation.py'], capture_output=True)
    with open('replay.json') as f: r = json.load(f)

    for step in r['steps']:
        for p in step:
            if not isinstance(p, dict) or 'observation' not in p: continue
            select = p['observation'].get('select')
            options = select.get('option', []) if select else p['observation'].get('actions', [])
            
            for opt in options:
                if opt.get('type') == 15 and opt.get('cardId') == 331:
                    print(f"XerneasEX can use Attack {opt.get('attackId')}")
