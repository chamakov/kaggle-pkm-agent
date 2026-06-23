import json
import subprocess
for i in range(3):
    subprocess.run(['python', '/app/run_simulation.py'], capture_output=True)
    with open('replay.json') as f: r = json.load(f)
    for step in r['steps']:
        for p in step:
            if not isinstance(p, dict) or 'observation' not in p: continue
            logs = p['observation'].get('logs', [])
            for log in logs:
                if not isinstance(log, dict): continue
                if log.get('type') == 15 and log.get('cardId') == 331:
                    print(f"XerneasEX (331) used attack: {log.get('attackId')}")
