import json

with open('replay.json') as f: r = json.load(f)

for step_idx, step in enumerate(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            if log.get('type') == 15 and log.get('playerIndex') == 0:
                print(f"--- Turn {step_idx} ---")
                print(f"We attacked with ID {log.get('attackId')}")
                
                # Check opponent damage BEFORE
                current = p['observation'].get('current', {})
                if current and current.get('players'):
                    opp = current['players'][1]
                    active_list = opp.get('active', [])
                    if active_list and active_list[0]:
                        active = active_list[0]
                        print(f"Opponent Active HP: {active.get('hp')}, Damage BEFORE: {active.get('damage')}")
                
                # Check opponent damage AFTER (next step)
                if step_idx + 1 < len(r['steps']):
                    next_step = r['steps'][step_idx + 1]
                    for np in next_step:
                        if not isinstance(np, dict) or 'observation' not in np: continue
                        ncurrent = np['observation'].get('current', {})
                        if ncurrent and ncurrent.get('players'):
                            nopp = ncurrent['players'][1]
                            nactive_list = nopp.get('active', [])
                            if nactive_list and nactive_list[0]:
                                nactive = nactive_list[0]
                                print(f"Opponent Active Damage AFTER: {nactive.get('damage')}")
                            else:
                                print(f"Opponent Active AFTER: NONE (Killed!)")
                            break
                print("")
                break
