import json

with open('replay.json') as f: r = json.load(f)

for step_idx, step in enumerate(r['steps']):
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        logs = p['observation'].get('logs', [])
        for log in logs:
            if not isinstance(log, dict): continue
            if log.get('type') == 15 and log.get('playerIndex') == 0:
                print(f"Attack ID {log.get('attackId')} used.")
                current = p['observation'].get('current', {})
                if current and current.get('players'):
                    opp = current['players'][1]
                    active_list = opp.get('active', [])
                    if active_list and active_list[0]:
                        active = active_list[0]
                        hp_before = active.get('hp')
                        dmg_before = active.get('damage', 0)
                        if dmg_before is None: dmg_before = 0
                        
                        # Find damage after
                        dmg_after = None
                        if step_idx + 1 < len(r['steps']):
                            for np in r['steps'][step_idx + 1]:
                                if not isinstance(np, dict) or 'observation' not in np: continue
                                ncurrent = np['observation'].get('current', {})
                                if ncurrent and ncurrent.get('players'):
                                    nopp = ncurrent['players'][1]
                                    nactive_list = nopp.get('active', [])
                                    if nactive_list and nactive_list[0]:
                                        nactive = nactive_list[0]
                                        if nactive.get('id') == active.get('id'): # same pokemon
                                            dmg_after = nactive.get('damage', 0)
                                            if dmg_after is None: dmg_after = 0
                                    break
                        if dmg_after is not None:
                            print(f"  -> Damage dealt: {dmg_after - dmg_before}")
                        else:
                            print(f"  -> Opponent Active disappeared (likely KO'd!)")
