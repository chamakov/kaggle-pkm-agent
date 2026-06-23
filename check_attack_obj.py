import json

with open('replay_tank_0.json') as f: r = json.load(f)

for step in r['steps']:
    for p in step:
        if not isinstance(p, dict) or 'observation' not in p: continue
        
        # Check all nested dicts for "type": 15
        def find_type_15(obj):
            if isinstance(obj, dict):
                if obj.get('type') == 15 and 'attackId' in obj:
                    print(f"Found attack option: {obj}")
                    exit(0)
                for k, v in obj.items():
                    find_type_15(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_type_15(item)
                    
        find_type_15(p['observation'])
