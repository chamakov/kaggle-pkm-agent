import json
import subprocess
wins = 0
turns = []
for i in range(4):
    print(f"--- BASELINE CYCLE {i+1} ---")
    subprocess.run(["python", "/app/run_simulation.py"], capture_output=True)
    try:
        with open("replay.json") as f: r = json.load(f)
        reward = r['steps'][-1][0].get('reward', -1)
        if reward == 1: wins += 1
        turns.append(len(r['steps']))
    except:
        pass
avg_turns = sum(turns) / len(turns) if turns else 0
print(f"\nVs Baseline: {wins}/4 Wins (Avg Turns: {avg_turns:.1f})")
