import subprocess
import json
import shutil
for i in range(10):
    subprocess.run(['python', '/app/run_simulation.py'], capture_output=True, text=True)
    with open('replay.json') as f: r = json.load(f)
    if 'Player 1: -1' in subprocess.run(['python', '-c', "import json; r=json.load(open('replay.json')); print('Player 1: -1' if r['steps'][-1][0]['reward'] == -1 else '')"], capture_output=True, text=True).stdout:
        shutil.copy('replay.json', 'replay_loss.json')
        break
