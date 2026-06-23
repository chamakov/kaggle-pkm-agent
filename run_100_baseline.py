import subprocess
import concurrent.futures

def run_game(i):
    res = subprocess.run(['python', '/app/run_simulation.py'], capture_output=True, text=True)
    if 'Player 1: 1' in res.stdout:
        return 'win'
    elif 'Player 1: -1' in res.stdout:
        return 'loss'
    return 'tie'

wins, losses, ties = 0, 0, 0
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(run_game, range(50))
    for r in results:
        if r == 'win': wins += 1
        elif r == 'loss': losses += 1
        else: ties += 1

print(f"Results after 50 games vs Baseline: {wins} Wins, {losses} Losses, {ties} Ties")
