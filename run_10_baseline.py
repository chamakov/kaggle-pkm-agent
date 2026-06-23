import subprocess
wins, losses, ties = 0, 0, 0

for i in range(10):
    res = subprocess.run(['python', '/app/run_simulation.py'], capture_output=True, text=True)
    if 'Player 1: 1' in res.stdout:
        wins += 1
    elif 'Player 1: -1' in res.stdout:
        losses += 1
    else:
        ties += 1
print(f"Results after 10 games vs Baseline: {wins} Wins, {losses} Losses, {ties} Ties")
