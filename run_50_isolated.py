import json
import random
import sys
from kaggle_environments import make

# Import agents
import run_simulation
from src.agent.agent import agent_fn as smart_agent_base

def smart_agent(obs, conf):
    if obs.get("select") is None: return run_simulation.deck1
    return smart_agent_base(obs, conf)

def baseline_agent(obs, conf):
    return run_simulation.rule_based_agent_2(obs, conf)

def play_match(i):
    env = make("cabt")
    steps = env.run([smart_agent, baseline_agent])
    
    final_step = steps[-1]
    r1 = final_step[0].reward
    r2 = final_step[1].reward
    
    # Get last state
    deck1_count = -1
    deck2_count = -1
    
    if len(steps) > 1:
        for p in final_step:
            if isinstance(p, dict) and 'observation' in p:
                current = p['observation'].get('current', {})
                if current and current.get('players'):
                    deck1_count = current['players'][0].get('deckCount', -1)
                    deck2_count = current['players'][1].get('deckCount', -1)
                break
                
    if r1 == 1: return 'win', len(steps), deck1_count, deck2_count
    if r1 == -1: return 'loss', len(steps), deck1_count, deck2_count
    return 'tie', len(steps), deck1_count, deck2_count

if __name__ == '__main__':
    import concurrent.futures
    wins, losses, ties = 0, 0, 0
    reasons = {'deck_out': 0, 'ko': 0, 'time_limit': 0}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(play_match, range(50))
        for r, length, d1, d2 in results:
            if r == 'win': wins += 1
            elif r == 'loss': losses += 1
            else: ties += 1
            
            # Heuristic to guess why match ended
            if r == 'tie' and length >= 99:
                reasons['time_limit'] += 1
            elif d1 == 0 or d2 == 0:
                reasons['deck_out'] += 1
            else:
                reasons['ko'] += 1
    
    print(f"Results after 50 games: {wins} Wins, {losses} Losses, {ties} Ties")
    print(f"Termination Guess: {reasons['ko']} by KO, {reasons['deck_out']} by Deck Out, {reasons['time_limit']} by Time Limit")
