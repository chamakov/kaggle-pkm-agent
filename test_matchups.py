import json
import random
from kaggle_environments import make
from src.agent.agent import agent_fn as smart_agent_base
from src.agent.meta_analyzer import MetaAnalyzer

# Custom Balanced Deck (Player 1) - Our Smart Agent
deck1 = [
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, # 20 Energies
    87, 87, 87, 87, # 4x Dedenne SSP (Setup)
    331, 331, 331, 331, # 4x XerneasEX (Attacker)
    408, 408, 408, 408, # 4x Houndour (Utility)
    474, 474, 474, 474, # 4x Porygon2 (Supporter)
    77, 77, 77, 77, # 4x Litten
    156, 156, 156, 156, # 4x Crabominable
    554, 554, 554, 554, # 4x Audino
    528, 528, 528, 528, # 4x Timburr
    530, 530, 530, 530, # 4x Conkeldurr
    585, 585, 585, 585  # 4x Items
]

# Opponent Deck A: Swarm / Aggro (Low HP, Fast)
deck_aggro = [
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, # 20 Energies
    77, 77, 77, 77, # 4x Litten
    157, 157, 157, 157, # 4x Chewtle
    408, 408, 408, 408, # 4x Houndour
    528, 528, 528, 528, # 4x Timburr
    156, 156, 156, 156, # 4x Crabominable
    474, 474, 474, 474, # 4x Porygon2
    585, 585, 585, 585, # 4x Item
    630, 630, 630, 630, # 4x Supporter
    530, 530, 530, 530, # 4x Conkeldurr
    576, 576, 576, 576  # 4x Samurott
][:60]

# Opponent Deck B: Heavy Tanks (High HP, Slow)
deck_tank = [
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, # 20 Energies
    331, 331, 331, 331, # 4x XerneasEX
    554, 554, 554, 554, # 4x Audino
    87, 87, 87, 87,     # 4x Dedenne
    408, 408, 408, 408, # 4x Houndour
    474, 474, 474, 474, # 4x Porygon2
    585, 585, 585, 585, # 4x Item
    630, 630, 630, 630, # 4x Supporter
    157, 157, 157, 157, # 4x Chewtle
    77, 77, 77, 77,     # 4x Litten
    156, 156, 156, 156  # 4x Crabominable
][:60]

def smart_agent(obs, conf):
    if obs.get("select") is None: return deck1
    return smart_agent_base(obs, conf)

def get_rule_based_agent(deck):
    def rule_based_agent(obs, conf):
        if obs.get("select") is None: return deck
        options = obs["select"].get("option", [])
        if not options: return []
        priority = {3: 60, 10: 50, 8: 40, 9: 30, 13: 20, 12: 10}
        best_idx, best_score = 0, -1
        for i, opt in enumerate(options):
            score = priority.get(opt.get("type", 0), 0)
            if score > best_score:
                best_score, best_idx = score, i
        count = obs["select"].get("maxCount", 1)
        if count > 1: return random.sample(list(range(len(options))), min(count, len(options)))
        return [best_idx]
    return rule_based_agent

def run_match(match_name, opp_deck, out_file):
    print(f"\\n--- Running Match: {match_name} ---")
    env = make("cabt")
    steps = env.run([smart_agent, get_rule_based_agent(opp_deck)])
    
    final_step = steps[-1]
    rewards = final_step[0]["reward"], final_step[1]["reward"]
    print(f"Total Steps: {len(steps)}")
    print(f"Final Rewards: Player 1: {rewards[0]}, Player 2: {rewards[1]}")
    
    with open(out_file, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"Replay saved to {out_file}")

if __name__ == "__main__":
    run_match("Smart Agent vs Aggro Swarm", deck_aggro, "replay_aggro.json")
    run_match("Smart Agent vs Heavy Tanks", deck_tank, "replay_tank.json")
