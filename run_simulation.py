import json
import random
from kaggle_environments import make

# Custom Balanced Deck (Player 1)
# Using known valid CABT card IDs: 
# 331 (XerneasEX), 297 (Magearna), 408 (Houndour), 5 (Psychic Energy)
deck1 = [
    331, 331, 331, 331, # 4x XerneasEX (Attacker)
    297, 297, 297, 297, # 4x Magearna (Draw Engine & Healer)
    408, 408, 408, 408, # 4x Houndour (Blocker/Paralyze)
] + [5] * 48            # 48x Psychic Energy

# The Kaggle baseline Abomasnow deck (Player 2)
deck2 = [
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    9, 9,
    77, 77, 77, 77,
    156, 156, 156, 156,
    157, 157, 157, 157,
    331, 331, 331, 331,
    408, 408, 408, 408,
    474, 474, 474, 474,
    528, 528, 528, 528,
    530, 530, 530, 530,
    532,
    554, 554, 554,
    576, 576, 576, 576,
    585, 585, 585, 585,
    630, 630, 630, 630,
]

def random_agent_1(observation, configuration):
    if observation.get("select") is None:
        return deck1
    select = observation["select"]
    options = list(range(len(select["option"])))
    count = select.get("maxCount", 1)
    return random.sample(options, min(count, len(options)))

def rule_based_agent_2(observation, configuration):
    if observation.get("select") is None:
        return deck2
    options = observation["select"].get("option", [])
    if not options: return []
    
    # Priority: Supporter/Item > Evolve > Bench > Attach Energy > Attack > End Turn
    priority = {3: 60, 10: 50, 8: 40, 9: 30, 13: 20, 12: 10}
    
    best_idx = 0
    best_score = -1
    for i, opt in enumerate(options):
        score = priority.get(opt.get("type", 0), 0)
        if score > best_score:
            best_score = score
            best_idx = i
            
    count = observation["select"].get("maxCount", 1)
    if count > 1:
        # If asked to pick multiple, pick random for simplicity of the rule-based bot
        import random
        return random.sample(list(range(len(options))), min(count, len(options)))
    return [best_idx]

from src.agent.agent import agent_fn as smart_agent_base

def smart_agent(obs, conf):
    if obs.get("select") is None: return deck1
    return smart_agent_base(obs, conf)

def main():
    print("Initializing CABT environment...")
    env = make("cabt")
    
    print("Running match: Synergy Agent vs Rule-Based Baseline...")
    steps = env.run([smart_agent, rule_based_agent_2])
    
    print(f"Match complete! Total steps (turns): {len(steps)}")
    
    # NEW: Save Meta-Learning Memory
    from src.agent.meta_analyzer import MetaAnalyzer
    
    final_step = steps[-1]
    my_reward = final_step[0].reward
    won_match = (my_reward == 1)
    
    seen_opponent_cards = set()
    # Iterate through history to find opponent's played cards
    for step in steps:
        if len(step) > 1 and step[0].observation:
            current = step[0].observation.get("current", {}) or {}
            players = current.get("players", [])
            if len(players) > 1: # Opponent is player 1
                for zone in ["active", "bench", "discard"]:
                    cards = players[1].get(zone, [])
                    for card in cards:
                        if isinstance(card, dict) and "id" in card:
                            seen_opponent_cards.add(card["id"])
                            
    if seen_opponent_cards:
        analyzer = MetaAnalyzer()
        opponent_cards = list(seen_opponent_cards)
        analyzer.record_match_data(opponent_cards, won_match)
        print(f"MetaAnalyzer: Saved {len(opponent_cards)} known opponent cards to archetype memory.")
    
    rewards = final_step[0]["reward"], final_step[1]["reward"]
    print(f"Final Rewards: Player 1: {rewards[0]}, Player 2: {rewards[1]}")
    
    out_file = "replay.json"
    with open(out_file, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"Replay saved to {out_file}")
    
    html_out = env.render(mode="html")
    with open("replay.html", "w") as f:
        f.write(html_out)
    print(f"Replay saved to replay.html")

if __name__ == "__main__":
    main()
