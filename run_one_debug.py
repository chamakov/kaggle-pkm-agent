import subprocess
import json
import run_simulation
from src.agent.agent import agent_fn as smart_agent_base

def smart_agent(obs, conf):
    if obs.get("select") is None: return run_simulation.deck1
    choice = smart_agent_base(obs, conf)
    
    # Let's see what the choices are
    actions = obs.get("select", {}).get("option", []) if obs.get("select") else obs.get("actions", [])
    if type(choice) == list and len(choice) > 0 and type(choice[0]) == int:
        c = choice[0]
        if c < len(actions):
            act = actions[c]
            print(f"Agent chose: {act}")
            
    return choice

def baseline_agent(obs, conf):
    return run_simulation.rule_based_agent_2(obs, conf)

from kaggle_environments import make
env = make("cabt")
steps = env.run([smart_agent, baseline_agent])
if steps[-1][0].reward == -1:
    print("WE LOST THIS DEBUG MATCH!")
