import os
import numpy as np

# Skeleton for the kaggle environments agent
# This will be submitted as the final model wrapper.

# In Kaggle environments, the agent is just a function
def rl_agent(obs_dict, config=None):
    from src.rl.vectorizer import vectorize_state, MAX_OPTIONS
    
    # 1. Vectorize the state
    # (El agente siempre es P1 desde su perspectiva en Kaggle environments)
    my_index = 0  # To be safe, wait, does kaggle pass my_index in config?
    # Actually, in CABT obs.current.yourIndex gives us our index, but vectorize_state already handles this implicitly if we pass 0 as we always get our perspective.
    # Wait! In kaggle_environments, your perspective is always player 1? 
    # Let's read from the raw obs just in case, but vectorize_state uses to_observation_class.
    
    vec = vectorize_state(obs_dict, my_index=0)
    
    # 2. Extract action mask
    action_mask = vec["action_mask"]
    
    # 3. Predict with the model
    # (Dummy random choice from valid options)
    valid_actions = np.where(action_mask == 1)[0]
    
    if len(valid_actions) > 0:
        action = int(np.random.choice(valid_actions))
    else:
        action = 0
        
    return [action]

# El entorno puede llamar agent = rl_agent
