import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make

import sys
import os
try:
    from src.rl.vectorizer import vectorize_state, MAX_OPTIONS
except ImportError:
    sys.path.append(os.getcwd())
    from src.rl.vectorizer import vectorize_state, MAX_OPTIONS

class CabtGymEnv(gym.Env):
    def __init__(self, opponent_agent=None, my_index=0):
        super().__init__()
        self.env = make("cabt")
        self.my_index = my_index
        self.opp_index = 1 - my_index
        
        # We can pass the path to slowking_agent.py here
        if opponent_agent is None:
            self.opponent_agent = "random"
        else:
            self.opponent_agent = opponent_agent
            
        self.observation_space = spaces.Dict({
            "card_ids": spaces.MultiDiscrete([1300] * 90),
            "scalars": spaces.Box(low=-1000, high=1000, shape=(40,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(MAX_OPTIONS,), dtype=np.int8)
        })
        
        self.action_space = spaces.Discrete(MAX_OPTIONS)
        self.runner = None
        self._last_state = None
        self._slowking_deck = [
            162, 162, 162, 162,  # Slowpoke
            163, 163, 163,       # Slowking
            144, 144,            # Kyurem
            756, 756,            # Mega Kangaskhan ex
            115,                 # Conkeldurr
            184,                 # Latias ex
            140,                 # Fezandipiti ex
            224,                 # Annihilape
            1071,                # Meowth ex
            183,                 # Smoochum
            880,                 # Spectrier
            272,                 # Lillie's Clefairy ex
            1227, 1227, 1227, 1227,  # Lillie's Determination
            1188, 1188, 1188,        # Ciphermaniac's Codebreaking
            1210,                # Brock's Scouting
            1231,                # Dawn
            1184,                # Lana's Aid
            1152, 1152, 1152, 1152,  # Poké Pad
            1121, 1121, 1121, 1121,  # Ultra Ball
            1097, 1097, 1097,        # Night Stretcher
            1146, 1146, 1146,        # Wondrous Patch
            1092,                # Secret Box
            1123,                # Switch
            1156,                # Lucky Helmet
            1248, 1248, 1248, 1248,  # Academy at Night
            19, 19, 19, 19,          # Telepathic Psychic Energy
            5, 5, 5,                 # Psychic Energy
            9, 9, 9                  # Boomerang Energy
        ]
        
    def _get_obs_dict(self, state):
        return state[self.my_index].observation
        
    def _is_my_turn(self, state):
        obs_dict = self._get_obs_dict(state)
        # It's our turn if we have options to select
        opts = obs_dict.get('select', {})
        if opts is None:
            return False
        return len(opts.get('option', [])) > 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # We run env.reset() which initializes the state
        state = self.env.reset(num_agents=2)
        
        # Send deck for both players (Step 0)
        actions = [self._slowking_deck, self._slowking_deck]
        state = self.env.step(actions)
        
        # Fast forward until it's our turn
        state = self._fast_forward(state)
        self._last_state = state
        
        vec = vectorize_state(self._get_obs_dict(state), self.my_index)
        return vec, {}
        
    def _fast_forward(self, state):
        # We loop until it's our turn or the game is done
        while not self._is_my_turn(state) and not self.env.done:
            # If we don't have options, our action is empty list
            my_action = []
            
            # Opponent action
            opp_obs = state[self.opp_index].observation
            opp_opts = opp_obs.get('select', {})
            
            if opp_opts and len(opp_opts.get('option', [])) > 0:
                # Get action from opponent agent (random for now)
                # If we had a real agent, we would load it and call it
                import random
                opp_action = [random.randint(0, len(opp_opts['option'])-1)]
            else:
                opp_action = []
                
            actions = [None, None]
            actions[self.my_index] = my_action
            actions[self.opp_index] = opp_action
            
            state = self.env.step(actions)
            
        return state
        
    def step(self, action):
        my_action = [int(action)]
        
        # Check if the action is valid using the action mask
        # If it is invalid, we do NOT pass it to the environment (which would crash/terminate it),
        # but instead we return a negative reward immediately and keep the same state.
        is_valid = False
        if self._last_state is not None:
            vec = vectorize_state(self._get_obs_dict(self._last_state), self.my_index)
            if vec["action_mask"][int(action)] == 1:
                is_valid = True
                
        if not is_valid:
            # Castigo por hacer una acción ilegal, no avanzamos el entorno
            reward = -10.0
            vec = vectorize_state(self._get_obs_dict(self._last_state), self.my_index)
            done = False
            return vec, reward, done, False, {"invalid_action": True}
        
        opp_obs = self._last_state[self.opp_index].observation
        opp_opts = opp_obs.get('select', {})
        
        if opp_opts and len(opp_opts.get('option', [])) > 0:
            import random
            opp_action = [random.randint(0, len(opp_opts['option'])-1)]
        else:
            opp_action = []
            
        actions = [None, None]
        actions[self.my_index] = my_action
        actions[self.opp_index] = opp_action
        
        state = self.env.step(actions)
        
        # Check if the game is done
        done = self.env.done
        
        # Obtain the selected action object to shape rewards
        last_obs = self._get_obs_dict(self._last_state)
        options = last_obs.get('select', {}).get('option', [])
        action_obj = options[int(action)] if options and int(action) < len(options) else {}
        action_type = action_obj.get('type')
        
        intermediate_reward = 0.0
        if action_type == 9: # ACTION_ATTACH_ENERGY
            intermediate_reward += 0.5
        elif action_type == 10: # ACTION_EVOLVE
            intermediate_reward += 2.0
        elif action_type == 13: # ACTION_ATTACK
            intermediate_reward += 1.0
        elif action_type == 12: # ACTION_END_TURN
            intermediate_reward -= 0.5
            
        # Reward
        final_reward = state[self.my_index].reward if state[self.my_index].reward is not None else 0.0
        
        # We cap intermediate rewards if the game is done to prioritize the Win/Loss signal (+1/-1)
        # But wait, Kaggle returns 1.0 for win, -1.0 for loss. If intermediate reward dominates, 
        # it might not care about winning. We scale intermediate rewards so they are much smaller than a Win.
        # Actually, let's just add it. The discount factor (gamma=0.99) will carry the win signal back.
        # For PPO, a win of +10 is better than +1 if intermediate rewards are like +1.
        if done:
            if final_reward > 0:
                final_reward = 10.0 # Amplificar la victoria para sobrepasar cualquier intermediate reward
            elif final_reward < 0:
                final_reward = -10.0 # Amplificar la derrota
                
        reward = final_reward + intermediate_reward
        
        if not done:
            state = self._fast_forward(state)
            # Recheck done after fast_forward
            done = self.env.done
            
        self._last_state = state
        vec = vectorize_state(self._get_obs_dict(state), self.my_index)
        
        return vec, float(reward), done, False, {}
