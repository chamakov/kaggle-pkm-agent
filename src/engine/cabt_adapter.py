import copy
import random
from src.agent.state_interface import GameStateInterface
from src.agent.belief_tracker import BeliefStateTracker

class CabtGameState(GameStateInterface):
    """
    Adapter bridging the Kaggle 'cabt' engine (pokemon-tcg-ai-battle) 
    with our Information Set MCTS implementation.
    """
    def __init__(self, observation: dict, env_state: dict = None, belief_tracker: BeliefStateTracker = None):
        """
        :param observation: The observation dict provided by kaggle-environments cabt engine.
        :param env_state: Optional underlying complete state (used for cloning/simulating internally).
        :param belief_tracker: The tracker used to determinize hidden info for MCTS.
        """
        self.observation = observation
        self.env_state = env_state
        self.belief_tracker = belief_tracker
        
        # We assume the CABT engine provides 'player' and 'legal_actions' in the observation.
        self.current_player = observation.get("player", 0)
        options = observation.get("select", {}).get("option", []) if observation.get("select") else []
        self.legal_moves = list(range(len(options)))
        self.terminal = observation.get("status", "ACTIVE") in ["DONE", "ERROR"]
        self.rewards = observation.get("rewards", [0.0, 0.0])

    def get_current_player(self) -> int:
        return self.current_player

    def get_legal_moves(self) -> list:
        # In CABT, actions are indices corresponding to the selected legal move
        return self.legal_moves

    def do_move(self, move):
        """
        Applies a move. This requires mutating the internal env_state using the CABT simulator logic.
        (To be completed when the CABT SDK source is dropped in the workspace).
        """
        pass

    def get_result(self, player: int) -> float:
        # Returns 1.0 for a win, 0.0 for a loss
        base_reward = 0.0
        if player < len(self.rewards):
            if self.rewards[player] > 0:
                base_reward = 1.0
            elif self.rewards[player] < 0:
                base_reward = 0.0
            elif self.rewards[player] == 0:
                base_reward = 0.5 # Draw
                
        # Reward Shaping to avoid Deck-Outs
        current = self.observation.get("current")
        if current and "players" in current and len(current["players"]) > player:
            my_deck = current["players"][player].get("deckCount", 0)
            
            # If we lost AND our deck is 0, apply a penalty below 0.0
            # to make deck-outs explicitly worse than a standard loss
            if base_reward == 0.0 and my_deck == 0:
                return -0.2
                
            # If we won, the state is strictly positive regardless of deck count
            if base_reward == 1.0:
                return 1.0
                
            # Danger Zone Heuristic (for depth-limited evaluations or ties)
            if my_deck <= 3:
                # Slight penalty for being close to decking out
                return base_reward - 0.01 * (4 - my_deck)
                
        return base_reward

    def is_terminal(self) -> bool:
        return self.terminal

    def clone(self) -> 'GameStateInterface':
        """
        Deep copies the environment state so MCTS simulations don't ruin the real game state.
        """
        c = CabtGameState(
            copy.deepcopy(self.observation), 
            copy.deepcopy(self.env_state) if self.env_state else None,
            self.belief_tracker # Belief tracker is shared, we don't mutate it during simulation
        )
        return c

    def clone_and_randomize(self, observer: int) -> 'GameStateInterface':
        """
        Information Set Randomization.
        We clone the state, but we randomize the opponent's hand and face-down prize cards
        based on our BELIEF STATE of what archetypes they are playing.
        """
        c = self.clone()
        
        if c.env_state is not None and c.belief_tracker is not None:
            # Randomize opponent's hand, deck, and prizes based on rules and tracking
            c.env_state = c.belief_tracker.determinize_state(c.env_state)
            
        return c
