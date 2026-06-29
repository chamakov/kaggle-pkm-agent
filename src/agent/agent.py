import sys
import os

# Ensure the src module is in path if this script is run directly by kaggle-environments
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Ensure the src module is in path if this script is run directly by kaggle-environments
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def agent_fn(observation, configuration):
    """
    Kaggle-environments entry point.
    
    :param observation: A dictionary containing the current game state 
                        (e.g., player index, legal_actions, board status).
    :param configuration: Game configuration parameters.
    :return: The chosen action (integer index or specific format required by cabt).
    """
    player_id = observation.get("player", 0)
    
    # Handle initial deck selection edge case
    select_data = observation.get("select")
    options = select_data.get("option", []) if select_data else []
    if not options:
        return []
        
    # We use the MiniMCTSEvaluator to lookahead and score options.
    from src.agent.evaluator import MiniMCTSEvaluator
    import random
    
    evaluator = MiniMCTSEvaluator()
    scored_options = []
    
    # Wrap observation in a mock state object for the sequencer
    class MockState:
        def __init__(self, obs):
            self.observation = obs
    mock_state = MockState(observation)
    
    for i in range(len(options)):
        score = evaluator.score_action(i, mock_state)
        # Add slight noise for tiebreaking
        score += random.uniform(0, 0.1)
        scored_options.append((score, i))
        
    # Sort options by synergy score descending
    scored_options.sort(key=lambda x: x[0], reverse=True)
    
    count = observation.get("select", {}).get("maxCount", 1)
    return [item[1] for item in scored_options[:min(count, len(options))]]
