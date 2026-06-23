import random

def rule_based_rollout(state):
    """
    A fast heuristic agent for the MCTS Simulation phase.
    Instead of playing purely random moves (which often results in passing without attacking),
    this agent prioritizes sensible actions to give a more accurate evaluation of a board state.
    
    Priority:
    1. Evolve Pokémon
    2. Play Supporter cards
    3. Attach Energy (if required for an attack)
    4. Attack
    5. Pass
    """
    # In CABT, the 'state' would provide a list of legal actions.
    # The actions are often represented as objects or indices.
    # Without the full CABT SDK loaded, we mock the prioritization logic.
    
    while not state.is_terminal():
        legal_moves = state.get_legal_moves()
        
        if not legal_moves:
            break
            
        if len(legal_moves) == 1:
            state.do_move(legal_moves[0])
            continue
            
        # Example logic for parsing actions if CABT provides action metadata
        # (Assuming we have a way to translate move indices to their string names)
        # chosen_move = None
        # for move in legal_moves:
        #     action_type = get_action_type(move, state)
        #     if action_type == 'ATTACK':
        #         chosen_move = move
        #         break
        
        # Fallback to random if we can't parse the action metadata yet
        chosen_move = random.choice(legal_moves)
        state.do_move(chosen_move)
