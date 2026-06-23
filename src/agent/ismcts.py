import math
import random
from typing import List, Any, Optional
from src.agent.synergy_analyzer import ComboSequencer

class Node:
    """
    A node in the Information Set Monte Carlo Tree Search.
    """
    def __init__(self, move: Any = None, parent: 'Node' = None, player_just_moved: int = None):
        self.move = move
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.untried_moves: List[Any] = []
        self.children: List['Node'] = []
        self.visits = 0
        self.wins = 0.0

    def add_child(self, move: Any, player_just_moved: int) -> 'Node':
        """
        Adds a new child node for this move.
        """
        n = Node(move=move, parent=self, player_just_moved=player_just_moved)
        self.untried_moves.remove(move)
        self.children.append(n)
        return n

    def update(self, result: float):
        """
        Updates the node statistics with a simulation result.
        """
        self.visits += 1
        self.wins += result

    def get_child_with_move(self, move: Any) -> Optional['Node']:
        for child in self.children:
            if child.move == move:
                return child
        return None

def ismcts(root_state: Any, itermax: int, observer: int) -> Any:
    """
    Conducts an ISMCTS search starting from root_state and returns the best move.
    """
    root_node = Node()
    sequencer = ComboSequencer()

    for _ in range(itermax):
        node = root_node
        
        # 1. Determinization
        # Randomize hidden information from the perspective of the observer
        state = root_state.clone_and_randomize(observer)

        # 2. Selection
        # Traverse the tree while fully expanded
        while not state.is_terminal() and not node.untried_moves:
            # Need to populate untried moves if this node was just visited for the first time
            if len(node.children) == 0 and len(node.untried_moves) == 0:
                 node.untried_moves = state.get_legal_moves()
                 if not node.untried_moves:
                     break # no moves available
                 
            # Find all legal moves from the current determinized state
            legal_moves = state.get_legal_moves()
            
            # Untried moves for this determinized state
            untried_legal_moves = [m for m in legal_moves if node.get_child_with_move(m) is None]

            if untried_legal_moves:
                # Some moves are untried, break selection to expand
                node.untried_moves = untried_legal_moves
                break
                
            # Use UCB1 formula to select the best child
            # We only consider children whose move is legal in the current determinized state
            legal_children = [c for c in node.children if c.move in legal_moves]
            if not legal_children:
                 # It's possible that due to determinization, a node has children but none are valid now
                 # We treat this as an unexpanded node
                 node.untried_moves = legal_moves
                 break
                 
            # UCB selection
            best_score = -float('inf')
            best_child = None
            for child in legal_children:
                # child.wins is wins for the player who just moved to reach this child
                # UCB value = (wins / visits) + c * sqrt(ln(parent.visits) / child.visits)
                # Note: wins are from the perspective of player_just_moved
                exploitation = child.wins / child.visits
                exploration = math.sqrt(2 * math.log(node.visits) / child.visits)
                
                # Progressive Bias
                synergy_score = sequencer.evaluate_synergy(child.move, state)
                progressive_bias = synergy_score / (child.visits + 1)
                
                score = exploitation + exploration + progressive_bias
                if score > best_score:
                    best_score = score
                    best_child = child
            
            state.do_move(best_child.move)
            node = best_child

        # 3. Expansion
        # Add a new child node for an untried move
        if node.untried_moves and not state.is_terminal():
            # Pick untried move with highest synergy to guide expansion
            best_untried = None
            best_untried_score = -float('inf')
            for m in node.untried_moves:
                s_score = sequencer.evaluate_synergy(m, state)
                s_score += random.uniform(0, 0.1) # Tie breaker
                if s_score > best_untried_score:
                    best_untried_score = s_score
                    best_untried = m
                    
            move = best_untried
            player_moving = state.get_current_player()
            state.do_move(move)
            node = node.add_child(move, player_moving)

        # 4. Simulation
        # Play out the game using a fast heuristic agent to get a better evaluation
        # than purely random moves.
        from src.agent.heuristic_rollout import rule_based_rollout
        rule_based_rollout(state)

        # 5. Backpropagation
        # Update the nodes in the path with the result
        while node is not None:
            if node.player_just_moved is not None:
                 result = state.get_result(node.player_just_moved)
                 node.update(result)
            else:
                 # root node
                 node.update(0)
            node = node.parent

    # Return the move that was most visited
    # Ensure it's legal in the TRUE root_state
    legal_root_moves = root_state.get_legal_moves()
    best_move = None
    most_visits = -1
    for child in root_node.children:
        if child.move in legal_root_moves and child.visits > most_visits:
            most_visits = child.visits
            best_move = child.move

    return best_move
