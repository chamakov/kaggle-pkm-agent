import time

class AgentEvaluator:
    """
    Evaluates the ISMCTS agent by tracking its performance,
    decision times, and the size/depth of its search trees.
    Crucial for the Kaggle Strategy Category write-up.
    """
    def __init__(self):
        self.match_results = []
        self.decision_times = []
        self.tree_sizes = []
        
    def evaluate_move(self, ismcts_agent_fn, state, observer):
        """
        Wraps the ISMCTS call to track metrics.
        """
        start_time = time.time()
        
        # Execute ISMCTS
        # In a real scenario, the agent_fn would return both the best move
        # and the root_node of the tree so we can analyze it.
        # For now, we assume it just returns the best move.
        best_move = ismcts_agent_fn(state, itermax=1000, observer=observer)
        
        end_time = time.time()
        self.decision_times.append(end_time - start_time)
        
        return best_move

    def log_match_result(self, won: bool, turns: int):
        self.match_results.append({"won": won, "turns": turns})
        
    def generate_strategy_report(self):
        """
        Generates data for the Kaggle Strategy Write-up.
        """
        wins = sum(1 for r in self.match_results if r['won'])
        total = len(self.match_results)
        win_rate = wins / total if total > 0 else 0
        avg_decision_time = sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0
        
        report = f"--- ISMCTS Agent Evaluation ---\n"
        report += f"Matches Played: {total}\n"
        report += f"Win Rate: {win_rate:.2%}\n"
        report += f"Average Decision Time per Turn: {avg_decision_time:.4f} seconds\n"
        return report

class MiniMCTSEvaluator:
    """
    A lightweight heuristic lookahead tree in Python.
    Since the CABT C++ sim API is hidden, we use this to estimate the next state 
    after applying an action, and then evaluate the synergy of that expected state.
    """
    def __init__(self):
        from src.agent.synergy_analyzer import ComboSequencer
        self.sequencer = ComboSequencer()
        
    def score_action(self, move_index: int, state) -> float:
        # 1. Base Heuristic Score of the current action
        base_score = self.sequencer.evaluate_synergy(move_index, state)
        
        # 2. 1-Ply Simulated Lookahead
        # We attempt to guess the board state AFTER the action to see if it improves our board
        expected_bonus = 0.0
        
        try:
            obs = state.observation
            select = obs.get("select", {})
            options = select.get("option", [])
            action = options[move_index]
            action_type = action.get("type")
            
            # If we play a supporter/item that searches, we assume it will let us play an attacker next
            if action_type == 3: # Supporter
                expected_bonus += 2.0
                
            # If we promote a tank, we assume our future threat level decreases
            if action_type == 8:
                in_play_area = action.get("inPlayArea", -1)
                if in_play_area == 4: # Active
                    expected_bonus += 5.0
                    
            # If we attack, we assume opponent's HP decreases
            if action_type == 13:
                expected_bonus += 5.0
        except Exception as e:
            pass # Fallback if state structure changes
            
        return base_score + expected_bonus
