from src.agent.ismcts import ismcts
from tests.dummy_game import CoinGuessGame

def test_ismcts_runs():
    game = CoinGuessGame()
    # Player 0 hides a coin
    game.do_move(0) # Player 0 hides heads (0)

    # Now it's player 1's turn. They don't know the coin.
    assert game.get_current_player() == 1

    # Run ISMCTS for player 1
    # We do 100 iterations.
    best_move = ismcts(root_state=game, itermax=100, observer=1)
    
    # The best move should be 0 or 1.
    assert best_move in [0, 1]
    print(f"ISMCTS chose move: {best_move}")

if __name__ == "__main__":
    test_ismcts_runs()
    print("All tests passed.")
