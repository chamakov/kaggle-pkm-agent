from src.agent.state_interface import GameStateInterface
import random

class CoinGuessGame(GameStateInterface):
    """
    A simple 2-player game with hidden information.
    Player 0 hides a coin (heads=0, tails=1).
    Player 1 guesses the coin.
    If Player 1 guesses right, they win. Else Player 0 wins.
    """
    def __init__(self):
        self.player_to_move = 0
        self.hidden_coin = None
        self.guess = None
        self.winner = None

    def get_current_player(self) -> int:
        return self.player_to_move

    def get_legal_moves(self) -> list:
        if self.player_to_move == 0:
            return [0, 1] # hide heads or tails
        elif self.player_to_move == 1:
            return [0, 1] # guess heads or tails
        return []

    def do_move(self, move):
        if self.player_to_move == 0:
            self.hidden_coin = move
            self.player_to_move = 1
        elif self.player_to_move == 1:
            self.guess = move
            if self.guess == self.hidden_coin:
                self.winner = 1
            else:
                self.winner = 0
            self.player_to_move = -1

    def get_result(self, player: int) -> float:
        if self.winner == player:
            return 1.0
        return 0.0

    def is_terminal(self) -> bool:
        return self.winner is not None

    def clone(self) -> 'GameStateInterface':
        c = CoinGuessGame()
        c.player_to_move = self.player_to_move
        c.hidden_coin = self.hidden_coin
        c.guess = self.guess
        c.winner = self.winner
        return c

    def clone_and_randomize(self, observer: int) -> 'GameStateInterface':
        c = self.clone()
        # If observer is 1 (guesser), they shouldn't know the hidden coin
        # So we randomize it
        if observer == 1 and self.hidden_coin is not None:
            c.hidden_coin = random.choice([0, 1])
        return c
