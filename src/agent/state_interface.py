from abc import ABC, abstractmethod
from typing import List, Any

class GameStateInterface(ABC):
    """
    Abstract base class for the Game State.
    The core Pokémon TCG engine must implement this interface
    so that the ISMCTS agent can interact with it.
    """

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Returns the ID of the player whose turn it is to move.
        """
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """
        Returns a list of all legal moves available from this state 
        for the current player.
        """
        pass

    @abstractmethod
    def do_move(self, move: Any) -> None:
        """
        Applies the given move to the state, mutating it.
        """
        pass

    @abstractmethod
    def get_result(self, player: int) -> float:
        """
        Returns the result of the game from the perspective of the given player.
        Usually 1.0 for a win, 0.0 for a loss, 0.5 for a draw.
        Should only be called if is_terminal() is True.
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the game has ended, False otherwise.
        """
        pass

    @abstractmethod
    def clone(self) -> 'GameStateInterface':
        """
        Returns a deep copy of the state. 
        Crucial for MCTS simulations so we don't mutate the real game.
        """
        pass

    @abstractmethod
    def clone_and_randomize(self, observer: int) -> 'GameStateInterface':
        """
        The core of Information Set MCTS.
        Creates a deep copy of the state, but randomizes any information
        that is hidden from the `observer` (e.g., shuffling the deck, 
        randomizing the opponent's hand based on belief states).
        """
        pass
