import random
import copy

class BeliefStateTracker:
    """
    Tracks the probabilities of hidden information in the Pokémon TCG (Opponent's Hand, 
    Deck, and Prize Cards) based on the official rules and observed game actions.
    """
    def __init__(self, opponent_id: int):
        self.opponent_id = opponent_id
        
        # A standard deck is 60 cards. 
        # Without knowing their decklist yet, we start with a generic pool or an assumed archetype.
        # For the Kaggle CABT environment, we might know the possible card pool.
        self.assumed_decklist = [] 
        
        # We track where we think cards are.
        self.known_opponent_hand = []
        self.opponent_hand_size = 7 # Starts at 7, then setup happens
        self.opponent_deck_size = 47 # 60 - 7 (hand) - 6 (prizes)
        self.opponent_prize_count = 6
        
        # All cards we haven't seen them play or discard
        self.unknown_cards_pool = [] 

    def update_from_observation(self, observation: dict):
        """
        Updates the belief state based on the latest CABT observation.
        Extracts known opponent cards for Meta Learning.
        """
        if not hasattr(self, "seen_opponent_cards"):
            self.seen_opponent_cards = set()
            
        current = observation.get("current")
        if not current:
            return
            
        players = current.get("players", [])
        if len(players) > self.opponent_id:
            opp_state = players[self.opponent_id]
            in_play = opp_state.get("inPlay", {})
            for area, cards_dict in in_play.items():
                for index, card in cards_dict.items():
                    card_id = card.get("cardId")
                    if card_id:
                        self.seen_opponent_cards.add(card_id)

    def record_card_played(self, card_id: str):
        """
        When the opponent plays a card, we remove it from the unknown pool.
        """
        if card_id in self.unknown_cards_pool:
            self.unknown_cards_pool.remove(card_id)

    def determinize_state(self, base_env_state):
        """
        Creates a concrete, fully observable game state from the belief state.
        This is required by ISMCTS to run simulations.
        
        According to TCG Rules:
        - The opponent has a hand of size X.
        - The opponent has Y prize cards.
        - The opponent has Z cards in deck.
        We must distribute the `unknown_cards_pool` into these zones randomly.
        """
        # We need a deep copy so we don't mutate the actual tracker's pool
        pool = copy.copy(self.unknown_cards_pool)
        random.shuffle(pool)
        
        determinized_hand = self.known_opponent_hand.copy()
        cards_needed_for_hand = self.opponent_hand_size - len(determinized_hand)
        
        # 1. Fill Hand
        for _ in range(max(0, cards_needed_for_hand)):
            if pool:
                determinized_hand.append(pool.pop())
                
        # 2. Fill Prizes
        determinized_prizes = []
        for _ in range(self.opponent_prize_count):
            if pool:
                determinized_prizes.append(pool.pop())
                
        # 3. Remaining goes to Deck
        determinized_deck = pool
        
        # The base_env_state (cabt simulator state) would then be mutated 
        # to reflect this specific configuration before the MCTS rollout begins.
        # e.g., base_env_state.set_opponent_hand(determinized_hand)
        
        return base_env_state
