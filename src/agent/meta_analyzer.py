import json
import os

class MetaAnalyzer:
    """
    Parses and stores tournament meta rulings, and dynamically records 
    opponent deck archetypes encountered across multiple games to build 
    an internal memory of the meta.
    """
    def __init__(self, memory_file_path: str = "archetype_memory.json"):
        self.rulings = {}
        self.memory_file_path = memory_file_path
        
        # Structure: { "archetype_signature": {"cards": {"card_name": frequency}, "wins": int, "losses": int} }
        self.archetype_memory = self._load_memory()
        
    def _load_memory(self) -> dict:
        if os.path.exists(self.memory_file_path):
            with open(self.memory_file_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_memory(self):
        with open(self.memory_file_path, 'w') as f:
            json.dump(self.archetype_memory, f, indent=4)
            
    def record_match_data(self, opponent_cards_seen: list, won_match: bool):
        """
        Records the opponent's deck structure after a game finishes.
        Creates a signature based on the most prominent Pokémon used.
        """
        if not opponent_cards_seen:
            return
            
        # Very simple signature: sort the cards alphabetically and hash, 
        # or just use the core attacking Pokemon as a signature.
        # Ensure all cards are strings for json serialization and joining
        str_cards = [str(c) for c in opponent_cards_seen]
        unique_cards = sorted(list(set(str_cards)))
        signature = ",".join(unique_cards[:5]) # Use top 5 alphabetically as a basic signature for grouping
        
        if signature not in self.archetype_memory:
            self.archetype_memory[signature] = {"cards": {}, "matches": 0, "wins_against": 0}
            
        memory_entry = self.archetype_memory[signature]
        memory_entry["matches"] += 1
        if won_match:
            memory_entry["wins_against"] += 1
            
        for card in str_cards:
            if card not in memory_entry["cards"]:
                memory_entry["cards"][card] = 0
            # Track the average frequency of this card appearing in this archetype
            memory_entry["cards"][card] += 1
            
        self._save_memory()

    def guess_opponent_archetype(self, observed_cards: list) -> dict:
        """
        Given the cards the opponent has played so far in a CURRENT game, 
        compare against our memory to guess the full decklist and probabilities 
        of hidden cards.
        """
        best_match = None
        highest_overlap = 0
        
        for sig, data in self.archetype_memory.items():
            # Count how many of the currently observed cards exist in this archetype memory
            overlap = sum(1 for card in observed_cards if card in data["cards"])
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_match = data
                
        return best_match if best_match else {}

    def load_rulings(self):
        """
        Scrapes the PokeGym Compendium for edge-case rulings to ensure 
        the simulator and belief tracker respect complex interactions.
        """
        url = "https://compendium.pokegym.net/"
        import urllib.request
        try:
            from bs4 import BeautifulSoup
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                soup = BeautifulSoup(html, 'html.parser')
                
                ruling_elements = soup.find_all('div', class_='ruling')
                for el in ruling_elements:
                    text = el.get_text(strip=True)
                    self.rulings[text[:20]] = text
                    
            print(f"Loaded {len(self.rulings)} ruling categories from PokeGym.")
        except Exception as e:
            print(f"Failed to fetch PokeGym Rulings: {e}")
