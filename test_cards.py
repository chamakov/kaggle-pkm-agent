from kaggle_environments import make

def test_deck(deck):
    env = make("cabt")
    
    def agent1(obs, conf):
        if obs.get("select") is None: return deck
        return [0]
    
    def agent2(obs, conf):
        if obs.get("select") is None: return deck
        return [0]
        
    steps = env.run([agent1, agent2])
    # If the battle crashes on turn 0, it has 2 steps and an error in steps[0][0]
    if len(steps) == 2 and "error" in steps[0][0]:
        return False
    return True

def main():
    try:
        with open("deck.csv", "r") as f:
            user_deck = [int(line.strip()) for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error loading deck: {e}")
        return

    unique_cards = list(set(user_deck))
    print(f"Testing {len(unique_cards)} unique cards for CABT engine compatibility...")
    
    # Base cards that we know work
    BASIC_POKEMON = 157 # Chewtle from baseline deck
    BASIC_ENERGY = 5    # Basic P Energy from baseline deck
    
    invalid_cards = []
    
    for card_id in unique_cards:
        # Construct a test deck: 1 Basic Pokemon, 4 copies of the test card, 55 Basic Energies
        # (If the test card IS the basic pokemon or energy, it just adds more, which is fine as long as <=4 except for basic energy)
        test_deck = [BASIC_POKEMON] * 4 + [BASIC_ENERGY] * 52 + [card_id] * 4
        # Slice to 60 just in case
        test_deck = test_deck[:60]
        
        is_valid = test_deck_runner(test_deck)
        if not is_valid:
            invalid_cards.append(card_id)
            print(f"Card ID {card_id} is REJECTED by the engine.")
        else:
            print(f"Card ID {card_id} is ACCEPTED.")
            
    print("\n--- Summary ---")
    print(f"Invalid Cards: {invalid_cards}")

def test_deck_runner(deck):
    return test_deck(deck)

if __name__ == "__main__":
    main()
