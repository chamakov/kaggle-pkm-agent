from kaggle_environments import make

try:
    env = make("cabt")
    print("SUCCESS! CABT engine loaded inside Docker.")
    
    # Quick test
    with open("deck.csv", "r") as f:
        deck = [int(line.strip()) for line in f.readlines() if line.strip()]
        
    print(f"Loaded deck with {len(deck)} cards.")
    env = make("cabt", configuration={"decks": [deck, deck]})
    print("Environment initialized with decks!")
except Exception as e:
    print(f"FAILED to load CABT engine: {e}")
