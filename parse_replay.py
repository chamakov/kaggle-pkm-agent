import json
import csv

def load_card_db():
    db = {}
    try:
        with open("resources/cards-things/EN_Card_Data.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                db[int(row["Card ID"])] = row["Card Name"]
    except Exception as e:
        print(f"Error loading DB: {e}")
    return db

def main():
    cards = load_card_db()
    with open("replay.json", "r") as f:
        replay = json.load(f)
        
    steps = replay.get("steps", [])
    print(f"Total Steps: {len(steps)}")
    
    with open("parsed_replay.txt", "w") as out:
        for i, step in enumerate(steps):
            out.write(f"\n--- STEP {i} ---\n")
            for player_id in range(2):
                player_state = step[player_id]
                action = player_state.get("action")
                obs = player_state.get("observation", {})
                logs = obs.get("logs", [])
                
                # Format the action if it's a list of ints (like step 0)
                formatted_action = action
                if isinstance(action, list) and len(action) > 0 and isinstance(action[0], int):
                    formatted_action = [cards.get(a, f"Unknown({a})") for a in action]
                    
                # Format the logs to include English card names
                formatted_logs = []
                for log in logs:
                    new_log = log.copy()
                    if "cardId" in new_log:
                        new_log["cardName"] = cards.get(new_log["cardId"], f"Unknown({new_log['cardId']})")
                    formatted_logs.append(new_log)
                    
                out.write(f"Player {player_id}:\n")
                if i > 0:
                    out.write(f"  Action: {formatted_action}\n")
                if formatted_logs:
                    out.write(f"  Logs: {formatted_logs}\n")

if __name__ == "__main__":
    main()
