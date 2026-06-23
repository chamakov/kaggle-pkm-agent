import csv
import sys
import re

def parse_decklist(decklist_text, csv_path):
    # Load the card database
    cards_db = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards_db.append(row)
            
    deck_ids = []
    missing = []
    
    lines = decklist_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Pokémon') or line.startswith('Trainer') or line.startswith('Energy'):
            continue
            
        # Parse line: count Name Expansion Number
        # Example: 4 Mega Kangaskhan ex MEG 104
        # Or: 1 Grass Energy MEE 1
        
        match = re.match(r'^(\d+)\s+(.+?)\s+([A-Z0-9]{3})\s+(\d+)$', line)
        if not match:
            # Maybe it doesn't have expansion/number
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if not match:
                print(f"Failed to parse line: {line}")
                continue
            count = int(match.group(1))
            name = match.group(2)
            expansion = None
            number = None
        else:
            count = int(match.group(1))
            name = match.group(2)
            expansion = match.group(3)
            number = match.group(4)
            
        # Special case for basic energies
        if name == "Grass Energy":
            name = "Basic {G} Energy"
        elif name == "Fire Energy":
            name = "Basic {R} Energy"
        elif name == "Water Energy":
            name = "Basic {W} Energy"
        elif name == "Lightning Energy":
            name = "Basic {L} Energy"
        elif name == "Psychic Energy":
            name = "Basic {P} Energy"
        elif name == "Fighting Energy":
            name = "Basic {F} Energy"
        elif name == "Darkness Energy":
            name = "Basic {D} Energy"
        elif name == "Metal Energy":
            name = "Basic {M} Energy"
            
        # Find the card in the DB
        found_id = None
        if expansion and number:
            for card in cards_db:
                if card['Expansion'] == expansion and card['Collection No.'] == number:
                    found_id = card['Card ID']
                    break
                    
        # If not found by exact expansion/number, try just by name
        if not found_id:
            # Replace straight apostrophes with curly ones to match the DB
            search_name = name.lower().replace("'", "’")
            for card in cards_db:
                if card['Card Name'].lower() == search_name:
                    found_id = card['Card ID']
                    break
                    
        if found_id:
            deck_ids.extend([found_id] * count)
        else:
            missing.append(line)
            
    return deck_ids, missing

if __name__ == "__main__":
    decklist_text = """
Pokémon: 10
4 Mega Kangaskhan ex MEG 104
3 Dwebble DRI 11
3 Crustle DRI 12

Trainer: 37
4 Lillie's Determination MEG 119
4 Boss's Orders MEG 114
4 Team Rocket's Petrel DRI 176
2 Hilda WHT 84
2 Eri TEF 146
1 Xerosic's Machinations SFA 64
1 Super Potion JTG 158
1 Bianca's Devotion TEF 142
1 Lisia's Appeal SSP 179
4 Jumbo Ice Cream PFL 91
3 Pokégear 3.0 SVI 186
2 Buddy-Buddy Poffin TEF 144
1 Ultra Ball MEG 131
1 Switch MEG 130
1 Hand Trimmer TEF 150
1 Hero's Cape TEF 152
1 Handheld Fan TWM 150
1 Team Rocket's Factory DRI 173
1 Community Center TWM 146
1 Festival Grounds TWM 149

Energy: 13
4 Spiky Energy JTG 159
4 Growing Grass Energy POR 86
4 Mist Energy TEF 161
1 Grass Energy MEE 1
    """
    
    csv_path = "/Users/chamakov/Repos/Pokemon-kaggle-competition/resources/cards-things/EN_Card_Data.csv"
    deck_ids, missing = parse_decklist(decklist_text, csv_path)
    
    if missing:
        print("Could not find the following cards in the database:")
        for m in missing:
            print(f"  - {m}")
            
    if deck_ids:
        out_path = "deck.csv"
        with open(out_path, "w") as f:
            for card_id in deck_ids:
                f.write(f"{card_id}\n")
        print(f"Successfully generated {out_path} with {len(deck_ids)} cards.")
    else:
        print("No cards parsed.")
