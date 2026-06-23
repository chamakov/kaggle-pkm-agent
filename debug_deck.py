import csv

deck_ids = [int(line.strip()) for line in open("deck.csv")]
print(f"Total IDs: {len(deck_ids)}, Unique IDs: {len(set(deck_ids))}")

with open("resources/cards-things/EN_Card_Data.csv") as f:
    for row in csv.DictReader(f):
        if int(row["Card ID"]) in deck_ids:
            print(row["Card Name"], "-", row.get("Type", ""), "-", row.get("Stage", ""))
