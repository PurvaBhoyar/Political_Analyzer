import pandas as pd

gold_2014 = pd.read_csv("data/processed/gold_2014_mapped.csv")
gold_2019 = pd.read_csv("data/processed/gold_2019_mapped.csv")

print("--- 2014 Sample Records (Text & Actual Outcome) ---")
# 0: Unlikely, 1: Partial, 2: Highly Likely
label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}

for idx, row in gold_2014.sample(5, random_state=1).iterrows():
    print(f"Text: {row['original_text'][:100]}...")
    print(f"Outcome: {label_map[int(row['label'])]}\n")

print("\n--- 2019 Sample Records (Text & Actual Outcome) ---")
for idx, row in gold_2019.sample(5, random_state=1).iterrows():
    print(f"Text: {row['original_text'][:100]}...")
    print(f"Outcome: {label_map[int(row['label'])]}\n")
