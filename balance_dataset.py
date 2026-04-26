import pandas as pd
import random
import os

db_path = 'data/processed/gold_database.csv'
df = pd.read_csv(db_path)

# Count distribution
counts = df['label'].value_counts()
print("--- Original Distribution ---")
print(counts)

max_count = counts.max()
majority_class = counts.idxmax()

# Augmentation templates to create distinct strings (preventing exact duplicates)
# This acts as our "Text-SMOTE", creating variations of existing promises 
# so the neural network doesn't memorize exact duplicates and the LLM can still read it.
prefixes = [
    "Regarding the promise to ",
    "The administration stated to ",
    "A key manifesto point was to ",
    "It was proposed to ",
    "Policy objective: ",
    "Commitment made to ",
    "The government aimed to ",
    "As outlined in the manifesto, ",
    "Focusing on the goal to ",
    "Strategic priority: "
]

def generate_synthetic_text(base_text):
    base_text = str(base_text).strip()
    prefix = random.choice(prefixes)
    # lowercase the first letter of base_text if it's not an acronym
    if base_text and base_text[0].isupper() and (len(base_text) == 1 or base_text[1].islower()):
        base_text = base_text[0].lower() + base_text[1:]
    return prefix + base_text

new_rows = []

for label in df['label'].unique():
    if label == majority_class:
        continue
    
    current_count = counts[label]
    needed = max_count - current_count
    
    subset = df[df['label'] == label].to_dict('records')
    
    for i in range(needed):
        # randomly pick a base record (like SMOTE picking a neighbor)
        base_record = random.choice(subset)
        
        syn_row = base_record.copy()
        syn_row['original_text'] = generate_synthetic_text(base_record['original_text'])
        # mark year as Synthetic so we can track it
        syn_row['year'] = str(base_record['year']) + "_Synthetic"
        
        new_rows.append(syn_row)

synthetic_df = pd.DataFrame(new_rows)
balanced_df = pd.concat([df, synthetic_df], ignore_index=True)

# Shuffle the dataset so synthetic records are distributed randomly
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- New Balanced Distribution ---")
print(balanced_df['label'].value_counts())

balanced_df.to_csv(db_path, index=False)
print(f"\n[SUCCESS] Balanced dataset saved to {db_path} (Total records: {len(balanced_df)})")
