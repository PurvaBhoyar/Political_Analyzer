from nlp_engine import parser, labeler
import pandas as pd
import os
import csv

def rebuild():
    print("--- REBUILDING GOLD DATABASE (SEMANTIC) ---")
    
    # 1. Process 2014
    print("Processing 2014 data...")
    df_2014_review = parser.extract_review_table(r"data/raw/2014 BJP Manifesto Review.pdf")
    df_2014_labeled = labeler.process_data(df_2014_review, use_semantic=True)
    # Mapping to 2014 original text
    df_2014_manifesto = parser.extract_manifesto_promises(r"data/raw/Manifesto_English.pdf")
    
    # 2. Process 2019
    print("Processing 2019 data...")
    df_2019_review = parser.extract_folder_data(r"data/raw/bjp_2019")
    df_2019_labeled = labeler.process_data(df_2019_review, use_semantic=True)
    
    # 3. Combine
    df_2014_labeled['year'] = 2014
    df_2019_labeled['year'] = 2019
    
    df_gold = pd.concat([df_2014_labeled, df_2019_labeled], ignore_index=True)
    
    # Rename for consistency with checker
    df_gold = df_gold.rename(columns={'text': 'original_text'})
    
    os.makedirs("data/processed", exist_ok=True)
    df_gold.to_csv("data/processed/gold_database.csv", index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    
    print(f"Rebuild Complete. Total Gold Records: {len(df_gold)}")
    print("Label breakdown:\n", df_gold['label'].value_counts())

if __name__ == "__main__":
    rebuild()
