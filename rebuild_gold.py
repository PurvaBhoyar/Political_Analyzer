from nlp_engine import parser, labeler
import pandas as pd
import os
import csv
from pathlib import Path

def rebuild():
    print("--- REBUILDING GOLD DATABASE (SEMANTIC) ---")
    
    # Use relative paths for portability
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # 1. Process 2014
    print("Processing 2014 data...")
    pdf_2014_path = raw_dir / "2014 BJP Manifesto Review.pdf"
    if not pdf_2014_path.exists():
        print(f"Error: {pdf_2014_path} not found.")
        return

    df_2014_review = parser.extract_review_table(str(pdf_2014_path))
    # The review table ALREADY contains the promise text and remark. 
    # We label based on the remark.
    df_2014_labeled = labeler.process_data(df_2014_review, use_semantic=True)
    df_2014_labeled['year'] = 2014
    
    # 2. Process 2019
    print("Processing 2019 data...")
    folder_2019_path = raw_dir / "bjp_2019"
    if not folder_2019_path.exists():
        print(f"Error: {folder_2019_path} not found.")
        return

    df_2019_review = parser.extract_folder_data(str(folder_2019_path))
    df_2019_labeled = labeler.process_data(df_2019_review, use_semantic=True)
    df_2019_labeled['year'] = 2019
    
    print("Processing 2019 unlabelled CSV with LLM...")
    unlabelled_csv_path = raw_dir / "2019_review_unlabelled.csv"
    if unlabelled_csv_path.exists():
        from dotenv import load_dotenv
        import time
        from nlp_engine import llm_reviewer

        load_dotenv()
        df_unlabelled = pd.read_csv(str(unlabelled_csv_path))
        
        if 'original_text' in df_unlabelled.columns:
            df_unlabelled = df_unlabelled.rename(columns={'original_text': 'text'})
            
        print(f"Auto-labeling {len(df_unlabelled)} records via Llama-3 (Groq)...")
        new_labels = []
        for i, row in df_unlabelled.iterrows():
            promise = str(row['text'])
            print(f"[{i+1}/{len(df_unlabelled)}] Labeling: {promise[:50]}...")
            
            try:
                res = llm_reviewer.review_promise(promise, "Unknown", 0.0, [])
                verdict = res.get('llm_verdict', '')
                
                if 'Likely Fulfilled' in verdict:
                    lbl = 2
                elif 'Partially Fulfilled' in verdict:
                    lbl = 1
                elif 'Unlikely' in verdict:
                    lbl = 0
                else:
                    lbl = 1 # Cannot Determine implicitly default to In Progress/Unknown
            except Exception as e:
                print(f"Error labeling: {e}")
                lbl = 1
                
            new_labels.append(lbl)
            time.sleep(0.5) # Rate limit protection

        df_unlabelled['label'] = new_labels
        dfs_to_concat = [df_2014_labeled, df_2019_labeled, df_unlabelled]
    else:
        dfs_to_concat = [df_2014_labeled, df_2019_labeled]
    
    # 3. Combine
    df_gold = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Rename for consistency with checker
    df_gold = df_gold.rename(columns={'text': 'original_text'})
    
    # Ensure all required columns exist
    if 'sector' not in df_gold.columns:
        df_gold['sector'] = "General"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "gold_database.csv"
    
    df_gold.to_csv(str(output_path), index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    
    print(f"Rebuild Complete. Total Gold Records: {len(df_gold)}")
    print("Label breakdown:\n", df_gold['label'].value_counts())

if __name__ == "__main__":
    rebuild()
