import pandas as pd
import re

def silver_labeler(text):
    """
    Assigns a label based on keywords in the 'remark' / outcome text.
    2 = Highly Likely (Success)
    1 = Partial / In Progress
    0 = Unlikely / Stalled
    """
    if not isinstance(text, str):
        return 0
        
    text = str(text).lower().strip()
    
    # --- 1. EXPANDED SUCCESS KEYWORDS (The Fix for 'Unlikely' bug) ---
    success_keywords = [
        "completed", "launched", "achieved", "implemented", "set up", 
        "established", "commissioned", "dedicated", "inaugurated", 
        "fulfilled", "execution completed", "done", "operational", 
        "functional", "created", "built", "notified", "started", 
        "commenced", "opened", "delivered"
    ]
    
    # --- 2. EXPANDED PARTIAL KEYWORDS ---
    partial_keywords = [
        "underway", "progress", "ongoing", "work in progress", 
        "tender", "sanctioned", "approved", "proposed", "draft", 
        "consultation", "identified", "process", "dpr", "phase"
    ]

    # --- 3. PRIORITY LOGIC ---
    if any(word in text for word in success_keywords):
        return 2
    elif any(word in text for word in partial_keywords):
        return 1
    else:
        return 0

def process_data(df):
    """
    Takes the raw dataframe (Promise, Remark), labels it, 
    and returns the clean format for BERT.
    """
    # Ensure column names are lower case for safety
    df.columns = [c.lower() for c in df.columns]
    
    # Apply the labeler function to the 'remark' column
    # (Assuming the parser output has a 'remark' column for the outcome)
    if 'remark' in df.columns:
        df['label'] = df['remark'].apply(silver_labeler)
    else:
        # Fallback if column is named differently
        print("Warning: 'remark' column not found. Using 'text' or last column.")
        df['label'] = df.iloc[:, -1].apply(silver_labeler)

    # BERT needs 'text' and 'label' columns
    # We rename 'promise' to 'text' if it exists
    if 'promise' in df.columns:
        df = df.rename(columns={'promise': 'text'})
    
    # Return only the columns we need for training
    return df[['text', 'label']]