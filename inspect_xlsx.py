import pandas as pd
import os

file_path = r'c:\Users\DELL\political_analyzer\data\raw\Copy of Manifesto Document Sheet.xlsx'
if os.path.exists(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print("\nShape:", df.shape)
        print("\nHead:\n", df.head())
        
        # Check for unique values in potential label columns
        for col in df.columns:
            if 'status' in col.lower() or 'remark' in col.lower() or 'progress' in col.lower():
                print(f"\nValue counts for {col}:\n", df[col].value_counts())
    except Exception as e:
        print(f"Error reading Excel: {e}")
else:
    print(f"File not found: {file_path}")
