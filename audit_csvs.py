import pandas as pd
import os

folder = r'c:\Users\DELL\political_analyzer\data\raw\bjp_2019'
if os.path.exists(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    for f in files:
        file_path = os.path.join(folder, f)
        try:
            # Use cp1252 to handle common Windows-Excel CSV symbols
            df = pd.read_csv(file_path, encoding='cp1252', on_bad_lines='skip')
            print(f"File: {f}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {df.columns.tolist()}")
            print("-" * 30)
        except Exception as e:
            print(f"Error reading {f}: {e}")
else:
    print(f"Folder not found: {folder}")
