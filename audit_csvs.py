import pandas as pd
import os
from pathlib import Path

# Use relative paths for portability
base_dir = Path(__file__).parent
folder = base_dir / 'data' / 'raw' / 'bjp_2019'

if folder.exists():
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    for f in files:
        file_path = folder / f
        try:
            # Use cp1252 to handle common Windows-Excel CSV symbols
            df = pd.read_csv(str(file_path), encoding='cp1252', on_bad_lines='skip')
            print(f"File: {f}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {df.columns.tolist()}")
            print("-" * 30)
        except Exception as e:
            print(f"Error reading {f}: {e}")
else:
    print(f"Folder not found: {folder}")
