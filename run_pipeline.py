from semantic_checker import build_and_run_checker
import os
from pathlib import Path

def run():
    base_dir = Path(__file__).parent
    db_path = base_dir / "data" / "processed" / "gold_database.csv"
    
    if not db_path.exists():
        print(f"Error: {db_path} not found. Please run rebuild_gold.py first.")
    else:
        build_and_run_checker()

if __name__ == "__main__":
    run()
