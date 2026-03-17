from semantic_checker import build_and_run_checker
import os

if __name__ == "__main__":
    # Ensure processed data exists
    if not os.path.exists("data/processed/gold_2019_mapped.csv"):
        print("Error: Historical mapping not found. Please run mapping scripts first.")
    else:
        build_and_run_checker()
