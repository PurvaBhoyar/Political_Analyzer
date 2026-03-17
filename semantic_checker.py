from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os

def build_and_run_checker():
    print("--- SEMANTIC HISTORICAL FACT-CHECKER (Tiered) ---")
    
    # 1. Load History
    db_path = "data/processed/gold_database.csv"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return
    history = pd.read_csv(db_path)
    
    # 2. Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Index History
    print(f"Indexing {len(history)} historical records...")
    history_embeddings = model.encode(history['original_text'].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # 4. Load 2024 Targets
    from nlp_engine import parser
    print("Extracting 2024 Manifesto promises...")
    df_2024 = parser.extract_manifesto_promises("data/raw/BJP-Election-english-2024.pdf")
    
    # 5. Batch Process 2024 Predictions
    print(f"Analyzing {len(df_2024)} promises...")
    query_embeddings = model.encode(df_2024['text'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    cos_scores = util.cos_sim(query_embeddings, history_embeddings)
    top_scores, top_indices = torch.max(cos_scores, dim=1)
    
    results = []
    label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}
    
    for i in range(len(df_2024)):
        best_idx = top_indices[i].item()
        score = top_scores[i].item()
        hist_match = history.iloc[best_idx]
        base_outcome = label_map[int(hist_match['label'])]
        
        # Apply Tiered Logic
        if score >= 0.70:
            forecast = f"High Confidence: {base_outcome}"
        elif score >= 0.60:
            forecast = f"Moderate Confidence: {base_outcome}"
        elif score >= 0.50:
            forecast = f"Low Confidence: {base_outcome}"
        else:
            forecast = "Indeterminate"
            
        results.append({
            "2024_Promise": df_2024.iloc[i]['text'],
            "Forecast": forecast,
            "Similarity_Score": round(score, 4),
            "Historical_Match": hist_match['original_text'],
            "Historical_Year": hist_match.get('year', 0)
        })
        
    df_results = pd.DataFrame(results)
    os.makedirs("data/output", exist_ok=True)
    df_results.to_csv("data/output/2024_fact_check_report.csv", index=False)
    print("\nTiered Report saved to: data/output/2024_fact_check_report.csv")

if __name__ == "__main__":
    build_and_run_checker()
