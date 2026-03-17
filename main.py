from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os

app = FastAPI()

# 1. Load the Semantic Engine
print("Loading Semantic Engine...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load and Index History
history_path = "data/processed/gold_database.csv"

if os.path.exists(history_path):
    history = pd.read_csv(history_path)
    print(f"Indexing {len(history)} historical records...")
    history_embeddings = model.encode(history['original_text'].astype(str).tolist(), convert_to_tensor=True)
else:
    history = None
    history_embeddings = None

class PromiseInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_outcome(input_data: PromiseInput):
    if history is None:
        return {"error": "History database not found. Run rebuild_gold.py first."}
    
    # 3. Semantic Search
    query_embedding = model.encode(input_data.text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, history_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)
    
    matches = []
    label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}
    
    for i in range(len(top_results.indices)):
        idx = top_results.indices[i].item()
        score = top_results.values[i].item()
        row = history.iloc[idx]
        matches.append({
            "historical_text": row['original_text'],
            "outcome": label_map[int(row['label'])],
            "similarity": round(float(score), 4),
            "year": int(row['year']) if 'year' in row else 0
        })

    # 4. Multi-Tiered Threshold Logic
    top_score = matches[0]['similarity']
    base_outcome = matches[0]['outcome']
    
    if top_score >= 0.70:
        forecast = f"High Confidence: {base_outcome}"
        reasoning = f"Strong historical match found from {matches[0]['year']}."
    elif top_score >= 0.60:
        forecast = f"Moderate Confidence: {base_outcome}"
        reasoning = f"Moderate similarity to {matches[0]['year']} historical data."
    elif top_score >= 0.50:
        forecast = f"Low Confidence: {base_outcome}"
        reasoning = "Weak historical match; treat result with caution."
    else:
        forecast = "Indeterminate"
        reasoning = "No relevant historical precedent found below 0.50 similarity."

    return {
        "promise": input_data.text,
        "forecast": forecast,
        "confidence": top_score,
        "reasoning": reasoning,
        "historical_evidence": matches
    }
