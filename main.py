from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os

app = FastAPI(title="Political Promise Fact-Checker", version="2.0")

print("Loading Semantic Engine...")
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    use_llm: bool = True

label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}

VERDICT_RANK = {
    "Likely Fulfilled": 3,
    "Partially Fulfilled": 2,
    "Unlikely to be Fulfilled": 1,
    "Cannot Determine": 0,
    "Unavailable": -1
}

SEMANTIC_TO_VERDICT = {
    "Highly Likely": "Likely Fulfilled",
    "Partial": "Partially Fulfilled",
    "Unlikely": "Unlikely to be Fulfilled",
}

def resolve_final_verdict(semantic_base: str, llm_result: dict, top_score: float) -> dict:
    """
    Semantic system is the primary verdict.
    LLM is a secondary reviewer — its disagreement is flagged but does NOT override.
    """
    semantic_verdict = SEMANTIC_TO_VERDICT.get(semantic_base, "Cannot Determine")
    llm_verdict = llm_result.get("llm_verdict", "Unavailable") if llm_result else "Unavailable"

    discrepancy = (
        llm_verdict not in ("Unavailable", "Cannot Determine")
        and llm_verdict != semantic_verdict
    )

    return {
        "final_verdict": semantic_verdict,
        "verdict_source": "semantic",
        "discrepancy": discrepancy,
        "discrepancy_note": (
            f"LLM reviewer disagrees: '{llm_verdict}'. "
            f"Consider this as additional context — semantic NLP is the primary output."
            if discrepancy else None
        )
    }


@app.post("/predict")
async def predict_outcome(input_data: PromiseInput):
    if history is None:
        return {"error": "History database not found. Run rebuild_gold.py first."}

    query_embedding = model.encode(input_data.text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, history_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)

    matches = []
    for i in range(len(top_results.indices)):
        idx = top_results.indices[i].item()
        score = top_results.values[i].item()
        row = history.iloc[idx]
        matches.append({
            "historical_text": str(row['original_text']),
            "outcome": label_map[int(row['label'])],
            "similarity": round(float(score), 4),
            "year": int(row['year']) if 'year' in row else 0
        })

    top_score = matches[0]['similarity']
    base_outcome = matches[0]['outcome']

    if top_score >= 0.70:
        semantic_forecast = f"High Confidence: {base_outcome}"
        semantic_reasoning = f"Strong historical match found from {matches[0]['year']}."
    elif top_score >= 0.60:
        semantic_forecast = f"Moderate Confidence: {base_outcome}"
        semantic_reasoning = f"Moderate similarity to {matches[0]['year']} historical data."
    elif top_score >= 0.50:
        semantic_forecast = f"Low Confidence: {base_outcome}"
        semantic_reasoning = "Weak historical match; treat result with caution."
    else:
        semantic_forecast = "Indeterminate"
        semantic_reasoning = "No relevant historical precedent found below 0.50 similarity."

    llm_result = None
    if input_data.use_llm:
        from nlp_engine.llm_reviewer import review_promise
        llm_result = review_promise(
            promise=input_data.text,
            semantic_forecast=semantic_forecast,
            confidence=top_score,
            historical_evidence=matches
        )

    verdict_info = resolve_final_verdict(base_outcome, llm_result, top_score)

    return {
        "promise": input_data.text,
        "final_verdict": verdict_info["final_verdict"],
        "verdict_source": verdict_info["verdict_source"],
        "discrepancy": verdict_info["discrepancy"],
        "discrepancy_note": verdict_info.get("discrepancy_note"),
        "semantic_analysis": {
            "forecast": semantic_forecast,
            "confidence": top_score,
            "reasoning": semantic_reasoning,
        },
        "llm_review": llm_result,
        "historical_evidence": matches
    }


@app.get("/health")
async def health():
    groq_configured = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "ok",
        "history_loaded": history is not None,
        "history_size": len(history) if history is not None else 0,
        "llm_ready": groq_configured
    }
