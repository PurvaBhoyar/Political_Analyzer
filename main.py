from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

app = FastAPI(title="Political Promise Fact-Checker", version="2.0")

# Setup Pathlib
BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "data" / "processed" / "gold_database.csv"
MODEL_SAVE_PATH = BASE_DIR / "models" / "paraphrase-multilingual-MiniLM-L12-v2-local"
FINE_TUNED_MODEL_PATH = BASE_DIR / "models" / "finetuned-politicheck-multilingual"

# Singleton-like loading to prevent multiple loads
model = None
history = None
history_embeddings = None

def get_model():
    global model
    if model is None:
        # Priority 1: Fine-tuned Model
        if FINE_TUNED_MODEL_PATH.exists():
            print(f"Loading FINE-TUNED Semantic Engine: {FINE_TUNED_MODEL_PATH}")
            model = SentenceTransformer(str(FINE_TUNED_MODEL_PATH))
        # Priority 2: Local Cache of Base Model
        elif MODEL_SAVE_PATH.exists():
            print(f"Loading Base Semantic Engine from LOCAL cache: {MODEL_SAVE_PATH}")
            model = SentenceTransformer(str(MODEL_SAVE_PATH))
        # Priority 3: Download from HuggingFace
        else:
            print("Downloading and Caching Base Semantic Engine...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
            model.save(str(MODEL_SAVE_PATH))
            print(f"Model saved to {MODEL_SAVE_PATH}")
    return model

@app.on_event("startup")
async def startup_event():
    global history, history_embeddings
    get_model()
    
    if HISTORY_PATH.exists():
        history = pd.read_csv(str(HISTORY_PATH))
        print(f"Indexing {len(history)} historical records...")
        # Pre-index for performance
        history_embeddings = model.encode(history['original_text'].astype(str).tolist(), convert_to_tensor=True)
        print("Indexing Complete.")
    else:
        print("WARNING: History database not found.")

class PromiseInput(BaseModel):
    text: str
    use_llm: bool = True

label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}

SEMANTIC_TO_VERDICT = {
    "Highly Likely": "Likely Fulfilled",
    "Partial": "Partially Fulfilled",
    "Unlikely": "Unlikely to be Fulfilled",
    "Low Confidence Highly Likely": "Likely Fulfilled (Low Confidence)",
    "Low Confidence Partial": "Partially Fulfilled (Low Confidence)",
    "Low Confidence Unlikely": "Unlikely to be Fulfilled (Low Confidence)",
    "Indeterminate": "Cannot Determine",
}

def resolve_final_verdict(semantic_base: str, llm_result: dict, top_score: float) -> dict:
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

    # Offload CPU-bound task (model.encode) to a thread
    query_embedding = await asyncio.to_thread(model.encode, input_data.text, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, history_embeddings)[0]
    # Diversity-aware retrieval: Get more candidates than needed
    top_candidates = torch.topk(cos_scores, k=min(10, len(history)))

    matches = []
    seen_labels = set()
    
    # Always include the absolute best match
    best_idx = top_candidates.indices[0].item()
    best_score = float(top_candidates.values[0].item())
    best_row = history.iloc[best_idx]
    best_label_str = label_map[int(best_row['label'])]
    
    matches.append({
        "historical_text": str(best_row['original_text']),
        "outcome": best_label_str,
        "similarity": round(best_score, 4),
        "year": int(best_row['year']) if 'year' in best_row else 0
    })
    seen_labels.add(best_label_str)

    # Diversity-aware retrieval for subsequent matches
    # Prefer matches with DIFFERENT labels if they are within a reasonable similarity range
    for i in range(1, len(top_candidates.indices)):
        if len(matches) >= 3:
            break
            
        idx = top_candidates.indices[i].item()
        score = float(top_candidates.values[i].item())
        row = history.iloc[idx]
        label_str = label_map[int(row['label'])]
        
        # Heuristic: If label is new AND score is within 10% of best, OR we just need to fill slots
        if (label_str not in seen_labels and (best_score - score) < 0.10) or (len(matches) < 3 and i > 5):
            matches.append({
                "historical_text": str(row['original_text']),
                "outcome": label_str,
                "similarity": round(score, 4),
                "year": int(row['year']) if 'year' in row else 0
            })
            seen_labels.add(label_str)

    # If we still don't have 3 matches (rare), fill with next best
    if len(matches) < 3:
        for i in range(1, len(top_candidates.indices)):
            if len(matches) >= 3: break
            idx = top_candidates.indices[i].item()
            score = float(top_candidates.values[i].item())
            row = history.iloc[idx]
            label_str = label_map[int(row['label'])]
            if not any(m['historical_text'] == str(row['original_text']) for m in matches):
                matches.append({
                    "historical_text": str(row['original_text']),
                    "outcome": label_str,
                    "similarity": round(score, 4),
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
        from nlp_engine.llm_reviewer import review_promise_async
        try:
            # Set a timeout for the LLM review to prevent hanging
            llm_result = await asyncio.wait_for(
                review_promise_async(
                    promise=input_data.text,
                    semantic_forecast=semantic_forecast,
                    confidence=top_score,
                    historical_evidence=matches
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            print("LLM Review timed out.")
            llm_result = {"error": "Timeout", "llm_verdict": "Unavailable", "llm_reasoning": "Skeptic layer timed out."}
        except Exception as e:
            print(f"LLM Review error: {e}")
            llm_result = {"error": str(e), "llm_verdict": "Unavailable", "llm_reasoning": f"Skeptic layer error: {e}"}

    # Determine verdict_base based on top_score thresholds
    if top_score >= 0.65:
        verdict_base = base_outcome
    elif top_score >= 0.50:
        verdict_base = "Low Confidence " + base_outcome
    else:
        verdict_base = "Indeterminate"

    verdict_info = resolve_final_verdict(verdict_base, llm_result, top_score)

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
