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
BASE_MODEL_SAVE_PATH = BASE_DIR / "models" / "paraphrase-multilingual-MiniLM-L12-v2-local"
FINE_TUNED_MODEL_PATH = BASE_DIR / "models" / "finetuned-politicheck-multilingual"

# Singleton-like loading to prevent multiple loads
model_base = None
model_finetuned = None
history = None
history_embeddings_base = None
history_embeddings_finetuned = None

def get_models():
    global model_base, model_finetuned
    if model_base is None:
        if BASE_MODEL_SAVE_PATH.exists():
            print(f"Loading Base Semantic Engine from LOCAL cache: {BASE_MODEL_SAVE_PATH}")
            model_base = SentenceTransformer(str(BASE_MODEL_SAVE_PATH))
        else:
            print("Downloading and Caching Semantic Engine...")
            model_base = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            os.makedirs(BASE_MODEL_SAVE_PATH.parent, exist_ok=True)
            model_base.save(str(BASE_MODEL_SAVE_PATH))
            print(f"Model saved to {BASE_MODEL_SAVE_PATH}")
            
    if model_finetuned is None:
        if FINE_TUNED_MODEL_PATH.exists():
            print(f"Loading FINE-TUNED Semantic Engine: {FINE_TUNED_MODEL_PATH}")
            model_finetuned = SentenceTransformer(str(FINE_TUNED_MODEL_PATH))
        else:
            model_finetuned = model_base

@app.on_event("startup")
async def startup_event():
    global history, history_embeddings_base, history_embeddings_finetuned
    get_models()
    
    if HISTORY_PATH.exists():
        history = pd.read_csv(str(HISTORY_PATH))
        print(f"Indexing {len(history)} historical records into dual vector spaces...")
        text_list = history['original_text'].astype(str).tolist()
        history_embeddings_base = model_base.encode(text_list, convert_to_tensor=True)
        history_embeddings_finetuned = model_finetuned.encode(text_list, convert_to_tensor=True)
        print("Dual Indexing Complete.")
    else:
        print("WARNING: History database not found.")
        
import re

def contains_hinglish(text):
    """Detect if text contains romanized Hindi words mixed with English."""
    # Pure Devanagari is already handled well by the multilingual model
    if re.search(r'[\u0900-\u097F]', text):
        return False
    # Common Hinglish patterns: words with aa, ee, oo, waa, dhi, etc.
    hinglish_patterns = [
        r'\b\w*aad\w*\b', r'\b\w*waad\w*\b', r'\b\w*aat\w*\b',
        r'\b\w*dhi\b', r'\bham\b', r'\bhum\b', r'\bsab\b',
        r'\bkaam\b', r'\bdesh\b', r'\bsarkar\b', r'\byojana\b',
        r'\bvikas\b', r'\bkisan\b', r'\bgareeb\b', r'\brozgar\b',
        r'\bsuraksha\b', r'\bswachh\b', r'\bshiksha\b',
    ]
    for pattern in hinglish_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    # Fallback: use langdetect
    try:
        from langdetect import detect
        lang = detect(text)
        if lang == 'hi':
            return True
    except:
        pass
    return False

async def normalize_hinglish(text):
    """Use Groq LLM to convert Hinglish text to clean English."""
    try:
        from groq import AsyncGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return text
        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a translator. Convert the following text to pure formal English. If it contains any Hindi words written in Roman script (Hinglish), translate those words to English. Keep the sentence structure and meaning identical. Output ONLY the translated sentence, nothing else."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        normalized = response.choices[0].message.content.strip()
        # Remove quotes if the LLM wrapped the output
        normalized = normalized.strip('"').strip("'")
        print(f"[Hinglish Normalizer] '{text}' -> '{normalized}'")
        return normalized
    except Exception as e:
        print(f"[Hinglish Normalizer] Failed: {e}")
        return text

class PromiseInput(BaseModel):
    text: str
    use_llm: bool = True

label_map = {0: "Unlikely", 1: "Partial", 2: "Highly Likely"}

SEMANTIC_TO_VERDICT = {
    "Highly Likely": "Likely Fulfilled",
    "Partial": "Partially Fulfilled",
    "Unlikely": "Unlikely to be Fulfilled",
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

    # Step 1: Normalize Hinglish to English via LLM if detected
    encoding_text = input_data.text
    hinglish_detected = contains_hinglish(input_data.text)
    if hinglish_detected:
        encoding_text = await normalize_hinglish(input_data.text)

    # Step 2: Dual-Model Ensembling (Highest score wins implicitly)
    query_embedding_base = await asyncio.to_thread(model_base.encode, encoding_text, convert_to_tensor=True)
    query_embedding_finetuned = await asyncio.to_thread(model_finetuned.encode, encoding_text, convert_to_tensor=True)
    
    cos_scores_base = util.cos_sim(query_embedding_base, history_embeddings_base)[0]
    cos_scores_finetuned = util.cos_sim(query_embedding_finetuned, history_embeddings_finetuned)[0]
    
    top_results_base = torch.topk(cos_scores_base, k=3)
    top_results_finetuned = torch.topk(cos_scores_finetuned, k=3)
    
    top_score_base_val = top_results_base.values[0].item()
    top_score_finetuned_val = top_results_finetuned.values[0].item()
    
    # Implicit routing: return data from the model with the highest similarity score
    used_base_model = top_score_base_val > top_score_finetuned_val
    top_results = top_results_base if used_base_model else top_results_finetuned

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
        from nlp_engine.llm_reviewer import review_promise_async
        llm_result = await review_promise_async(
            promise=input_data.text,
            semantic_forecast=semantic_forecast,
            confidence=top_score,
            historical_evidence=matches
        )

    # Use base_outcome only if confidence is above 0.50; otherwise use "Indeterminate"
    verdict_base = base_outcome if top_score >= 0.50 else "Indeterminate"
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
