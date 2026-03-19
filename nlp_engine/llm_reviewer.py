import os
import json
from groq import Groq

_client = None

SYSTEM_PROMPT = """You are a senior Indian political analyst and independent fact-checker with deep expertise in BJP governance from 2014 to 2024. You have comprehensive knowledge of what actually happened in India during this period — which schemes were launched, which succeeded, which failed, and which were quietly dropped.

Your task is to analyze political promises and give an INDEPENDENT verdict. You will be given:
- A BJP manifesto promise
- A semantic similarity system's forecast (based on text matching, not real outcomes)
- Historical promise-outcome pairs that were matched

CRITICAL RULES YOU MUST FOLLOW:
1. Use your OWN knowledge of real-world outcomes as the PRIMARY basis for your verdict. The semantic system only matches text — it does not know what actually happened.
2. If similarity >= 0.85, treat it as the SAME promise repeated from a previous election. Your first question must be: "Was this actually delivered in the previous term?"
3. DO NOT simply agree with the semantic system. Be an independent critical analyst. The semantic system frequently labels promises as 'Highly Likely' when actual delivery was poor or partial.
4. Reference specific schemes, policies, budget allocations, or governance outcomes from your knowledge (e.g., Ujjwala Yojana, Jan Dhan, Smart Cities Mission, Diamond Quadrilateral, doubling farmer income, bullet trains, etc.)
5. Be precise and critical. Overpromising is common in Indian politics — your job is to cut through it.

VERDICT OPTIONS:
- "Likely Fulfilled": Strong historical delivery + clear political will + implementation evidence
- "Partially Fulfilled": Some progress but significant gaps, delays, or missed targets
- "Unlikely to be Fulfilled": Pattern of repeated promise without delivery, structural barriers, or track record of failure
- "Cannot Determine": Genuinely insufficient data or too new to assess"""


def build_prompt(promise: str, semantic_forecast: str, confidence: float, historical_evidence: list) -> str:
    similarity_warning = ""
    if historical_evidence and historical_evidence[0].get('similarity', 0) >= 0.85:
        similarity_warning = (
            f"\n⚠️ HIGH SIMILARITY ALERT ({round(historical_evidence[0]['similarity'] * 100, 1)}%): "
            f"This promise is almost IDENTICAL to one made in {historical_evidence[0].get('year', 'a previous election')}. "
            f"You MUST assess whether that original promise was actually delivered before giving your verdict.\n"
        )

    evidence_block = ""
    for i, e in enumerate(historical_evidence[:3], 1):
        evidence_block += (
            f"\n  [{i}] Year: {e.get('year', 'N/A')} | Semantic Label: {e.get('outcome', 'N/A')} "
            f"| Similarity: {round(e.get('similarity', 0) * 100, 1)}%"
            f"\n      Text: \"{str(e.get('historical_text', ''))[:250]}\"\n"
        )

    return f"""Analyze the following BJP manifesto promise:

PROMISE:
"{promise}"
{similarity_warning}
SEMANTIC SYSTEM OUTPUT (text-matching only — not real outcomes):
- Forecast: {semantic_forecast}
- Top Confidence: {round(confidence * 100, 1)}%

MATCHED HISTORICAL RECORDS:
{evidence_block}
Based on your knowledge of what actually happened in India, give your independent verdict.

Respond ONLY in this exact JSON format:
{{
  "llm_verdict": "<Likely Fulfilled | Partially Fulfilled | Unlikely to be Fulfilled | Cannot Determine>",
  "llm_reasoning": "<2-3 sentences using specific real-world evidence, scheme names, or outcomes you know about>",
  "key_factors": ["<specific factor with real context>", "<specific factor with real context>"],
  "repeat_promise": <true | false>
}}"""


def review_promise(promise: str, semantic_forecast: str, confidence: float, historical_evidence: list) -> dict:
    try:
        client = get_client()
        prompt = build_prompt(promise, semantic_forecast, confidence, historical_evidence)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        return json.loads(raw)

    except ValueError as e:
        return {"error": str(e), "llm_verdict": "Unavailable", "llm_reasoning": "GROQ_API_KEY not configured."}
    except Exception as e:
        return {"error": str(e), "llm_verdict": "Unavailable", "llm_reasoning": "LLM review failed."}


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client
