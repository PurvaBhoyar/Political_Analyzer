from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

OUTPUT_PATH = "data/output/project_report.pdf"
os.makedirs("data/output", exist_ok=True)

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

styles = getSampleStyleSheet()

# Custom Styles
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#16213e"), alignment=TA_CENTER, spaceAfter=20)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, textColor=colors.HexColor("#0f3460"), spaceBefore=16, spaceAfter=6, borderPad=4)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#533483"), spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=16, spaceAfter=6, alignment=TA_JUSTIFY)
BULLET = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=10, leading=15, leftIndent=16, spaceAfter=4)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, backColor=colors.HexColor("#f4f4f4"), borderPad=6, leading=13)
HIGHLIGHT = ParagraphStyle("Highlight", parent=styles["Normal"], fontSize=10, leading=15, backColor=colors.HexColor("#eef4ff"), borderPad=6, spaceAfter=6)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=4)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8, spaceBefore=4)

def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def body(text): return Paragraph(text, BODY)
def bullet(text): return Paragraph(f"• {text}", BULLET)
def code(text): return Paragraph(text, CODE)
def sp(h=8): return Spacer(1, h)
def highlight(text): return Paragraph(text, HIGHLIGHT)

def section_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

# ─── CONTENT STARTS ───────────────────────────────────────────────────────────

story = []

# ── Cover ──────────────────────────────────────────────────────────────────────
story += [
    sp(40),
    Paragraph("Political Promise Fact-Checker", TITLE),
    Paragraph("An NLP-Based Historical Analysis & Outcome Prediction System", SUBTITLE),
    hr(),
    sp(10),
    body("This document provides a complete technical walkthrough of the project — from raw data to final prediction — written for faculty presentation. Every concept is explained from first principles."),
    sp(20),
]

info_table = Table([
    ["Subject", "Natural Language Processing (NLP)"],
    ["Focus Party", "Bharatiya Janata Party (BJP)"],
    ["Data Coverage", "2014 Manifesto + Review | 2019 Manifesto + Review | 2024 Manifesto"],
    ["Core NLP Model", "all-MiniLM-L6-v2 (Sentence Transformers)"],
    ["LLM Reviewer", "LLaMA 3.3 70B via Groq API (secondary layer)"],
    ["API Framework", "FastAPI (Python)"],
    ["Output", "Verdict: Likely Fulfilled / Partially Fulfilled / Unlikely / Indeterminate"],
], colWidths=[5*cm, 11*cm])
info_table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#eef4ff"), colors.white]),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(info_table)
story.append(PageBreak())

# ── Section 1: What Does This Project Do ──────────────────────────────────────
story += [h1("1. What Does This Project Do?"), hr()]
story += [
    body("The Political Promise Fact-Checker is a system that takes any political promise (text) as input and predicts <b>how likely it is to be fulfilled</b>, based on 10 years of BJP's actual governance history."),
    sp(),
    highlight("Simple analogy: Imagine a student who always promises to study but never does. If they promise again, you'd say 'Unlikely to be fulfilled' because of their track record. This system does the same — but for political promises, using NLP."),
    sp(),
    body("The system answers questions like:"),
    bullet("Has BJP made this kind of promise before?"),
    bullet("When they made it before, did they actually deliver?"),
    bullet("Based on that history, what is the likely outcome this time?"),
]

story += [sp(12), h1("2. Data Sources — What Raw Data Do We Have?"), hr()]
story += [
    body("All data is stored in the <b>data/raw/</b> folder. We use three types of sources:"),
    sp(6),
]

data_table = section_table([
    ["File / Folder", "Type", "What It Contains", "Used For"],
    ["Manifesto_English.pdf", "PDF", "BJP 2014 Election Manifesto — all promises made before 2014 elections", "Source of 2014 promises"],
    ["2014 BJP Manifesto Review.pdf", "PDF", "Independent review of BJP's 2014-2019 term — what was done and what was not", "Labels for 2014 promises"],
    ["BJP-Election-english-2019.pdf", "PDF", "BJP 2019 Election Manifesto — all promises made before 2019 elections", "Reference only"],
    ["data/raw/bjp_2019/ (9 CSV files)", "CSVs", "Sector-wise report card of BJP 2019-2024 term. Sectors: Agriculture, Economy, Health, Infra, Security, etc.", "Labels for 2019 promises"],
    ["BJP-Election-english-2024.pdf", "PDF", "BJP 2024 Election Manifesto — the promises we want to PREDICT outcomes for", "Target for prediction"],
], col_widths=[3.8*cm, 1.8*cm, 7*cm, 3.2*cm])
story.append(data_table)

story += [
    sp(8),
    body("The 2019 CSV report cards look like this (one row = one promise):"),
    sp(4),
    section_table([
        ["PROMISE", "STATUS AND COMMENTS", "SECTOR"],
        ["Ensure LPG gas cylinder connection to all poor rural households", "Completed under Ujjwala Yojana — 8 crore connections given", "INFRA"],
        ["Double the Farmers' Income (2016-2022)", "Target not met. Farm income grew but did not double by 2022", "AGRICULTURE"],
    ], col_widths=[6*cm, 7.5*cm, 3*cm]),
]

story.append(PageBreak())

# ── Section 3: Pipeline ────────────────────────────────────────────────────────
story += [h1("3. Full Data Processing Pipeline"), hr()]
story += [
    body("The system works in a two-phase pipeline:"),
    sp(4),
    h2("Phase 1 — Build the Gold Database (run once, offline)"),
    body("This is done by running <b>rebuild_gold.py</b>. It creates a knowledge base of 524 historical promise-outcome pairs."),
    sp(4),
]

pipeline1 = section_table([
    ["Step", "What Happens", "Code File"],
    ["Step 1", "PDF Parser reads the 2014 Review PDF page-by-page, extracts tables row-by-row. Each row = one promise + one remark (outcome status)", "nlp_engine/parser.py → extract_review_table()"],
    ["Step 2", "CSV Reader reads all 9 sector CSVs from bjp_2019/ folder. Merges them into one combined dataframe with a 'sector' column", "nlp_engine/parser.py → extract_folder_data()"],
    ["Step 3", "Semantic Labeler converts each 'remark' text into a numeric label: 2 = Fulfilled, 1 = Partial, 0 = Unlikely. Uses NLP similarity, not keyword matching.", "nlp_engine/labeler.py → process_data()"],
    ["Step 4", "Both years are merged with a 'year' column (2014/2019) and saved as gold_database.csv — 524 total records", "rebuild_gold.py"],
], col_widths=[2*cm, 8.5*cm, 5.5*cm])
story.append(pipeline1)

story += [
    sp(10),
    h2("Phase 2 — Real-Time Prediction (runs on every API request)"),
    body("This is done by <b>main.py</b> (FastAPI server). When a user sends a promise text via POST /predict:"),
    sp(4),
]

pipeline2 = section_table([
    ["Step", "What Happens", "Technical Term"],
    ["Step 1", "The input promise text is converted into a 384-dimensional vector (a list of 384 numbers representing meaning)", "Text Encoding / Embedding"],
    ["Step 2", "This vector is compared against all 524 vectors in the Gold Database using Cosine Similarity", "Semantic Search"],
    ["Step 3", "Top 3 most similar historical records are retrieved with their similarity scores (0.0 to 1.0)", "Top-K Retrieval"],
    ["Step 4", "The label of the best match is used as the base prediction. Confidence tiers are applied (≥0.70, ≥0.60, ≥0.50)", "Threshold Classification"],
    ["Step 5 (Optional)", "The semantic output is sent to LLaMA 3.3 70B via Groq API for a political context review", "LLM Secondary Review"],
    ["Step 6", "Final verdict is assembled and returned as JSON", "API Response"],
], col_widths=[2*cm, 9*cm, 4.8*cm])
story.append(pipeline2)

story.append(PageBreak())

# ── Section 4: NLP Model ───────────────────────────────────────────────────────
story += [h1("4. The Core NLP Model — all-MiniLM-L6-v2"), hr()]
story += [
    body("This is the heart of the system. It is a <b>Sentence Transformer</b> model — a type of BERT-based neural network pre-trained on 1 billion sentence pairs."),
    sp(6),
    h2("What is a Sentence Transformer?"),
    body("Normal word-level models (like basic BERT) understand individual words. A Sentence Transformer understands the <b>meaning of an entire sentence</b>. It converts any sentence into a fixed-size vector of 384 numbers — called an <b>embedding</b>."),
    sp(4),
    highlight('Example: "We will build highways across India" → [0.23, -0.41, 0.89, ... 384 numbers]\n"Road infrastructure development across the country" → [0.21, -0.38, 0.91, ... 384 numbers]\nThese two vectors are very CLOSE → High similarity → Same topic'),
    sp(6),
    h2("Why this specific model?"),
]

model_table = section_table([
    ["Property", "Value", "Why It Matters"],
    ["Model Name", "sentence-transformers/all-MiniLM-L6-v2", "Open-source, no API key needed"],
    ["Architecture", "6-layer MiniLM (distilled BERT)", "Fast enough to run on CPU"],
    ["Embedding Dimension", "384", "Compact but highly accurate"],
    ["Training Data", "1 Billion sentence pairs", "Understands diverse English including political language"],
    ["Model Size", "~80 MB", "Downloads once, runs offline forever"],
    ["Inference Speed", "~25ms per sentence (CPU)", "Real-time API response"],
], col_widths=[4.5*cm, 6*cm, 5.5*cm])
story.append(model_table)

story += [
    sp(10),
    h2("What is Cosine Similarity?"),
    body("After encoding two sentences into vectors, we measure how 'close' they are using Cosine Similarity. The formula is:"),
    sp(4),
    code("Cosine Similarity = cos(θ) = (A · B) / (||A|| × ||B||)"),
    sp(4),
    body("The result is a number between 0 and 1:"),
]

cos_table = section_table([
    ["Score", "Meaning", "Example"],
    ["0.90 – 1.00", "Near-identical meaning (exact match)", "'Build Ram Temple' vs 'Construct Ram Mandir in Ayodhya'"],
    ["0.70 – 0.89", "Same topic, very similar promise", "'LPG to poor families' vs 'Gas connections to rural households'"],
    ["0.50 – 0.69", "Related topic, different angle", "'Financial inclusion' vs 'Zero-balance bank accounts'"],
    ["Below 0.50", "Unrelated or no good match", "No historical precedent found"],
], col_widths=[3*cm, 5.5*cm, 7.5*cm])
story.append(cos_table)

story.append(PageBreak())

# ── Section 5: Labeling ────────────────────────────────────────────────────────
story += [h1("5. How Are Promises Labeled? (Semantic Labeling)"), hr()]
story += [
    body("The remark column in each CSV says things like <i>'Completed under Ujjwala Yojana'</i> or <i>'Target not met, still pending'</i>. We need to convert these free-text remarks into 3 numeric labels:"),
    sp(4),
]

label_table = section_table([
    ["Label", "Numeric Value", "Meaning", "Example Remark Text"],
    ["Highly Likely / Fulfilled", "2", "Promise was completed or has strong delivery track record", "'Completed and operational', 'Successfully implemented'"],
    ["Partial", "1", "Some progress was made but target not fully achieved", "'Work in progress', 'Partially implemented', 'Ongoing'"],
    ["Unlikely", "0", "No progress, cancelled, or repeatedly failed", "'No progress', 'Stalled', 'Not achieved'"],
], col_widths=[3.5*cm, 2.5*cm, 5*cm, 5*cm])
story.append(label_table)

story += [
    sp(8),
    h2("How Does Semantic Labeling Work?"),
    body("Instead of using keyword matching (which is fragile), we use the NLP model itself to assign labels. We define 6 'anchor sentences' — 2 per label — and measure which anchor each remark is most similar to:"),
    sp(4),
    section_table([
        ["Label", "Anchor Sentences Used"],
        ["2 (Fulfilled)", '"Project completed and operational." | "Successfully implemented and achieved."'],
        ["1 (Partial)", '"Work in progress and ongoing." | "Partial implementation underway."'],
        ["0 (Unlikely)", '"No progress made, stalled." | "Cancelled or likely to be unfulfilled."'],
    ], col_widths=[2.5*cm, 13.5*cm]),
    sp(6),
    body("The remark gets the label of whichever anchor it's closest to in vector space. This is more robust than keyword matching because it understands <b>meaning</b>, not just words."),
]

story.append(PageBreak())

# ── Section 6: Gold Database ───────────────────────────────────────────────────
story += [h1("6. The Gold Database"), hr()]
story += [
    body("The Gold Database (<b>data/processed/gold_database.csv</b>) is the core knowledge base. It has 524 rows — each row is one historical promise with its actual outcome label."),
    sp(4),
    section_table([
        ["original_text", "label", "year", "sector"],
        ["Ensure LPG gas cylinder connection to all poor rural households", "2", "2019", "INFRA"],
        ["Double the Farmers' Income (2016-2022)", "0", "2019", "AGRICULTURE"],
        ["BJP reiterates its stand on the Article 370...", "1", "2014", "—"],
        ["We will launch Diamond Quadrilateral project of High Speed Train network", "2", "2014", "—"],
    ], col_widths=[7.5*cm, 1.5*cm, 1.5*cm, 5.5*cm]),
    sp(6),
    body("Total records: <b>524</b> — split roughly as:"),
    bullet("<b>~185 records</b> labeled 0 (Unlikely) — promises that failed"),
    bullet("<b>~30 records</b> labeled 1 (Partial) — partial delivery"),
    bullet("<b>~309 records</b> labeled 2 (Highly Likely / Fulfilled) — completed"),
    sp(4),
    highlight("Note: The label distribution is skewed toward '2' because BJP's review documents tend to report successes. The LLM layer provides a critical second opinion to compensate for this bias."),
]

story.append(PageBreak())

# ── Section 7: Confidence Thresholds ──────────────────────────────────────────
story += [h1("7. Confidence Tiers & Thresholds"), hr()]
story += [
    body("Once the best historical match is found, the similarity score determines the confidence tier of the prediction. These thresholds were set based on empirical testing:"),
    sp(6),
]

thresh_table = section_table([
    ["Similarity Score", "Confidence Tier", "What It Means", "Example"],
    ["≥ 0.70", "High Confidence", "Strong match — prediction is reliable", "Article 370 (0.96), Diamond Quadrilateral (0.947)"],
    ["0.60 – 0.69", "Moderate Confidence", "Good match — prediction is reasonable", "Article 370 2014 remark (0.61)"],
    ["0.50 – 0.59", "Low Confidence", "Weak match — treat with caution", "Financial inclusion promises (~0.55)"],
    ["< 0.50", "Indeterminate", "No relevant history found", "Very novel or vague promises"],
], col_widths=[3*cm, 3.5*cm, 5*cm, 5.5*cm])
story.append(thresh_table)

story += [
    sp(6),
    h2("Why 0.70 as High Confidence threshold?"),
    body("In practice, political promises use varied language. A 0.70 similarity in sentence embeddings represents semantically near-identical content — it reliably finds the 'same' promise made in a previous term. Below 0.70, there's enough linguistic distance that the match could be coincidental."),
]

story.append(PageBreak())

# ── Section 8: LLM Integration ─────────────────────────────────────────────────
story += [h1("8. LLM Integration — Groq + LLaMA 3.3 70B"), hr()]
story += [
    body("The LLM (Large Language Model) is a <b>secondary reviewer</b>, not the primary system. The NLP semantic output is the authoritative answer. The LLM adds political and real-world context on top."),
    sp(6),
    h2("Why Is the LLM Secondary?"),
    body("This is a key architectural decision. The NLP model is fully deterministic, auditable, and based on labeled historical data from actual BJP review reports. The LLM is probabilistic and based on general training data. For an academic NLP project, the primary output must come from the NLP pipeline."),
    sp(6),
    h2("What Does the LLM Do?"),
]

llm_table = section_table([
    ["LLM Role", "Description"],
    ["Independent Reviewer", "Analyzes the semantic output and gives its own verdict based on real-world political knowledge (e.g., knows Article 370 was abrogated in 2019)"],
    ["Discrepancy Detector", "When LLM disagrees with semantic output, a discrepancy flag is raised — useful for identifying mislabeled gold data"],
    ["Context Provider", "Adds specific scheme names, policy names, budget allocations, governance context that the NLP model cannot infer from text alone"],
    ["Repeat Promise Detector", "Flags if a 2024 promise is identical to a past manifesto promise (repeat_promise: true)"],
], col_widths=[4.5*cm, 11.5*cm])
story.append(llm_table)

story += [
    sp(8),
    h2("Model Used: LLaMA 3.3 70B via Groq"),
    body("We use Groq's free API tier with the LLaMA 3.3 70B model. Key parameters:"),
]

params_table = section_table([
    ["Parameter", "Value", "Reason"],
    ["model", "llama-3.3-70b-versatile", "70B parameters — best reasoning available for free"],
    ["temperature", "0.1", "Near-deterministic — same promise gives same verdict every time"],
    ["max_tokens", "600", "Enough for structured JSON response with full reasoning"],
    ["response_format", "json_object", "Forces valid parseable JSON — prevents hallucinated text"],
], col_widths=[3.5*cm, 5*cm, 7.5*cm])
story.append(params_table)

story.append(PageBreak())

# ── Section 9: Real Example Walkthrough ───────────────────────────────────────
story += [h1("9. Real Example Walkthrough — Article 370"), hr()]
story += [
    body('Input promise: <i>"We reiterate our position since the time of the Jan Sangh to the abrogation of Article 370."</i>'),
    sp(8),
]

walkthrough = section_table([
    ["Stage", "What Happened", "Result"],
    ["1. Encode", "Promise text → 384-D vector using MiniLM", "Vector: [0.12, -0.33, ...]"],
    ["2. Search", "Compared against 524 gold vectors using cosine similarity", "Best match found: 2014 record about Article 370"],
    ["3. Match", '"BJP reiterates its stand on Article 370..." from 2014, label=1 (Partial), similarity=0.6136', "Score: 0.61 → Moderate Confidence tier"],
    ["4. Semantic Output", "Label 1 = 'Partial'. Score 0.61 → Moderate Confidence", "Forecast: 'Moderate Confidence: Partial'"],
    ["5. LLM Review", "LLM knows Article 370 was ACTUALLY abrogated in Aug 2019. Disagrees with Partial label.", "LLM Verdict: 'Likely Fulfilled'"],
    ["6. Discrepancy Flag", "Semantic says Partial, LLM says Likely Fulfilled → discrepancy=true", "Discrepancy note shown in output"],
    ["7. Final Verdict", "Semantic is primary → Final = 'Partially Fulfilled'. LLM note added as context.", "final_verdict: Partially Fulfilled"],
], col_widths=[2.5*cm, 10*cm, 3.5*cm])
story.append(walkthrough)

story += [
    sp(8),
    highlight("Why does discrepancy happen here? The 2014 gold database labeled this as 'Partial' because in 2014, Article 370 had NOT yet been abrogated. But Article 370 WAS abrogated in 2019. The gold database reflects the state in 2014, not the final outcome. The LLM correctly identifies this gap — which is why the discrepancy flag is valuable for improving the data."),
]

story.append(PageBreak())

# ── Section 10: Architecture Summary ──────────────────────────────────────────
story += [h1("10. Architecture Summary"), hr()]
story += [
    body("Complete system architecture from raw data to API response:"),
    sp(8),
    section_table([
        ["Layer", "Component", "Technology", "Role"],
        ["Data Ingestion", "parser.py", "pdfplumber, pandas", "Extract text/tables from PDFs and CSVs"],
        ["Labeling", "labeler.py", "SentenceTransformers + Cosine Similarity", "Convert remark text → 0/1/2 label"],
        ["Knowledge Base", "gold_database.csv", "CSV (524 rows)", "Indexed historical promise-outcome pairs"],
        ["Embedding Engine", "SentenceTransformer", "all-MiniLM-L6-v2 (384-D)", "Convert text to meaning vectors"],
        ["Semantic Search", "util.cos_sim + torch.topk", "PyTorch", "Find top-3 most similar historical promises"],
        ["Threshold Logic", "resolve_final_verdict()", "Python", "Map similarity score to confidence tier"],
        ["LLM Reviewer", "llm_reviewer.py", "Groq API (LLaMA 3.3 70B)", "Political context review & discrepancy detection"],
        ["API", "main.py", "FastAPI + uvicorn", "REST endpoint exposing /predict and /health"],
    ], col_widths=[3.2*cm, 3.3*cm, 4.5*cm, 5*cm]),
]

story += [
    sp(14),
    h1("11. API Endpoints"),
    hr(),
    h2("POST /predict"),
    body("Accepts a JSON body with a promise text and returns the full analysis."),
    sp(4),
    code('Request:  {"text": "We will build 100 smart cities.", "use_llm": true}\n\nResponse fields:\n  final_verdict      → Primary NLP output\n  verdict_source     → "semantic" or "semantic_only"\n  discrepancy        → true/false — LLM agrees or disagrees\n  discrepancy_note   → Explanation if discrepancy exists\n  semantic_analysis  → forecast, confidence score, reasoning\n  llm_review         → LLM verdict, reasoning, key_factors\n  historical_evidence → Top 3 matched historical records'),
    sp(10),
    h2("GET /health"),
    code("Returns: { status, history_loaded, history_size, llm_ready }"),
]

story.append(PageBreak())

# ── Section 12: Accuracy & Limitations ────────────────────────────────────────
story += [h1("12. Accuracy, Limitations & Future Scope"), hr()]
story += [
    h2("Strengths"),
    bullet("No dependency on expensive GPU — runs entirely on CPU"),
    bullet("No paid API required for core NLP — MiniLM is free and local"),
    bullet("Explainable output — every prediction comes with the actual historical evidence used"),
    bullet("LLM discrepancy flag helps identify where gold database labels may be wrong"),
    bullet("Sub-second inference speed (25ms for embedding, <100ms total without LLM)"),
    sp(6),
    h2("Known Limitations"),
    bullet("Gold Database has label drift: 2014 labels reflect status-in-2014, not final-2019 outcomes (e.g., Article 370 marked Partial in 2014 data but was fulfilled in 2019)"),
    bullet("MiniLM was trained on general English — Indian political terminology and Hindi transliterations may reduce similarity scores"),
    bullet("Label distribution is imbalanced: ~59% label=2, which biases predictions toward 'Highly Likely'"),
    bullet("2024 predictions are forecasts only — actual outcomes unknown until 2029"),
    sp(6),
    h2("Future Scope"),
    bullet("Add Hindi/multilingual support using IndicBERT or MuRIL for regional language manifesto analysis"),
    bullet("Expand gold database to include state-level manifesto data (UP, Maharashtra, etc.)"),
    bullet("Add Named Entity Recognition (NER) to extract specific entities — schemes, amounts, deadlines — from promises"),
    bullet("Automatically update gold database after each election term using web scraping"),
    bullet("Add confidence calibration layer to fix the label imbalance bias"),
]

story.append(PageBreak())

# ── Section 13: How to Run ─────────────────────────────────────────────────────
story += [
    h1("13. How to Run the Project"),
    hr(),
    h2("Step 1 — Set up environment"),
    code("python -m venv venv\nvenv\\Scripts\\activate\npip install -r requirements.txt"),
    sp(8),
    h2("Step 2 — Rebuild Gold Database (one-time)"),
    code("python rebuild_gold.py\n# Output: data/processed/gold_database.csv (524 records)"),
    sp(8),
    h2("Step 3 — Set Groq API Key (for LLM layer)"),
    code("$env:GROQ_API_KEY = \"gsk_your_key_here\"   # PowerShell\n# Free key from: https://console.groq.com"),
    sp(8),
    h2("Step 4 — Start the API Server"),
    code("uvicorn main:app --reload\n# Server runs at http://localhost:8000"),
    sp(8),
    h2("Step 5 — Test a Promise"),
    code('curl -X POST http://localhost:8000/predict\n     -H "Content-Type: application/json"\n     -d {"text": "We will double farmers income by 2022.", "use_llm": true}'),
    sp(8),
    h2("Step 6 — Run Batch Analysis of 2024 Manifesto"),
    code("python semantic_checker.py\n# Output: data/output/2024_fact_check_report.csv"),
]

# Build PDF
doc.build(story)
print(f"PDF generated at: {OUTPUT_PATH}")
