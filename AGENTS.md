# Repository Guidelines

## Project Structure & Module Organization
The **Political Analyzer** is a semantic fact-checking project that predicts the outcome of future political promises based on 10 years of historical performance.

- **`nlp_engine/`**: Core logic for the pipeline.
  - `parser.py`: Extracts promises from PDF manifestos and CSV review folders.
  - `labeler.py`: Categorizes historical outcomes (Unlikely, Partial, Highly Likely).
- **`semantic_checker.py`**: The core "Fact-Checker" engine. It uses **Sentence-Transformers (MiniLM)** to map 2024 promises to the most similar historical entries from 2014/2019.
- **`data/`**: 
  - `raw/`: Raw PDFs and CSV folders.
  - `processed/`: Mapped "Gold" datasets linking manifestos to their real-world reviews.
  - `output/`: Generated reports (e.g., `2024_fact_check_report.csv`).
- **`main.py`**: FastAPI application for real-time semantic prediction.

## Build, Test, and Development Commands
Ensure a Python environment is active and `requirements.txt` is installed (plus `sentence-transformers` and `thefuzz`).

- **Run Fact-Checker**: `python run_pipeline.py` (Runs the semantic analyzer on the 2024 Manifesto)
- **Start API server**: `uvicorn main:app --reload`
- **Check Results**: Open `data/output/2024_fact_check_report.csv` to see the historical proof for every prediction.

## Coding Style & Naming Conventions
- **NLP**: Uses **Semantic Similarity (RAG-style)**. Instead of a standard classifier, it retrieves the closest historical fact and inherits its outcome.
- **Library**: `sentence-transformers/all-MiniLM-L6-v2`.
