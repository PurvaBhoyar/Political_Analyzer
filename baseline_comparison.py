import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Set seeds for reproducibility
np.random.seed(42)

def run_comparison():
    print("--- RUNNING BASELINE COMPARISON (Section 6i) ---")
    
    # Load Gold Database
    df = pd.read_csv('data/processed/gold_database.csv')
    X = df['original_text'].astype(str)
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 1. Baseline: TF-IDF + Random Forest ---
    print("Training Baseline: TF-IDF + Random Forest...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_tfidf, y_train)
    rf_preds = rf.predict(X_test_tfidf)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')

    # --- 2. Proposed: Semantic RAG (Zero-Shot/Few-Shot style) ---
    # In our project, we don't 'train' the RAG, we use it for retrieval.
    # To 'evaluate' it as a classifier, we use it to predict labels for the test set.
    print("Evaluating Proposed: Semantic RAG (MiniLM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Anchors used in labeler.py
    anchors = {
        2: ["Project completed and operational.", "Successfully implemented and achieved."],
        1: ["Work in progress and ongoing.", "Partial implementation underway."],
        0: ["No progress made, stalled.", "Cancelled or likely to be unfulfilled."]
    }
    
    anchor_texts = []
    anchor_labels = []
    for label, texts in anchors.items():
        for text in texts:
            anchor_texts.append(text)
            anchor_labels.append(label)
    
    anchor_embs = model.encode(anchor_texts, convert_to_tensor=True)
    test_embs = model.encode(X_test.tolist(), convert_to_tensor=True)
    
    # Calculate similarity to anchors
    cos_scores = util.cos_sim(test_embs, anchor_embs)
    best_anchor_indices = torch.argmax(cos_scores, dim=1).tolist()
    rag_preds = [anchor_labels[idx] for idx in best_anchor_indices]

    rag_acc = accuracy_score(y_test, rag_preds)
    rag_f1 = f1_score(y_test, rag_preds, average='weighted')

    # --- 3. Results Comparison ---
    results = pd.DataFrame({
        "Model": ["Baseline (TF-IDF + RF)", "Proposed (Semantic RAG)"],
        "Accuracy": [rf_acc, rag_acc],
        "F1-Score": [rf_f1, rag_f1]
    })

    print("\nComparison Results:")
    print(results.to_string(index=False))

    # Save to CSV
    os.makedirs('data/output', exist_ok=True)
    results.to_csv('data/output/baseline_comparison.csv', index=False)

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.35

    plt.bar(x - width/2, results['Accuracy'], width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, results['F1-Score'], width, label='F1-Score', color='salmon')

    plt.title('Baseline vs Proposed Model Performance', fontsize=14)
    plt.xticks(x, results['Model'])
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('data/output/baseline_comparison_plot.png')
    print("\nVisualizations saved to data/output/")
    plt.show()

if __name__ == "__main__":
    run_comparison()
