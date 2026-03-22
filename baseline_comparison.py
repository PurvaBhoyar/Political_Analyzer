import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import label_binarize
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Set seeds for reproducibility
np.random.seed(42)

def run_comparison():
    print("--- RUNNING ADVANCED BASELINE COMPARISON (Section 6c, 6f, 6i) ---")
    
    # Load Gold Database
    df = pd.read_csv('data/processed/gold_database.csv')
    X = df['original_text'].astype(str)
    y = df['label']
    n_classes = len(np.unique(y))

    # Split data (Section 5k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 1. Baseline: TF-IDF + Random Forest ---
    print("Training Baseline: TF-IDF + Random Forest...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_tfidf, y_train)
    rf_preds = rf.predict(X_test_tfidf)
    rf_probs = rf.predict_proba(X_test_tfidf)

    # --- 2. Proposed: Semantic RAG (MiniLM) ---
    print("Evaluating Proposed: Semantic RAG (MiniLM)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
    
    cos_scores = util.cos_sim(test_embs, anchor_embs)
    
    # Get probabilities/scores for RAG (mapping max similarity per class)
    rag_probs = []
    for i in range(len(X_test)):
        class_scores = []
        for label in [0, 1, 2]:
            label_indices = [idx for idx, l in enumerate(anchor_labels) if l == label]
            class_scores.append(cos_scores[i][label_indices].max().item())
        rag_probs.append(class_scores)
    rag_probs = np.array(rag_probs)
    # Softmax to turn similarities into pseudo-probabilities for ROC/PR
    rag_probs = np.exp(rag_probs) / np.sum(np.exp(rag_probs), axis=1)[:, None]
    rag_preds = np.argmax(rag_probs, axis=1)

    # --- 3. Performance Metrics (Section 6c) ---
    def get_metrics(y_true, y_pred, name):
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted'),
            "Recall": recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": f1_score(y_true, y_pred, average='weighted')
        }

    rf_metrics = get_metrics(y_test, rf_preds, "Baseline (RF)")
    rag_metrics = get_metrics(y_test, rag_preds, "Proposed (RAG)")
    
    results = pd.DataFrame([rf_metrics, rag_metrics])
    print("\nComparison Metrics:")
    print(results.to_string(index=False))
    
    os.makedirs('data/output/metrics', exist_ok=True)
    results.to_csv('data/output/metrics/performance_comparison.csv', index=False)

    # --- 4. Required Graphs (Section 6f) ---
    
    # A. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix: Baseline (RF)')
    sns.heatmap(confusion_matrix(y_test, rag_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Confusion Matrix: Proposed (RAG)')
    plt.savefig('data/output/metrics/confusion_matrices.png')
    plt.close()

    # B. ROC-AUC Curves (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), ['blue', 'red', 'green']):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rag_probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC-AUC Curves: Proposed (RAG) System')
    plt.legend(loc="lower right")
    plt.savefig('data/output/metrics/roc_auc_curves.png')
    plt.close()

    # C. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), ['blue', 'red', 'green']):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], rag_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f'Class {i}')
    plt.title('Precision-Recall Curves: Proposed (RAG) System')
    plt.legend(loc="lower left")
    plt.savefig('data/output/metrics/precision_recall_curves.png')
    plt.close()

    print("\nAll performance graphs (Confusion Matrix, ROC-AUC, PR Curves) saved to data/output/metrics/")

if __name__ == "__main__":
    run_comparison()
