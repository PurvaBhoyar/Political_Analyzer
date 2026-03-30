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

def plot_rf_learning_curve(X_train_tfidf, y_train):
    print("Generating Random Forest Learning Curve (Prompt 4)...")
    n_estimators_range = range(10, 110, 10)
    oob_errors = []

    for n in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, class_weight='balanced', random_state=42)
        rf.fit(X_train_tfidf, y_train)
        oob_errors.append(1 - rf.oob_score_)

    plt.figure(figsize=(10, 6))
    plt.plot(list(n_estimators_range), oob_errors, marker='o', linestyle='-', color='r')
    plt.title("Random Forest: OOB Error vs Number of Estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("OOB Error (1 - oob_score)")
    plt.grid(True)
    os.makedirs('data/output/metrics', exist_ok=True)
    plt.savefig('data/output/metrics/rf_learning_curve.png')
    plt.close()
    print("RF learning curve saved to data/output/metrics/rf_learning_curve.png")

def run_comparison():
    print("--- UPGRADED THREE-MODEL COMPARISON (Prompt 3) ---")
    
    # Load Gold Database
    df = pd.read_csv('data/processed/gold_database.csv')
    X = df['original_text'].astype(str)
    y = df['label']
    class_names = ['Unlikely', 'Partial', 'Highly Likely']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # 1. Model 1: Baseline (TF-IDF + RF)
    print("Evaluating Model 1: TF-IDF + Random Forest...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Run learning curve check (Prompt 4)
    plot_rf_learning_curve(X_train_tfidf, y_train)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_tfidf, y_train)
    rf_preds = rf.predict(X_test_tfidf)
    rf_probs = rf.predict_proba(X_test_tfidf)

    # 2. Model 2: Base Multilingual MiniLM
    print("Evaluating Model 2: Base Multilingual MiniLM...")
    base_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
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
    
    anchor_embs_base = base_model.encode(anchor_texts, convert_to_tensor=True)
    test_embs_base = base_model.encode(X_test.tolist(), convert_to_tensor=True)
    
    def get_rag_probs(test_embs, anchor_embs, anchor_labels):
        cos_scores = util.cos_sim(test_embs, anchor_embs)
        probs = []
        for i in range(len(test_embs)):
            scores = []
            for label in [0, 1, 2]:
                label_indices = [idx for idx, l in enumerate(anchor_labels) if l == label]
                scores.append(cos_scores[i][label_indices].max().item())
            # Softmax
            exp_scores = np.exp(scores)
            probs.append(exp_scores / np.sum(exp_scores))
        return np.array(probs)

    base_probs = get_rag_probs(test_embs_base, anchor_embs_base, anchor_labels)
    base_preds = np.argmax(base_probs, axis=1)

    # 3. Model 3: Fine-tuned Multilingual MiniLM
    ft_model_path = 'models/finetuned-politicheck-multilingual'
    if not os.path.exists(ft_model_path):
        print(f"WARNING: Fine-tuned model not found at {ft_model_path}. Skipping Model 3.")
        models_to_compare = [
            ("Baseline (RF)", rf_preds, rf_probs),
            ("Base Multilingual MiniLM", base_preds, base_probs)
        ]
    else:
        print("Evaluating Model 3: Fine-tuned Multilingual MiniLM...")
        ft_model = SentenceTransformer(ft_model_path)
        anchor_embs_ft = ft_model.encode(anchor_texts, convert_to_tensor=True)
        test_embs_ft = ft_model.encode(X_test.tolist(), convert_to_tensor=True)
        ft_probs = get_rag_probs(test_embs_ft, anchor_embs_ft, anchor_labels)
        ft_preds = np.argmax(ft_probs, axis=1)
        models_to_compare = [
            ("Baseline (RF)", rf_preds, rf_probs),
            ("Base Multilingual MiniLM", base_preds, base_probs),
            ("Proposed Fine-tuned", ft_preds, ft_probs)
        ]

    # --- Metrics Collection ---
    metrics_list = []
    for name, preds, probs in models_to_compare:
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average='weighted'),
            "Recall": recall_score(y_test, preds, average='weighted'),
            "F1-Score": f1_score(y_test, preds, average='weighted')
        })
        
        report = classification_report(y_test, preds, target_names=class_names)
        print(f"\nClassification Report for {name}:")
        print(report)
        
        # Save classification report to text file (Problem 4)
        sanitized_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        report_path = f'data/output/metrics/report_{sanitized_name}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('data/output/metrics/full_comparison.csv', index=False)

    # --- Visualizations ---
    os.makedirs('data/output/metrics', exist_ok=True)

    # Grouped Bar Chart
    metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/output/metrics/model_comparison_bar.png')
    plt.close()

    # Side-by-side Confusion Matrices
    n_models = len(models_to_compare)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    for i, (name, preds, _) in enumerate(models_to_compare):
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('data/output/metrics/confusion_matrices_all3.png')
    plt.close()

    # ROC-AUC and PR Curves
    for name, _, probs in models_to_compare:
        sanitized_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # ROC
        plt.figure(figsize=(8, 6))
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC-AUC: {name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig(f'data/output/metrics/roc_auc_{sanitized_name}.png')
        plt.close()

        # PR
        plt.figure(figsize=(8, 6))
        for i in range(3):
            p, r, _ = precision_recall_curve(y_test_bin[:, i], probs[:, i])
            plt.plot(r, p, label=f'{class_names[i]}')
        plt.title(f'Precision-Recall: {name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(f'data/output/metrics/pr_curve_{sanitized_name}.png')
        plt.close()

    print("\nAll comparison results and charts saved to data/output/metrics/")

if __name__ == "__main__":
    run_comparison()
