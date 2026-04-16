import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)

def run_comparison():
    print("--- DENSE RAG EVALUATION (BASE VS FINE-TUNED) ---")
    
    # Load Gold Database
    df = pd.read_csv('data/processed/gold_database.csv')
    X = df['original_text'].astype(str)
    y = df['label'].values
    class_names = ['Unlikely', 'Partial', 'Highly Likely']

    # Split data (Ensuring the same dataset slice is used to represent the historical vs test query split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    print("Loading Base Multilingual Model...")
    base_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Encoding database and queries...")
    train_embs_base = base_model.encode(X_train.tolist(), convert_to_tensor=True, show_progress_bar=False)
    test_embs_base = base_model.encode(X_test.tolist(), convert_to_tensor=True, show_progress_bar=False)
    
    def get_knn_rag_probs(test_embs, train_embs, train_labels):
        cos_scores = util.cos_sim(test_embs, train_embs)
        probs = []
        preds = []
        for i in range(len(test_embs)):
            # Probabilities derived from similarity scores to all labels
            scores = [0.0, 0.0, 0.0]
            for label in [0, 1, 2]:
                label_indices = np.where(train_labels == label)[0]
                if len(label_indices) > 0:
                    # use the max similarity score belonging to this historical label group
                    scores[label] = cos_scores[i][label_indices].max().item()
            
            exp_scores = np.exp(scores)
            p = exp_scores / np.sum(exp_scores)
            probs.append(p)
            
            # Prediction remains pure top-1 nearest neighbor strictly via Cosine Sim
            best_idx = torch.argmax(cos_scores[i]).item()
            preds.append(train_labels[best_idx])
            
        return np.array(preds), np.array(probs)

    print("Evaluating Base Model KNN RAG Retrieval...")
    base_preds, base_probs = get_knn_rag_probs(test_embs_base, train_embs_base, y_train)

    # Fine-tuned Model Validation
    ft_model_path = 'models/finetuned-politicheck-multilingual'
    if not os.path.exists(ft_model_path):
        print(f"WARNING: Fine-tuned model not found at {ft_model_path}. Skipping.")
        models_to_compare = [
            ("Base Multilingual MiniLM", base_preds, base_probs)
        ]
    else:
        print("Loading Fine-tuned Multilingual MiniLM...")
        ft_model = SentenceTransformer(ft_model_path)
        
        print("Encoding semantic space...")
        train_embs_ft = ft_model.encode(X_train.tolist(), convert_to_tensor=True, show_progress_bar=False)
        test_embs_ft = ft_model.encode(X_test.tolist(), convert_to_tensor=True, show_progress_bar=False)
        
        print("Evaluating Fine-tuned Model KNN RAG Retrieval...")
        ft_preds, ft_probs = get_knn_rag_probs(test_embs_ft, train_embs_ft, y_train)
        
        models_to_compare = [
            ("Base Multilingual MiniLM", base_preds, base_probs),
            ("Proposed Fine-tuned MiniLM", ft_preds, ft_probs)
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
        
        # Save exact classification reports
        sanitized_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        report_path = f'data/output/metrics/report_{sanitized_name}.txt'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('data/output/metrics/dense_comparison.csv', index=False)

    # --- Visualizations ---
    # Side-by-side Confusion Matrices
    n_models = len(models_to_compare)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    for i, (name, preds, _) in enumerate(models_to_compare):
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('data/output/metrics/confusion_matrices_rag.png')
    plt.close()

    # ROC-AUC curves
    for name, _, probs in models_to_compare:
        sanitized_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.figure(figsize=(8, 6))
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve: {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'data/output/metrics/roc_auc_{sanitized_name}.png')
        plt.close()

    print("\nSuccessfully wiped TF-IDF. True RAG metrics explicitly recorded in data/output/metrics/")

if __name__ == "__main__":
    run_comparison()
