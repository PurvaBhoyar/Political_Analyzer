import pandas as pd
import numpy as np
import os
from rouge_score import rouge_scorer
import bert_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_metrics():
    print("--- Evaluating Advanced RAG Metrics ---")
    
    report_path = 'data/output/2024_fact_check_report.csv'
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found. Please run run_pipeline.py first.")
        # If it doesn't exist, we can try to run the pipeline automatically or just inform user
        return
        
    df = pd.read_csv(report_path)
    if '2024_Promise' not in df.columns or 'Historical_Match' not in df.columns:
        print("Error: Required columns '2024_Promise' and 'Historical_Match' not found in report.")
        return
        
    promises = df['2024_Promise'].astype(str).tolist()
    matches = df['Historical_Match'].astype(str).tolist()
    
    print(f"Loaded {len(promises)} fact-check pairs. Computing ROUGE scores...")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_f, rouge2_f, rougeL_f = [], [], []
    
    for p, m in zip(promises, matches):
        scores = scorer.score(p, m)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)
        
    df['ROUGE_1'] = rouge1_f
    df['ROUGE_2'] = rouge2_f
    df['ROUGE_L'] = rougeL_f
    
    print(f"Average ROUGE-1: {np.mean(rouge1_f):.4f}")
    print(f"Average ROUGE-2: {np.mean(rouge2_f):.4f}")
    print(f"Average ROUGE-L: {np.mean(rougeL_f):.4f}")
    
    print("\nComputing BERTScore... (this may take a moment to load the model)")
    # Using a smaller fast model to avoid huge downloads during dev
    P, R, F1 = bert_score.score(promises, matches, lang='en', model_type='distilbert-base-uncased', verbose=True)
    
    df['BERTScore_P'] = P.numpy()
    df['BERTScore_R'] = R.numpy()
    df['BERTScore_F1'] = F1.numpy()
    
    print(f"Average BERTScore F1: {df['BERTScore_F1'].mean():.4f}")
    
    os.makedirs('data/output/metrics', exist_ok=True)
    out_path = 'data/output/metrics/advanced_rag_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f"\nDetailed metrics saved to: {out_path}")
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    
    # Avoid overlapping completely by using distplot with alpha
    sns.histplot(df['BERTScore_F1'], bins=20, kde=True, color='blue', label='BERTScore F1', alpha=0.5)
    sns.histplot(df['ROUGE_L'], bins=20, kde=True, color='orange', label='ROUGE-L', alpha=0.5)
    
    plt.title('Distribution of Retrieval Quality Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plot_path = 'data/output/metrics/rag_quality_dist.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Score distribution plot saved to: {plot_path}")

if __name__ == "__main__":
    evaluate_metrics()
