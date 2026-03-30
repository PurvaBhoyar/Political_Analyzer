import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import itertools
import random

def run_finetuning():
    print("--- STARTING MODEL FINE-TUNING (Prompt 2) ---")
    
    # 1. Load data
    db_path = 'data/processed/gold_database.csv'
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return
    df = pd.read_csv(db_path)
    
    # 2. Create sentence pairs
    print("Generating sentence pairs...")
    sentences = df['original_text'].astype(str).tolist()
    labels = df['label'].tolist()
    
    # Generate all possible pairs (limited)
    all_pairs = list(itertools.combinations(range(len(sentences)), 2))
    random.seed(42)
    random.shuffle(all_pairs)
    
    # --- USER CONFIGURED PARAMETERS ---
    max_pairs = 2000 
    selected_pairs = all_pairs[:max_pairs]
    
    examples = []
    for idx1, idx2 in selected_pairs:
        s1 = sentences[idx1]
        s2 = sentences[idx2]
        score = 1.0 if labels[idx1] == labels[idx2] else 0.0
        examples.append(InputExample(texts=[s1, s2], label=score))
    
    # 3. Train/Val Split
    train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
    
    # 4. Load Model (With Safe Checkpoint Support for Windows)
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model_save_path = 'models/finetuned-politicheck-multilingual'
    checkpoint_copy = 'models/checkpoint_tmp'
    
    if os.path.exists(model_save_path):
        import shutil
        print(f"Resuming from existing checkpoint: {model_save_path}")
        # On Windows, we must copy the model to a tmp dir to avoid file locking during save
        if os.path.exists(checkpoint_copy):
            shutil.rmtree(checkpoint_copy)
        shutil.copytree(model_save_path, checkpoint_copy)
        model = SentenceTransformer(checkpoint_copy)
    else:
        print(f"Loading base model: {model_name}")
        model = SentenceTransformer(model_name)
    
    # 5. Prepare DataLoader and Evaluator
    batch_size = 16
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Evaluator
    val_sentences1 = [e.texts[0] for e in val_examples]
    val_sentences2 = [e.texts[1] for e in val_examples]
    val_scores = [e.label for e in val_examples]
    
    eval_name = 'politicheck-val'
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_sentences1, val_sentences2, val_scores, name=eval_name)
    
    # 6. Fine-tune
    num_epochs = 3 
    os.makedirs('data/output/metrics', exist_ok=True)
    
    epoch_scores = []
    
    print(f"Fine-tuning {model_name} for {num_epochs} epochs...")
    print(f"Progress will be saved after each epoch to: {model_save_path}")
    
    for epoch in range(num_epochs):
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=1,
                  warmup_steps=100,
                  output_path=model_save_path) # Save after every epoch
        
        # Evaluate
        results = evaluator(model)
        
        # Handle dict return from newer sentence-transformers
        if isinstance(results, dict):
            score = results.get(f'{eval_name}_spearman_cosine', 0.0)
        else:
            score = results
            
        epoch_scores.append(score)
        print(f"Epoch {epoch+1} Evaluation (Spearman Correlation): {score:.4f}")

    # 7. Plot and Summary
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_scores, marker='o', linestyle='-', color='b')
    plt.title("Fine-tuning: Validation Spearman Correlation per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.grid(True)
    plt.savefig('data/output/metrics/finetuning_curve.png')
    plt.close()
    
    print("\nFine-tuning Summary:")
    print(f"Total pairs: {len(examples)}")
    print(f"Train pairs: {len(train_examples)}")
    print(f"Val pairs: {len(val_examples)}")
    for i, s in enumerate(epoch_scores):
        print(f"Epoch {i+1}: {s:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    print("Finetuning curve saved to: data/output/metrics/finetuning_curve.png")

if __name__ == "__main__":
    run_finetuning()
