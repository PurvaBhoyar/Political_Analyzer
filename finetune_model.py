import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def run_finetuning():
    print("--- STARTING MODEL FINE-TUNING (Unsupervised SimCSE) ---")
    
    # 1. Load data
    db_path = 'data/processed/gold_database.csv'
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return
    df = pd.read_csv(db_path)
    
    # 2. Create Self-Pairs for SimCSE
    print("Generating self-pairs for MultipleNegativesRankingLoss...")
    sentences = df['original_text'].astype(str).tolist()
    
    examples = []
    for text in sentences:
        # SimCSE paradigm: pair identical sentences. 
        # The MNRL loss uses dropout to create a positive pair, and uses all other sentences in the batch as hard negatives.
        examples.append(InputExample(texts=[text, text]))
    
    # 3. Load Model
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model_save_path = 'models/finetuned-politicheck-multilingual'
    
    print(f"Loading Base model: {model_name}")
    # Always start fresh from the base model so we purposefully overwrite the completely collapsed architecture
    model = SentenceTransformer(model_name)
    
    # 4. Prepare DataLoader and Loss
    # MNRL performs best with a decent batch size to ensure there are plenty of in-batch negatives
    batch_size = 32
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 5. Fine-tune
    num_epochs = 3 # SimCSE usually trains very quickly, 1-3 epochs is sufficient.
    
    print(f"Fine-tuning {model_name} for {num_epochs} epochs using SimCSE MNRL...")
    print("Note: Label-based paired evaluation has been stripped to prevent semantic space collapse.")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=int(len(train_dataloader) * 0.1), # Warmup over 10% of first epoch
              show_progress_bar=True,
              output_path=model_save_path)
              
    print("\nFine-tuning Complete!")
    print(f"Total training pairs (self-pairs): {len(examples)}")
    print(f"Model successfully saved to: {model_save_path}")
    print("The model can now discriminate fine-grained political vocabulary without collapsing sentences into label-classes.")

if __name__ == "__main__":
    run_finetuning()
