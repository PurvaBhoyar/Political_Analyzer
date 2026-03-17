from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import os

class ManifestoDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

def train_model(df):
    """
    Trains a heavyweight MuRIL model on the Gold dataset.
    MuRIL is public (not gated) and specialized for Indian languages.
    """
    print("----- TRAINING START (MuRIL) -----")
    print("Label Distribution:")
    print(df['label'].value_counts())
    
    # 1. Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Setup MuRIL
    model_name = "google/muril-base-cased"
    print(f"Loading {model_name}... (Downloading ~950MB, please wait)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 3. Tokenize
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=128)
    dataset = ManifestoDataset(encodings, df['label'].tolist())

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir='./models/checkpoints',
        num_train_epochs=10,
        per_device_train_batch_size=4,   # Optimized for CPU
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        save_strategy="no",
        dataloader_pin_memory=False,
        seed=42
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    
    print("Fine-tuning MuRIL... (Starting training iterations)")
    trainer.train()
    
    print("Saving Final MuRIL Model...")
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("Training Complete. Model saved to ./models/final_model")
