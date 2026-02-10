from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd

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
    print("----- DIAGNOSIS START -----")
    print("Checking Label Distribution (If mostly 0, fix labeler.py!):")
    print(df['label'].value_counts())
    print("---------------------------")

    # 1. SHUFFLE DATA (Critical to prevent model collapse)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Setup Model & Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # 3. Tokenize
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=128)
    dataset = ManifestoDataset(encodings, df['label'].tolist())

    # 4. ENHANCED TRAINING ARGUMENTS (The Real Fix)
    args = TrainingArguments(
        output_dir='./models/checkpoints',
        num_train_epochs=5,              # 5 Epochs is good
        per_device_train_batch_size=8,   # Batch size 8 is fine for CPU
        learning_rate=2e-5,              # CRITICAL: Controls how fast it learns
        weight_decay=0.01,               # CRITICAL: Prevents overfitting
        warmup_steps=50,                 # Helps model settle in
        logging_dir='./logs',            # Enables logs
        logging_steps=10,                # Prints loss every 10 steps
        save_strategy="epoch",           # Save backup every epoch
        dataloader_pin_memory=False,     # Fixes CPU warning
        seed=42                          # Ensures reproducibility
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving Final Model...")
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("Training Complete. Model saved to ./models/final_model")