from nlp_engine import parser, labeler, trainer

# 1. Extraction from your Review PDF
df_raw = parser.extract_review_table("data/raw/2014 BJP Manifesto Review.pdf")

# 2. Automated Labeling
df_labeled = labeler.process_data(df_raw)

# 3. Fine-tuning BERT
trainer.train_model(df_labeled)
print("Project Setup Complete. Run 'uvicorn main:app --reload' to start.")