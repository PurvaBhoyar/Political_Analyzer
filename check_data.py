import pandas as pd
df = pd.read_csv("processed_data.csv")
print("Total Rows:", len(df))
print("Label Counts:\n", df['label'].value_counts())