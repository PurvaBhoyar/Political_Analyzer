import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from pathlib import Path

def run_eda(db_path="data/processed/gold_database.csv", output_dir="data/output/eda"):
    base_dir = Path(__file__).parent
    abs_db_path = base_dir / db_path
    abs_output_dir = base_dir / output_dir
    
    os.makedirs(abs_output_dir, exist_ok=True)
    
    if not abs_db_path.exists():
        print(f"Error: {abs_db_path} not found.")
        return

    df = pd.read_csv(str(abs_db_path))
    
    # 1. Descriptive Statistics Table (Section 5e)
    stats = {
        "Total Records": len(df),
        "Unique Sectors": df['sector'].nunique() if 'sector' in df.columns else 0,
        "Year Distribution": df['year'].value_counts().to_dict(),
        "Label Distribution": df['label'].value_counts().to_dict()
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(abs_output_dir / "descriptive_stats.csv", index=False)
    print("Descriptive stats saved.")

    # 2. Label Distribution Plot (Section 5j/6f)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
    plt.title('Distribution of Historical Outcomes (Labels)')
    plt.xlabel('Label (0: Unlikely, 1: Partial, 2: Likely)')
    plt.ylabel('Count')
    plt.savefig(abs_output_dir / "label_distribution.png")
    plt.close()

    # 3. Sector Distribution (Section 5e)
    if 'sector' in df.columns and not df['sector'].isnull().all():
        plt.figure(figsize=(12, 6))
        df['sector'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribution of Promises by Sector')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(abs_output_dir / "sector_distribution.png")
        plt.close()

    # 4. Word Cloud (Section 5i)
    text = " ".join(df['original_text'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Manifesto Word Cloud')
    plt.savefig(abs_output_dir / "wordcloud.png")
    plt.close()

    print(f"EDA Outputs saved to {abs_output_dir}")

if __name__ == "__main__":
    run_eda()
