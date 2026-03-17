import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def process_data(df, use_semantic=True):
    """
    Highly optimized batch labeling using Semantic Similarity.
    """
    df.columns = [c.lower() for c in df.columns]
    target_col = 'remark' if 'remark' in df.columns else df.columns[-1]
    
    # Standardize column name
    if 'promise' in df.columns:
        df = df.rename(columns={'promise': 'text'})
    
    print(f"Labeling {len(df)} rows using {'Semantic' if use_semantic else 'Keyword'} logic...")
    
    if not use_semantic:
        # Simple keyword fallback
        def simple_label(t):
            t = str(t).lower()
            if any(w in t for w in ["completed", "achieved", "implemented"]): return 2
            if any(w in t for w in ["underway", "progress", "ongoing"]): return 1
            return 0
        df['label'] = df[target_col].apply(simple_label)
    else:
        model = get_model()
        
        # Batch Anchors
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
        
        # Batch Encode
        remarks = df[target_col].astype(str).tolist()
        remark_embs = model.encode(remarks, convert_to_tensor=True, show_progress_bar=True)
        anchor_embs = model.encode(anchor_texts, convert_to_tensor=True)
        
        # Similarity Matrix
        cos_scores = util.cos_sim(remark_embs, anchor_embs)
        best_anchor_indices = torch.argmax(cos_scores, dim=1).tolist()
        
        df['label'] = [anchor_labels[idx] for idx in best_anchor_indices]

    cols = ['text', 'label']
    if 'sector' in df.columns: cols.append('sector')
    return df[cols]
