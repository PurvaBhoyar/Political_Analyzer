# Technical Specifications (Section 5d, 5n)

## 1. Model Hyperparameters (Section 5n)

| Component | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| **Embedding Model** | Model Name | `paraphrase-multilingual-MiniLM-L12-v2` | SBERT dense vector model |
| | Dimensions | 512 | Size of the output vector |
| | Max Seq Length | 128 tokens | Longer text is truncated |
| **LLM Skeptic** | Model Name | `llama-3.3-70b-versatile` | Groq-hosted Llama-3 model |
| | Temperature | 0.1 | Low temperature for factual consistency |
| | Max Tokens | 600 | Limit for reasoning and JSON response |
| | Top-P | 1.0 | Standard nucleus sampling |
| **Vector Search** | Similarity Metric | `Cosine Similarity` | Dot product of normalized vectors |
| | Top-K | 3 | Number of historical matches retrieved |

## 2. Mathematical Foundations (Section 5d)

### A. Cosine Similarity
Used to measure semantic distance between 2024 promises and historical data.
$$ \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} $$

### B. Softmax (Probability Mapping)
Used in the baseline evaluation to convert similarities into class probabilities.
$$ \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} $$

### C. F1-Score
The harmonic mean of precision and recall, used to evaluate model balance.
$$ F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} $$
