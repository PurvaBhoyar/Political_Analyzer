import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os

# Download necessary NLTK data (Section 5h)
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

class ManifestoPreprocessor:
    def __init__(self):
        setup_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        # Remove numbers, special chars, extra whitespace (Section 5h)
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        text = text.lower().strip()
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [w for w in tokens if w not in self.stop_words]

    def stem(self, tokens):
        return [self.stemmer.stem(w) for w in tokens]

    def full_preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem(filtered)
        return " ".join(stemmed)

if __name__ == "__main__":
    pre = ManifestoPreprocessor()
    sample = "We will ensure that all farmers receive minimum support price for their crops 2024."
    print(f"Original: {sample}")
    print(f"Preprocessed: {pre.full_preprocess(sample)}")
