"""
Data Preprocessing Module for Spam SMS Text Classifier
Handles text normalization, tokenization, stopword removal, and stemming/lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure required NLTK resources are loaded or fall back safely for serverless environments
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

try:
    _stopwords = set(stopwords.words('english'))
except Exception:
    try:
        nltk.download('stopwords', quiet=True)
        _stopwords = set(stopwords.words('english'))
    except Exception:
        _stopwords = FALLBACK_STOPWORDS


def clean_text_for_nlp(text: str) -> str:
    """
    Cleans raw text for NLP TF-IDF vectorization:
    1. Converts text to lowercase.
    2. Strips URLs.
    3. Handles punctuation and special characters.
    4. Normalizes whitespaces.
    5. Tokenizes and filters stopwords.
    6. Applies Porter Stemming for vocabulary generalization.
    
    Returns cleaned and normalized string of tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags if any
    text = re.sub(r'<.*?>', ' ', text)

    # Normalize currency symbols to token representations for TF-IDF if desired or isolate them
    text = re.sub(r'[\$£€₹]', ' currency_symbol ', text)

    # Remove generic punctuation and non-alphanumeric except spaces
    # Keep words and essential tokens
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords and apply stemming
    stemmer = PorterStemmer()
    processed_tokens = [
        stemmer.stem(token) for token in tokens
        if token not in _stopwords and len(token) > 1
    ]

    return " ".join(processed_tokens)


def map_labels(label_series):
    """
    Maps categorical string labels ('ham', 'spam') to binary numeric values:
    ham -> 0
    spam -> 1
    """
    mapping = {'ham': 0, 'spam': 1}
    return label_series.astype(str).str.strip().str.lower().map(mapping)
