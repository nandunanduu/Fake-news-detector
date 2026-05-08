"""
Text preprocessing utilities for fake news detection.
"""
import re
import ssl
import nltk
import pandas as pd

# Fix SSL cert issues on macOS before downloading NLTK data
try:
    _ctx = ssl.create_default_context()
except Exception:
    pass
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Built-in English stopword list as a reliable fallback when NLTK download fails
_FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o',
    're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn',
}

try:
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words('english'))
except Exception:
    STOP_WORDS = _FALLBACK_STOPWORDS

try:
    from nltk.tokenize import word_tokenize as _wt
    def _tokenize(text: str):
        return _wt(text)
except Exception:
    def _tokenize(text: str):  # type: ignore[misc]
        return text.split()


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline: lowercase → strip URLs/HTML →
    remove punctuation & digits → tokenize → drop stopwords.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Keep only alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and filter
    tokens = _tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Add a 'cleaned_text' column to the DataFrame."""
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df
