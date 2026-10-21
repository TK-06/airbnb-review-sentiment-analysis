"""Data and model loading for the sentiment dashboard."""
import re
import string
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned" / "reviews.csv"
RAW_SAMPLE_PATH = BASE_DIR / "data" / "raw_sample.csv"
USER_REVIEWS_PATH = BASE_DIR / "data" / "user_reviews.csv"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


@lru_cache(maxsize=1)
def _cleaning_tools():
    """Stopword set, lemmatizer and stemmer, loaded once (fetches NLTK data if missing)."""
    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)
    from nltk.corpus import stopwords

    return set(stopwords.words("english")), WordNetLemmatizer(), PorterStemmer()


def clean_text(text: str) -> str:
    """Clean one review the way notebook 1 did before the vectorizer was fitted.

    The TF-IDF vectorizer and Gradient Boosting model were trained on stemmed,
    stopword-stripped text (`cleaned_comments`), so raw text has to go through
    the same steps or most of its words won't be in the vocabulary.
    """
    stop_words, lemmatizer, stemmer = _cleaning_tools()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    text = re.sub(r"\d+", "", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    return " ".join(stemmer.stem(word) for word in text.split())


def load_data() -> pd.DataFrame:
    """Load the cleaned, labeled review dataset used throughout the dashboard."""
    return pd.read_csv(DATA_PATH).dropna().drop_duplicates()


def load_raw_sample() -> pd.DataFrame:
    """Load the small pre-cleaning sample, for the dashboard's before/after view."""
    return pd.read_csv(RAW_SAMPLE_PATH)


def load_models():
    """Load the TF-IDF vectorizer and the winning Gradient Boosting classifier."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict_sentiment(text: str, vectorizer, model) -> str:
    """Predict the sentiment label for a single piece of review text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("text has no usable words after cleaning")
    features = vectorizer.transform([cleaned])
    return model.predict(features)[0]


def save_user_review(name: str, review: str, sentiment: str) -> None:
    """Append a dashboard-submitted review to the user reviews CSV."""
    row = pd.DataFrame({"name": [name], "review": [review], "sentiment": [sentiment]})
    row.to_csv(USER_REVIEWS_PATH, mode="a", header=False, index=False)
