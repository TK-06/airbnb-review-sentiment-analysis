"""Text cleaning utilities for raw Airbnb review comments."""
import re
import string

from nltk.stem import PorterStemmer, WordNetLemmatizer

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


def _require_str(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")


def remove_urls(text: str) -> str:
    _require_str(text)
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_mentions_hashtags(text: str) -> str:
    _require_str(text)
    return re.sub(r"\@\w+|\#\w+", "", text)


def remove_punctuation(text: str) -> str:
    _require_str(text)
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text: str, stop_words: set) -> str:
    _require_str(text)
    return " ".join(word for word in text.split() if word not in stop_words)


def remove_numbers(text: str) -> str:
    _require_str(text)
    return re.sub(r"\d+", "", text)


def remove_emojis(text: str) -> str:
    _require_str(text)
    return _EMOJI_PATTERN.sub("", text)


def handle_whitespace(text: str) -> str:
    _require_str(text)
    return re.sub(r"\s+", " ", text).strip()


def lemmatize(text: str, lemmatizer: WordNetLemmatizer) -> str:
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def stem(text: str, stemmer: PorterStemmer) -> str:
    return " ".join(stemmer.stem(word) for word in text.split())


def clean_text(
    text: str,
    stop_words: set,
    lemmatizer: WordNetLemmatizer,
    stemmer: PorterStemmer,
) -> str:
    """Full cleaning pipeline applied to one review comment."""
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, stop_words)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = handle_whitespace(text)
    text = lemmatize(text, lemmatizer)
    text = stem(text, stemmer)
    return text
