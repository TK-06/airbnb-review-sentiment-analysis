# %% [markdown]
# # 1. Data Preprocessing
#
# Downloads and cleans InsideAirbnb's London review data, labels sentiment,
# and balances classes. Adapted from Taimour Karim's original
# `1_dataPreprocessing.ipynb` — see the README Credits section. The true raw
# dataset behind the upstream notebook was not available (404 on the source
# repo), so this pulls an equivalent raw dataset independently from
# InsideAirbnb's public data.

# %%
import gzip
import re
import shutil
import string
from pathlib import Path

import nltk
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from tqdm import tqdm

tqdm.pandas()

# %%
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# %% [markdown]
# ## Download raw data
#
# InsideAirbnb's London detailed-reviews snapshot (~277MB compressed,
# public third-party data, not redistributed in this repo). Skips the
# download if it's already present.

# %%
SNAPSHOT_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    "2026-06-19/data/reviews.csv.gz"
)
RAW_DIR = Path("../data/raw")
GZ_PATH = RAW_DIR / "reviews.csv.gz"
CSV_PATH = RAW_DIR / "reviews.csv"

if CSV_PATH.exists():
    print(f"{CSV_PATH} already exists, skipping download.")
else:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SNAPSHOT_URL} ...")
    response = requests.get(SNAPSHOT_URL, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(GZ_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=GZ_PATH.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    print("Decompressing ...")
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    GZ_PATH.unlink()
    print(f"Done: {CSV_PATH}")

# %% [markdown]
# ## Load raw data

# %%
df = pd.read_csv(CSV_PATH)
print(df.shape)
df.head()

# %%
df.info()
df.isnull().sum()

# %% [markdown]
# Save a small sample of the *raw, pre-cleaning* data for the dashboard's
# "Uncleaned Data" comparison view (the full raw file is too large to commit).

# %%
df.head(1000).to_csv("../data/raw_sample.csv", index=False)

# %% [markdown]
# ## Text cleaning
#
# Ported from Taimour Karim's original preprocessing notebook, with one
# change: these take `stop_words`/`lemmatizer`/`stemmer` as parameters
# instead of relying on module-level globals, so each step stays a pure,
# easy-to-follow function.


# %%
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


def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_mentions_hashtags(text):
    return re.sub(r"\@\w+|\#\w+", "", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text, stop_words):
    return " ".join(word for word in text.split() if word not in stop_words)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_emojis(text):
    return _EMOJI_PATTERN.sub("", text)


def handle_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text, stop_words, lemmatizer, stemmer):
    """Full cleaning pipeline applied to one review comment."""
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, stop_words)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = handle_whitespace(text)
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    text = " ".join(stemmer.stem(word) for word in text.split())
    return text


# %% [markdown]
# ## Clean and label

# %%
df = df.drop(columns=["id", "reviewer_id", "listing_id"])
# A small number of rows have a genuinely null comment (no text to analyze at
# all) - drop them outright rather than stringifying NaN into a fake "nan"
# token that would pollute cleaned_comments and, later, the TF-IDF vocabulary.
df = df.dropna(subset=["comments"])
df["comments"] = df["comments"].astype(str)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df["cleaned_comments"] = df["comments"].progress_apply(
    lambda text: clean_text(text, stop_words, lemmatizer, stemmer)
)

# %% [markdown]
# Some comments (e.g. URL-only, emoji-only, or all-stopword text) clean down
# to an empty string. These carry no signal to analyze and, if kept, round-trip
# through CSV as NaN (pandas writes "" as a blank cell, and reads a blank cell
# back as NaN by default) - drop them now while they're still recognizable as
# empty strings, rather than let them resurface as a confusing null later.

# %%
empty_after_cleaning = (df["cleaned_comments"].str.strip() == "").sum()
print("Empty after cleaning (dropping):", empty_after_cleaning)
df = df[df["cleaned_comments"].str.strip() != ""]

# %%
print("Duplicates before dedup:", df.duplicated().sum())
df = df.drop_duplicates()

# %%
df["polarity"] = df["cleaned_comments"].progress_apply(
    lambda text: TextBlob(text).sentiment.polarity
)
df["sentiment"] = pd.cut(
    df["polarity"], bins=3, labels=["negative", "neutral", "positive"]
)
print(df["sentiment"].value_counts())

# %% [markdown]
# `pd.cut`'s equal-width binning of TextBlob polarity is a crude labeling
# heuristic and produces heavy class imbalance (see counts above) — flagged
# as a known limitation in the README, not fixed here (out of scope: this
# project reproduces and documents the pipeline, not the sentiment-labeling
# methodology itself).
#
# Balance classes — dynamically downsampling every class to the size of the
# smallest one, unlike the original's hardcoded "3915 per class" (tuned to a
# specific historical snapshot that no longer applies).


# %%
def balance_classes(df, label_col, random_state=42):
    """Downsample every class to the size of the smallest class."""
    min_count = df[label_col].value_counts().min()
    balanced = df.groupby(label_col, group_keys=False).sample(
        n=min_count, random_state=random_state
    )
    return balanced.sort_values(by=label_col).reset_index(drop=True)


df = balance_classes(df, "sentiment")
print(df["sentiment"].value_counts())

# %% [markdown]
# ## Final cleanup and save

# %%
df = df.dropna()
df = df[df["date"].str.contains(r"^\d", na=False)]
df["date"] = pd.to_datetime(df["date"])
print(df.shape)
df.head()

# %%
df.to_csv("../data/cleaned/reviews.csv", index=False)
print("Saved data/cleaned/reviews.csv")
