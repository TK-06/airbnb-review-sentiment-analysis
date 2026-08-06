# Airbnb Review Sentiment Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a single inherited notebook into a working, honestly-attributed, end-to-end sentiment analysis project (data sourcing → cleaning → 7 classical ML models → fixed LSTM → fixed DistilBERT → working Dash dashboard), tested, documented, and pushed to a new public GitHub repo.

**Architecture:** A small `src/` package holds pure, unit-tested text-cleaning and class-balancing logic shared by notebook 1. Four notebooks (`1_data_preprocessing`, `2_classical_ml`, `3_deep_learning_lstm`, `4_llm_bert`) form the pipeline, each producing artifacts the next stage (or the dashboard) consumes. A `dashboard/` package separates data/model loading (`data.py`, unit-tested) from chart-building (`charts.py`, unit-tested) from Dash wiring (`app.py`, smoke-tested manually — Dash callback testing is out of scope).

**Tech Stack:** Python 3.13, pandas, scikit-learn, NLTK, TextBlob, TensorFlow + tf-keras, HuggingFace Transformers (DistilBERT), Dash + Plotly, pytest, jupytext (for building notebooks from readable `.py` source).

## Global Constraints

- README must include a "Credits / Origin" section attributing Taimour Karim's [Beginner-Data-Science-Projects](https://github.com/tkarim45/Beginner-Data-Science-Projects) (MIT licensed) as the pipeline this project completes and documents — not framed as solo original research.
- The raw InsideAirbnb download (~277MB) is never committed to git — only the small derived `data/cleaned/reviews.csv` and a small `data/raw_sample.csv` illustrative sample.
- Only `models/tfidf_vectorizer.pkl` and `models/gradient_boosting_model.pkl` are committed. No other classifier `.pkl`, no LSTM/BERT weights.
- No live deployment — local-run only, documented setup instructions.
- Dashboard stays a Dash app (not rebuilt in Streamlit).
- Target Python 3.13 in a dedicated venv; do not add speculative version-pinning unless an actual install failure is hit during implementation.
- LICENSE is MIT, crediting both Taimour Karim (original pipeline, 2024-present) and TK-06 (this implementation, 2026).

---

## File Structure

```
airbnb-review-sentiment-analysis/
├── README.md, requirements.txt, .gitignore, LICENSE, pyproject.toml
├── docs/superpowers/specs/, docs/superpowers/plans/
├── src/
│   ├── __init__.py
│   ├── text_cleaning.py     # pure text-cleaning functions
│   └── balance.py           # dynamic class balancing
├── scripts/
│   ├── __init__.py
│   └── download_raw_data.py # downloads + decompresses InsideAirbnb London reviews
├── notebooks/
│   ├── 1_data_preprocessing.ipynb (+ paired .py via jupytext)
│   ├── 2_classical_ml.ipynb
│   ├── 3_deep_learning_lstm.ipynb (+ paired .py)
│   └── 4_llm_bert.ipynb (+ paired .py)
├── data/
│   ├── raw/                 # gitignored — created by download script
│   ├── cleaned/reviews.csv  # committed, produced by notebook 1
│   ├── raw_sample.csv       # committed, small illustrative pre-cleaning sample
│   └── user_reviews.csv     # committed seed file for dashboard's review form
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── gradient_boosting_model.pkl
├── dashboard/
│   ├── __init__.py
│   ├── data.py               # load_data, load_models, predict_sentiment, load_raw_sample, save_user_review
│   ├── charts.py              # chart builders
│   └── app.py                 # Dash layout & callbacks
└── tests/
    ├── test_text_cleaning.py
    ├── test_balance.py
    ├── test_download_raw_data.py
    ├── test_pipeline.py
    └── test_charts.py
```

---

### Task 1: Project scaffold, environment, licensing

**Files:**
- Create: `requirements.txt`, `.gitignore`, `LICENSE`, `pyproject.toml`
- Create: `src/__init__.py`, `scripts/__init__.py`, `dashboard/__init__.py`
- Create directories: `notebooks/`, `data/raw/`, `data/cleaned/`, `models/`, `tests/`

**Interfaces:**
- Produces: a working venv with all dependencies installed; `pyproject.toml` makes `src`, `scripts`, `dashboard` importable from `tests/` without an editable install.

- [ ] **Step 1: Create directories**

```bash
mkdir -p src scripts notebooks data/raw data/cleaned models dashboard tests
```

- [ ] **Step 2: Write `requirements.txt`**

```
pandas
numpy
scikit-learn
nltk
textblob
joblib
tensorflow
tf-keras
transformers
dash
plotly
wordcloud
matplotlib
seaborn
requests
tqdm
jupyter
jupytext
pytest
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
data/raw/*
!data/raw/.gitkeep
*.csv.gz
.ipynb_checkpoints/
catboost_info/
.DS_Store
*.egg-info/
```

- [ ] **Step 4: `touch data/raw/.gitkeep`** (so the empty gitignored dir still exists after clone)

- [ ] **Step 5: Write `LICENSE`**

```
MIT License

Copyright (c) 2024-present Taimour Karim (original pipeline design and code,
see README Credits section)
Copyright (c) 2026 TK-06 (this implementation)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 7: Create empty package markers**

```bash
: > src/__init__.py
: > scripts/__init__.py
: > dashboard/__init__.py
```

- [ ] **Step 8: Create and activate a venv, install dependencies**

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
```

Expected: installs cleanly. If `tensorflow`/`transformers` fail to build on Python 3.13, note the exact error here and resolve then — do not preemptively downgrade Python.

- [ ] **Step 9: One-time NLTK resource download**

```bash
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']]"
```

- [ ] **Step 10: Verify pytest runs (even with zero tests yet)**

Run: `pytest`
Expected: `no tests ran` (or similar) with no import errors.

- [ ] **Step 11: Commit**

```bash
git add requirements.txt .gitignore LICENSE pyproject.toml src/__init__.py scripts/__init__.py dashboard/__init__.py data/raw/.gitkeep
git commit -m "Scaffold project structure, environment, and MIT license"
```

---

### Task 2: Shared preprocessing utilities (`src/`)

**Files:**
- Create: `src/text_cleaning.py`
- Create: `src/balance.py`
- Test: `tests/test_text_cleaning.py`
- Test: `tests/test_balance.py`

**Interfaces:**
- Produces: `clean_text(text, stop_words, lemmatizer, stemmer) -> str` and the individual cleaning steps it composes; `balance_classes(df, label_col, random_state=42) -> pd.DataFrame`. Notebook 1 (Task 3) imports both.

- [ ] **Step 1: Write the failing tests — `tests/test_text_cleaning.py`**

```python
import pytest
from nltk.stem import PorterStemmer, WordNetLemmatizer

from src.text_cleaning import (
    clean_text,
    handle_whitespace,
    remove_emojis,
    remove_mentions_hashtags,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
    remove_urls,
)


def test_remove_urls_strips_http_and_www_links():
    assert remove_urls("Check http://example.com and www.test.org now") == "Check  and  now"


def test_remove_urls_rejects_non_string():
    with pytest.raises(TypeError):
        remove_urls(123)


def test_remove_mentions_hashtags_strips_both():
    assert remove_mentions_hashtags("Great host @jane_doe #airbnb #london") == "Great host   "


def test_remove_punctuation_strips_all_punctuation():
    assert remove_punctuation("Great place, would stay again!") == "Great place would stay again"


def test_remove_stopwords_drops_only_stopwords():
    stop_words = {"is", "the", "a"}
    assert remove_stopwords("this is a test", stop_words) == "this test"


def test_remove_numbers_strips_digits():
    assert remove_numbers("Room 42 on floor 3") == "Room  on floor "


def test_remove_emojis_strips_emoji_leaves_text():
    assert remove_emojis("Loved it \U0001F600 would return") == "Loved it  would return"


def test_handle_whitespace_collapses_and_trims():
    assert handle_whitespace("  too   much   space   ") == "too much space"


def test_clean_text_full_pipeline_lowercases_strips_stopword_and_stems():
    stop_words = {"the", "was"}
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    result = clean_text("The Room was Running!", stop_words, lemmatizer, stemmer)
    assert result == "room run"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_text_cleaning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.text_cleaning'`

- [ ] **Step 3: Write `src/text_cleaning.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_text_cleaning.py -v`
Expected: 9 passed

- [ ] **Step 5: Write the failing test — `tests/test_balance.py`**

```python
import pandas as pd

from src.balance import balance_classes


def test_balance_classes_downsamples_to_smallest_class():
    df = pd.DataFrame({
        "sentiment": ["positive"] * 10 + ["neutral"] * 6 + ["negative"] * 3,
        "text": [f"review {i}" for i in range(19)],
    })
    balanced = balance_classes(df, "sentiment")
    counts = balanced["sentiment"].value_counts()
    assert counts["positive"] == 3
    assert counts["neutral"] == 3
    assert counts["negative"] == 3
    assert len(balanced) == 9


def test_balance_classes_is_deterministic_given_random_state():
    df = pd.DataFrame({
        "sentiment": ["a"] * 5 + ["b"] * 2,
        "text": [f"row {i}" for i in range(7)],
    })
    first = balance_classes(df, "sentiment", random_state=1)
    second = balance_classes(df, "sentiment", random_state=1)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_balance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.balance'`

- [ ] **Step 7: Write `src/balance.py`**

```python
"""Class-balancing utilities for the labeled review dataset."""
import pandas as pd


def balance_classes(
    df: pd.DataFrame, label_col: str, random_state: int = 42
) -> pd.DataFrame:
    """Downsample every class to the size of the smallest class.

    Unlike a hardcoded sample size, this adapts to whatever class
    distribution the input data happens to have.
    """
    min_count = df[label_col].value_counts().min()
    balanced = df.groupby(label_col, group_keys=False).sample(
        n=min_count, random_state=random_state
    )
    return balanced.sort_values(by=label_col).reset_index(drop=True)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_balance.py -v`
Expected: 2 passed

- [ ] **Step 9: Commit**

```bash
git add src/text_cleaning.py src/balance.py tests/test_text_cleaning.py tests/test_balance.py
git commit -m "Add tested text-cleaning and class-balancing utilities"
```

---

### Task 3: Raw data acquisition & notebook 1 (data preprocessing)

**Files:**
- Create: `scripts/download_raw_data.py`
- Test: `tests/test_download_raw_data.py`
- Create: `notebooks/1_data_preprocessing.py` (jupytext source) → build → `notebooks/1_data_preprocessing.ipynb`
- Produces (generated, not hand-written): `data/cleaned/reviews.csv`, `data/raw_sample.csv`

**Interfaces:**
- Consumes: `src.text_cleaning.clean_text`, `src.balance.balance_classes` (Task 2).
- Produces: `data/cleaned/reviews.csv` with columns `date, reviewer_name, comments, cleaned_comments, polarity, sentiment` — consumed by Task 4 (notebook 2), Task 5, Task 6, and Task 7 (dashboard). `data/raw_sample.csv` (raw pre-cleaning columns: `listing_id, id, date, reviewer_id, reviewer_name, comments`) — consumed by Task 9 (dashboard "Uncleaned Data" view).

- [ ] **Step 1: Write the failing test — `tests/test_download_raw_data.py`**

```python
import gzip

from scripts.download_raw_data import decompress


def test_decompress_writes_uncompressed_content(tmp_path):
    original = "listing_id,date,comments\n1,2024-01-01,hello\n"
    gz_path = tmp_path / "reviews.csv.gz"
    csv_path = tmp_path / "reviews.csv"
    with gzip.open(gz_path, "wt") as f:
        f.write(original)

    decompress(gz_path, csv_path)

    assert csv_path.read_text() == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download_raw_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.download_raw_data'`

- [ ] **Step 3: Write `scripts/download_raw_data.py`**

```python
"""Download and decompress the InsideAirbnb London detailed reviews dataset.

Source: https://insideairbnb.com/get-the-data/ (public, third-party data —
not redistributed in this repo). Run this once before executing
notebooks/1_data_preprocessing.ipynb:

    python scripts/download_raw_data.py
"""
import gzip
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

SNAPSHOT_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    "2026-06-19/data/reviews.csv.gz"
)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
GZ_PATH = RAW_DIR / "reviews.csv.gz"
CSV_PATH = RAW_DIR / "reviews.csv"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def decompress(gz_path: Path, csv_path: Path) -> None:
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists, skipping download.")
        return
    print(f"Downloading {SNAPSHOT_URL} ...")
    download(SNAPSHOT_URL, GZ_PATH)
    print("Decompressing ...")
    decompress(GZ_PATH, CSV_PATH)
    GZ_PATH.unlink()
    print(f"Done: {CSV_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_download_raw_data.py -v`
Expected: 1 passed

- [ ] **Step 5: Actually run the download (real network call, one-time)**

```bash
python scripts/download_raw_data.py
```

Expected: progress bar completes, `data/raw/reviews.csv` exists (~800MB-1GB). This is a real download — expect several minutes depending on connection.

- [ ] **Step 6: Commit the script and test**

```bash
git add scripts/download_raw_data.py tests/test_download_raw_data.py
git commit -m "Add InsideAirbnb raw data download script"
```

- [ ] **Step 7: Write the jupytext source `notebooks/1_data_preprocessing.py`**

```python
# %% [markdown]
# # 1. Data Preprocessing
#
# Downloads (via `scripts/download_raw_data.py`, run separately) and cleans
# InsideAirbnb's London review data, labels sentiment, and balances classes.
# Adapted from Taimour Karim's original `1_dataPreprocessing.ipynb` — see the
# README Credits section. The true raw dataset behind the upstream notebook
# was not available (404 on the source repo), so this pulls an equivalent
# raw dataset independently from InsideAirbnb's public data.

# %%
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().parent))
from src.balance import balance_classes
from src.text_cleaning import clean_text

tqdm.pandas()

# %%
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# %% [markdown]
# ## Load raw data

# %%
df = pd.read_csv("../data/raw/reviews.csv")
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
# ## Clean and label

# %%
df = df.drop(columns=["id", "reviewer_id", "listing_id"])
df["comments"] = df["comments"].astype(str)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df["cleaned_comments"] = df["comments"].progress_apply(
    lambda text: clean_text(text, stop_words, lemmatizer, stemmer)
)

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
# Balance classes — dynamically, unlike the original's hardcoded "3915 per
# class" (tuned to a specific historical snapshot that no longer applies).

# %%
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
```

- [ ] **Step 8: Build the notebook from the jupytext source**

```bash
jupytext --to notebook notebooks/1_data_preprocessing.py
```

Expected: creates `notebooks/1_data_preprocessing.ipynb` alongside the `.py` source.

- [ ] **Step 9: Execute the notebook end to end**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/1_data_preprocessing.ipynb
```

Expected: completes without error. This is the slow step — cleaning + TextBlob polarity over ~1.5M rows on CPU can take tens of minutes; the `tqdm` progress bars will show live progress.

- [ ] **Step 10: Verify the output artifacts**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/cleaned/reviews.csv')
print(df.shape)
print(df['sentiment'].value_counts())
print(df.columns.tolist())
assert df.isnull().sum().sum() == 0
assert set(df['sentiment'].unique()) == {'negative', 'neutral', 'positive'}
assert df['sentiment'].value_counts().nunique() == 1, 'classes are not balanced'
print('OK')
"
```

Expected: prints `OK`, with all three sentiment classes at equal counts.

- [ ] **Step 11: Commit**

```bash
git add notebooks/1_data_preprocessing.py notebooks/1_data_preprocessing.ipynb data/cleaned/reviews.csv data/raw_sample.csv
git commit -m "Add notebook 1: source raw InsideAirbnb data, clean, label, balance"
```

---

### Task 4: Notebook 2 — classical ML (adapt existing notebook)

**Files:**
- Copy: `C:\Users\Palan\OneDrive\Documents\computer_programming\Projects\2_machineLearning.ipynb` → `notebooks/2_classical_ml.ipynb`
- Modify (via patch script, see Step 2): specific cells within the copied notebook

**Interfaces:**
- Consumes: `data/cleaned/reviews.csv` (Task 3).
- Produces: `models/tfidf_vectorizer.pkl`, `models/gradient_boosting_model.pkl` — consumed by Task 8 (dashboard/data.py tests) and Task 9 (dashboard app).

This notebook already exists and works (7 classifiers trained, Gradient Boosting the historical winner at ~90.2% on the old data snapshot). Rather than rewriting it, copy it as-is and apply precise, minimal edits: point it at the new data/model paths, drop 6 of the 7 `joblib.dump` calls (only the vectorizer + winning Gradient Boosting model get committed, per the models-scope decision — the other 5 classifiers' results are already captured in this notebook's own printed metrics), remove 3 unused imports, and fix two real bugs found while reading it: `sns.distplot` was removed from modern seaborn, and two cells have a copy-paste bug mislabeling the Logistic Regression confusion matrix/report as "XGBoost".

- [ ] **Step 1: Copy the existing notebook**

```bash
cp "/c/Users/Palan/OneDrive/Documents/computer_programming/Projects/2_machineLearning.ipynb" notebooks/2_classical_ml.ipynb
```

- [ ] **Step 2: Write and run a one-off patch script**

Create `scripts/_patch_notebook_2.py` (temporary — delete after use in Step 4):

```python
"""One-off patch script for notebooks/2_classical_ml.ipynb. Run once, then delete."""
import nbformat

path = "notebooks/2_classical_ml.ipynb"
nb = nbformat.read(path, as_version=4)
cells = nb["cells"]

# Cell 3: drop unused imports (XGBClassifier needs a package we don't otherwise
# need; LabelEncoder and HistGradientBoostingClassifier are never used)
cells[3]["source"] = (
    cells[3]["source"]
    .replace("from sklearn.preprocessing import LabelEncoder\n", "")
    .replace("from xgboost import XGBClassifier\n", "")
    .replace(
        "from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier",
        "from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier",
    )
)

# Cell 6: point at the new cleaned-data path
cells[6]["source"] = cells[6]["source"].replace(
    "df = pd.read_csv('Dataset/Cleaned Data/reviews.csv')",
    "df = pd.read_csv('../data/cleaned/reviews.csv')",
)

# Cell 12: sns.distplot was removed in modern seaborn; use histplot
cells[12]["source"] = cells[12]["source"].replace(
    "sns.distplot(reviewer_freq, kde=False)",
    "sns.histplot(reviewer_freq, kde=False)",
)

# Cell 28: fix mislabeled confusion-matrix title (copy-paste bug — this is
# the Logistic Regression matrix, not XGBoost, which isn't even used)
cells[28]["source"] = cells[28]["source"].replace(
    'plt.title("XGBoost Classifier - Confusion Matrix")',
    'plt.title("Logistic Regression - Confusion Matrix")',
)

# Cell 29: same copy-paste bug in the printed report label
cells[29]["source"] = cells[29]["source"].replace(
    'print("Classification Report For XGBoost Classifier:")',
    'print("Classification Report For Logistic Regression:")',
)

_NOTE = (
    "# Not saved to disk: this notebook trains and evaluates all 7 classifiers"
    " for comparison, but per the project's models-scope decision, only the"
    " TF-IDF vectorizer and the winning Gradient Boosting model (below) are"
    " committed — the dashboard only needs those two, and every model's"
    " metrics are already captured in this notebook's own printed output."
)

# Cells 30, 36, 41, 46, 51, 61: remove the joblib.dump calls for the 6
# classifiers we are not committing (Logistic Regression, AdaBoost,
# K-Neighbors, Decision Tree, Random Forest, SVC)
for idx in (30, 36, 41, 46, 51, 61):
    cells[idx]["source"] = _NOTE

# Cell 31: TF-IDF vectorizer -> new models/ path (this IS committed)
cells[31]["source"] = (
    "# save the vectorizer\n"
    "joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')"
)

# Cell 56: Gradient Boosting -> new models/ path (this IS committed — the
# dashboard's live predictor uses this model)
cells[56]["source"] = (
    "# save the gradient boosting model (the dashboard's live predictor uses this one)\n"
    "joblib.dump(gbc, '../models/gradient_boosting_model.pkl')"
)

nbformat.write(nb, path)
print("Patched notebooks/2_classical_ml.ipynb")
```

Run: `python scripts/_patch_notebook_2.py`
Expected: prints `Patched notebooks/2_classical_ml.ipynb`

- [ ] **Step 3: Spot-check the patch**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/2_classical_ml.ipynb', as_version=4)
print(nb['cells'][6]['source'])
print('---')
print(nb['cells'][31]['source'])
print('---')
print(nb['cells'][56]['source'])
"
```

Expected: cell 6 shows the new data path, cells 31/56 show the new `../models/` paths.

- [ ] **Step 4: Delete the one-off patch script**

```bash
rm scripts/_patch_notebook_2.py
```

- [ ] **Step 5: Execute the notebook end to end**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/2_classical_ml.ipynb
```

Expected: completes without error (dataset is ~11-12k rows, all 7 classical models train quickly — well under a minute total).

- [ ] **Step 6: Verify the committed model artifacts and record results**

```bash
python -c "
import joblib
v = joblib.load('models/tfidf_vectorizer.pkl')
m = joblib.load('models/gradient_boosting_model.pkl')
pred = m.predict(v.transform(['the host was wonderful and the flat was spotless']))
print('sample prediction:', pred[0])
assert pred[0] in {'negative', 'neutral', 'positive'}
print('OK')
"
```

Expected: prints a sentiment label and `OK`. Also open the notebook's Gradient Boosting cell output and note the actual accuracy/precision/recall/F1 — these real numbers (not Taimour's original 90.2%, since the data differs) go into the README results table in Task 10.

- [ ] **Step 7: Commit**

```bash
git add notebooks/2_classical_ml.ipynb models/tfidf_vectorizer.pkl models/gradient_boosting_model.pkl
git commit -m "Add notebook 2: classical ML (7 classifiers), fix seaborn/copy-paste bugs"
```

---

### Task 5: Notebook 3 — LSTM (fix vocab-mismatch bug)

**Files:**
- Create: `notebooks/3_deep_learning_lstm.py` (jupytext source) → build → `.ipynb`

**Interfaces:**
- Consumes: `data/cleaned/reviews.csv` (Task 3).
- Produces: nothing consumed by later tasks (weights intentionally not committed per Global Constraints) — this task's deliverable is a notebook that demonstrably learns (loss decreases, accuracy clearly above the ~33% random-guess baseline seen in the original broken version).

The original (`3_deepLearning.ipynb`, pulled from the source repo) never learns: training loss sits flat at `ln(3) ≈ 1.0986` for all 5 epochs, test accuracy 32.7%. Root cause: `Tokenizer(num_words=500, ...)` caps the vocabulary far below the `Embedding(5000, ...)` layer's declared size, so most tokens end up out-of-vocabulary. Fix: match the two, and right-size the model for CPU-only training per the earlier compute decision.

- [ ] **Step 1: Write `notebooks/3_deep_learning_lstm.py`**

```python
# %% [markdown]
# # 3. Deep Learning: LSTM
#
# Adapted from Taimour Karim's original `3_deepLearning.ipynb` (see README
# Credits). **The original never learned** — training loss sat flat at
# `ln(3) ≈ 1.0986` (the entropy of pure random 3-class guessing) for all 5
# epochs, test accuracy 32.7%. Root cause: `Tokenizer(num_words=500)` capped
# the vocabulary far below the `Embedding(5000, ...)` layer's declared size,
# so most tokens were out-of-vocabulary. This version matches vocabulary
# sizes and right-sizes the model for CPU-only training.

# %%
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv("../data/cleaned/reviews.csv")
df = df.dropna()
print(df.shape)

# %% [markdown]
# ## Prepare sequences

# %%
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)


df["processed_comments"] = df["cleaned_comments"].apply(preprocess_text)

# %%
VOCAB_SIZE = 5000  # matches the Embedding layer below — this is the bug fix

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["processed_comments"])
sequences = tokenizer.texts_to_sequences(df["processed_comments"])

# %%
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# %%
labels = pd.get_dummies(df["sentiment"]).values

# %%
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# %% [markdown]
# ## Model
#
# Reduced embedding/LSTM size vs. the original (128/64 -> 64/32) — the
# dataset is small (~12k rows), a smaller model trains faster on CPU without
# needing the extra capacity.

# %%
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 64, input_length=max_length))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# %%
early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# %% [markdown]
# ## Evaluate
#
# Sanity check: with 3 balanced classes, a model that learned nothing would
# plateau at ~33% accuracy and a flat loss around `ln(3) ≈ 1.099` — exactly
# what the original, buggy notebook showed. This version should clearly beat
# that baseline.

# %%
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}  (random-guess baseline: {np.log(3):.4f})")
print(f"Test Accuracy: {accuracy * 100:.2f}%  (random-guess baseline: 33.3%)")
assert accuracy > 0.40, "Model is not clearly beating the random-guess baseline — investigate before proceeding"
```

- [ ] **Step 2: Build the notebook**

```bash
jupytext --to notebook notebooks/3_deep_learning_lstm.py
```

- [ ] **Step 3: Execute the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/3_deep_learning_lstm.ipynb
```

Expected: completes without error, no assertion failure. If the final `assert` trips, the model is still not learning — re-check the vocabulary size, or the `processed_comments` column for degenerate (near-empty) text before concluding the bug fix is complete. Note the actual final test accuracy for the README.

- [ ] **Step 4: Commit**

```bash
git add notebooks/3_deep_learning_lstm.py notebooks/3_deep_learning_lstm.ipynb
git commit -m "Add notebook 3: LSTM, fixed vocabulary-size bug that prevented learning"
```

---

### Task 6: Notebook 4 — DistilBERT (fix optimizer bug)

**Files:**
- Create: `notebooks/4_llm_bert.py` (jupytext source) → build → `.ipynb`

**Interfaces:**
- Consumes: `data/cleaned/reviews.csv` (Task 3).
- Produces: nothing consumed by later tasks (weights not committed) — deliverable is a notebook that demonstrably learns.

The original (`4_LLM.ipynb`) has the identical flat-loss/32.7%-accuracy failure as notebook 3, but a different, identifiable root cause: it builds a correctly-configured `Adam(learning_rate=2e-5, ...)` optimizer object but then calls `model.compile(optimizer='adam', ...)`, passing the *string* instead of that object — silently falling back to Keras's default learning rate (0.001), roughly 50x too high for BERT-family fine-tuning. Fix: pass the actual optimizer object. Also swap `bert-base-uncased` → `distilbert-base-uncased` for CPU speed (per the compute decision), and add early stopping (the original had none, unlike notebook 3).

- [ ] **Step 1: Write `notebooks/4_llm_bert.py`**

```python
# %% [markdown]
# # 4. LLM: DistilBERT fine-tuning
#
# Adapted from Taimour Karim's original `4_LLM.ipynb` (see README Credits).
# **The original never learned** — identical symptom to notebook 3 (flat loss
# at `ln(3) ≈ 1.0986`, 32.7% test accuracy), but a different root cause:
# `model.compile(optimizer='adam', ...)` passed the *string* `'adam'` instead
# of the `Adam(learning_rate=2e-5, ...)` object built one line earlier —
# silently using Keras's default LR (0.001), ~50x too high for BERT-family
# fine-tuning. This version passes the actual optimizer object, swaps to
# DistilBERT (faster on CPU, per project's compute decision), and adds early
# stopping.

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv("../data/cleaned/reviews.csv")
df = df.dropna()
print(df.shape)

# %% [markdown]
# ## Tokenize

# %%
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

tokens = tokenizer.batch_encode_plus(
    df["cleaned_comments"].tolist(),
    max_length=64,  # reviews are short; reduced from 128 for CPU speed
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors="tf",
)

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# %%
df["sentiment"] = df["sentiment"].astype("category")
labels = df["sentiment"].cat.codes

# %%
input_ids_np = input_ids.numpy()
attention_mask_np = attention_mask.numpy()
labels_np = labels.to_numpy()

train_input_ids, test_input_ids, train_attention_mask, test_attention_mask, train_labels, test_labels = train_test_split(
    input_ids_np, attention_mask_np, labels_np, test_size=0.2, random_state=42
)

train_input_ids = tf.convert_to_tensor(train_input_ids)
train_attention_mask = tf.convert_to_tensor(train_attention_mask)
train_labels = tf.cast(tf.convert_to_tensor(train_labels), dtype=tf.int32)

test_input_ids = tf.convert_to_tensor(test_input_ids)
test_attention_mask = tf.convert_to_tensor(test_attention_mask)
test_labels = tf.cast(tf.convert_to_tensor(test_labels), dtype=tf.int32)

# %% [markdown]
# ## Model
#
# The bug fix is in the `compile()` call below: passing the `optimizer`
# object, not the string `'adam'`.

# %%
model = TFDistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(df["sentiment"].cat.categories)
)

optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)
model.compile(
    optimizer=optimizer,  # was the string 'adam' in the original — the bug
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# %%
early_stopping = EarlyStopping(monitor="val_loss", patience=1, verbose=1)

history = model.fit(
    {"input_ids": train_input_ids, "attention_mask": train_attention_mask},
    train_labels,
    batch_size=16,
    epochs=4,
    validation_split=0.1,
    callbacks=[early_stopping],
)

# %% [markdown]
# ## Evaluate
#
# Same sanity check as notebook 3: a model that learned nothing plateaus at
# ~33% accuracy / `ln(3) ≈ 1.099` loss — what the original buggy notebook
# showed. This version should clearly beat that baseline.
#
# If training is too slow on your machine, further reduce `max_length`,
# `epochs`, or subsample `df` before tokenizing — the goal is proving the
# optimizer bug fix works (loss decreasing, accuracy clearly above random),
# not maximizing accuracy.

# %%
result = model.evaluate(
    x={"input_ids": test_input_ids, "attention_mask": test_attention_mask},
    y=test_labels,
)
print(f"Test loss: {result[0]:.4f}  (random-guess baseline: {np.log(3):.4f})")
print(f"Test accuracy: {result[1] * 100:.2f}%  (random-guess baseline: 33.3%)")
assert result[1] > 0.40, "Model is not clearly beating the random-guess baseline — investigate before proceeding"
```

- [ ] **Step 2: Build the notebook**

```bash
jupytext --to notebook notebooks/4_llm_bert.py
```

- [ ] **Step 3: Execute the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/4_llm_bert.ipynb
```

Expected: completes without error, no assertion failure. This is the slowest notebook — DistilBERT fine-tuning on CPU; expect it to take real wall-clock time (could be 10+ minutes depending on hardware). If it's impractically slow, reduce `max_length` further or subsample `df` before tokenizing (noted in the notebook itself). Note the actual final test accuracy for the README.

- [ ] **Step 4: Commit**

```bash
git add notebooks/4_llm_bert.py notebooks/4_llm_bert.ipynb
git commit -m "Add notebook 4: DistilBERT, fixed optimizer-object bug that prevented learning"
```

---

### Task 7: Dashboard data module (`dashboard/data.py`)

**Files:**
- Create: `dashboard/data.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `data/cleaned/reviews.csv`, `data/raw_sample.csv`, `data/user_reviews.csv`, `models/tfidf_vectorizer.pkl`, `models/gradient_boosting_model.pkl` (all from Tasks 3-4).
- Produces: `load_data() -> pd.DataFrame`, `load_raw_sample() -> pd.DataFrame`, `load_models() -> (vectorizer, model)`, `predict_sentiment(text, vectorizer, model) -> str`, `save_user_review(name, review, sentiment) -> None`. Consumed by Task 9 (`dashboard/app.py`).

- [ ] **Step 1: Create the seed `data/user_reviews.csv`**

```bash
echo "name,review,sentiment" > data/user_reviews.csv
```

- [ ] **Step 2: Write the failing tests — `tests/test_pipeline.py`**

```python
import pandas as pd
import pytest

from dashboard.data import load_data, load_models, load_raw_sample, predict_sentiment


def test_load_data_returns_expected_columns():
    df = load_data()
    expected = {
        "date", "reviewer_name", "comments",
        "cleaned_comments", "polarity", "sentiment",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_load_data_has_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_load_raw_sample_returns_pre_cleaning_columns():
    df = load_raw_sample()
    assert "comments" in df.columns
    assert len(df) > 0


def test_load_models_returns_fitted_vectorizer_and_classifier():
    vectorizer, model = load_models()
    assert hasattr(vectorizer, "transform")
    assert hasattr(model, "predict")


def test_predict_sentiment_returns_valid_label():
    vectorizer, model = load_models()
    result = predict_sentiment(
        "The host was amazing and the room was spotless", vectorizer, model
    )
    assert result in {"negative", "neutral", "positive"}


def test_predict_sentiment_rejects_empty_string():
    vectorizer, model = load_models()
    with pytest.raises(ValueError):
        predict_sentiment("", vectorizer, model)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.data'`

- [ ] **Step 4: Write `dashboard/data.py`**

```python
"""Data and model loading for the sentiment dashboard."""
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned" / "reviews.csv"
RAW_SAMPLE_PATH = BASE_DIR / "data" / "raw_sample.csv"
USER_REVIEWS_PATH = BASE_DIR / "data" / "user_reviews.csv"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"


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
    features = vectorizer.transform([text])
    return model.predict(features)[0]


def save_user_review(name: str, review: str, sentiment: str) -> None:
    """Append a dashboard-submitted review to the user reviews CSV."""
    row = pd.DataFrame({"name": [name], "review": [review], "sentiment": [sentiment]})
    row.to_csv(USER_REVIEWS_PATH, mode="a", header=False, index=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add dashboard/data.py tests/test_pipeline.py data/user_reviews.csv
git commit -m "Add dashboard data module with tests"
```

---

### Task 8: Dashboard charts module (`dashboard/charts.py`)

**Files:**
- Create: `dashboard/charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: a `pd.DataFrame` with columns `reviewer_name, sentiment, cleaned_comments` (matches `load_data()` output from Task 7).
- Produces: `create_graph`, `create_graph2`, `circle`, `time_series`, `time_sentiment`, `generate_wordcloud` — each `(df) -> plotly.graph_objects.Figure`. Consumed by Task 9 (`dashboard/app.py`).

The original `charts.py` (pulled from source repo) has three real bugs beyond cosmetics: `circle()` and the dropped `bubble()`/`polar_area()` return hardcoded fake data instead of computing anything from `df` (per the models-scope-adjacent decision to remove fake-data stubs — `circle()` gets the same fix rather than staying inconsistent); `create_graph2()` claims to show each top reviewer's positive/neutral/negative split but actually plots the *same* total-count values three times in different colors — it never groups by sentiment at all; and a module-level function is named `wordcloud`, shadowing the imported `wordcloud` package it needs. All three are fixed below. Every chart also had a literal `<b>text<b>` HTML bug (missing closing slash) — fixed throughout.

- [ ] **Step 1: Write the failing tests — `tests/test_charts.py`**

```python
import pandas as pd
import plotly.graph_objects as go

from dashboard.charts import (
    circle,
    create_graph,
    create_graph2,
    generate_wordcloud,
    time_sentiment,
    time_series,
)


def _sample_df():
    return pd.DataFrame({
        "reviewer_name": ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"],
        "sentiment": ["positive", "negative", "neutral", "positive", "positive", "negative"],
        "cleaned_comments": [
            "great stay", "bad service", "ok place",
            "lovely host", "would return", "too noisy",
        ],
        "date": [
            "2024-01-15", "2024-02-03", "2024-02-20",
            "2024-03-01", "2024-03-10", "2024-03-15",
        ],
    })


def test_create_graph_returns_a_figure():
    fig = create_graph(_sample_df())
    assert isinstance(fig, go.Figure)


def test_create_graph2_splits_counts_by_sentiment_not_duplicated_totals():
    fig = create_graph2(_sample_df())
    trace_names = {trace.name for trace in fig.data}
    assert trace_names.issubset({"Positive", "Neutral", "Negative"})
    # regression check for the original bug: traces must NOT all be identical
    values_per_trace = [tuple(trace.y) for trace in fig.data]
    assert len(set(values_per_trace)) > 1


def test_circle_uses_real_sentiment_counts_not_hardcoded_values():
    df = _sample_df()
    fig = circle(df)
    values = sorted(fig.data[0].values)
    expected = sorted(df["sentiment"].value_counts().tolist())
    assert values == expected


def test_generate_wordcloud_returns_a_figure():
    fig = generate_wordcloud(_sample_df())
    assert isinstance(fig, go.Figure)


def test_time_series_returns_one_point_per_month_present():
    fig = time_series(_sample_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 3  # Jan, Feb, Mar in the sample


def test_time_sentiment_returns_one_trace_per_sentiment_seen():
    fig = time_sentiment(_sample_df())
    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"positive", "negative", "neutral"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_charts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.charts'`

- [ ] **Step 3: Write `dashboard/charts.py`**

```python
"""Chart-building functions for the sentiment dashboard."""
import string

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

_LAYOUT_DEFAULTS = dict(
    xaxis_title="<b>Reviewer Name</b>",
    yaxis_title="<b>Frequency</b>",
    font=dict(size=14, family="Arial", color="black"),
    hoverlabel=dict(font_size=14, font_family="Rockwell"),
    plot_bgcolor="#f7f7f7",
    paper_bgcolor="#f7f7f7",
    margin=dict(l=0, r=0, t=30, b=0),
)


def create_graph(df: pd.DataFrame) -> go.Figure:
    """Bar chart of the top 5 reviewers by review count."""
    top5 = df["reviewer_name"].value_counts().head(5)
    fig = px.bar(
        top5, x=top5.index, y=top5.values,
        color=top5.values, color_continuous_scale="RdBu",
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor="black")
    fig.update_yaxes(showgrid=False, showline=False, zeroline=False, linecolor="black")
    return fig


def create_graph2(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of each top-5 reviewer's reviews, split by sentiment.

    The original version plotted the same total-count values three times
    under different colors instead of actually grouping by sentiment — fixed
    here to compute a real per-reviewer, per-sentiment breakdown.
    """
    top_reviewers = df["reviewer_name"].value_counts().head(5).index
    subset = df[df["reviewer_name"].isin(top_reviewers)]
    counts = subset.groupby(["reviewer_name", "sentiment"]).size().unstack(fill_value=0)
    counts = counts.reindex(top_reviewers)

    colors = {"positive": "#660000", "neutral": "#800020", "negative": "#C41E3A"}
    fig = go.Figure(data=[
        go.Bar(name=sentiment.capitalize(), x=counts.index, y=counts[sentiment],
               marker_color=colors[sentiment])
        for sentiment in ["positive", "neutral", "negative"]
        if sentiment in counts.columns
    ])
    fig.update_layout(barmode="stack", **_LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor="black")
    fig.update_yaxes(showgrid=False, showline=False, zeroline=False, linecolor="black")
    return fig


def generate_wordcloud(df: pd.DataFrame) -> go.Figure:
    """Treemap word cloud of the top 100 most common words across all reviews.

    Renamed from the original's `wordcloud`, which shadowed the imported
    `wordcloud` package at module scope.
    """
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    text = df["cleaned_comments"].astype(str).str.lower()
    text = text.str.replace(f"[{string.punctuation}]", " ", regex=True)
    words = " ".join(text).split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]

    counts = pd.Series(words).value_counts().head(100)
    word_df = pd.DataFrame({"word": counts.index, "count": counts.values})

    fig = px.treemap(
        word_df, path=["word"], values="count",
        color="count", color_continuous_scale="RdBu",
    )
    fig.update_layout(**{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "xaxis_title" and k != "yaxis_title"})
    return fig


def circle(df: pd.DataFrame) -> go.Figure:
    """Hollow donut chart of the real sentiment distribution.

    The original hardcoded [40, 30, 30] instead of computing anything from
    `df` — fixed here to use actual sentiment counts.
    """
    counts = df["sentiment"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=[label.capitalize() for label in counts.index],
        values=counts.values,
        hole=0.6,
    )])
    fig.update_layout(**{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "xaxis_title" and k != "yaxis_title"})
    return fig


def time_series(df: pd.DataFrame) -> go.Figure:
    """Line chart of review volume per month."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    by_month = df.groupby("year_month").size().reset_index(name="counts")
    fig = px.line(by_month, x="year_month", y="counts")
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


def time_sentiment(df: pd.DataFrame) -> go.Figure:
    """Line chart of positive/neutral/negative review volume per month."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["year_month"] = df["date"].dt.strftime("%Y-%m")

    fig = go.Figure()
    for sentiment in ["positive", "negative", "neutral"]:
        by_month = (
            df[df["sentiment"] == sentiment]
            .groupby("year_month")
            .size()
            .reset_index(name="counts")
        )
        fig.add_trace(go.Scatter(
            x=by_month["year_month"], y=by_month["counts"],
            mode="lines+markers", name=sentiment,
        ))
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_charts.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/charts.py tests/test_charts.py
git commit -m "Add dashboard charts module, fix fake-data and sentiment-grouping bugs"
```

---

### Task 9: Dashboard app (`dashboard/app.py`)

**Files:**
- Create: `dashboard/app.py`

**Interfaces:**
- Consumes: everything from `dashboard/data.py` (Task 7) and `dashboard/charts.py` (Task 8).
- Produces: a runnable Dash application. Nothing downstream consumes this — it's the final user-facing artifact, verified by manual smoke test (no automated Dash callback tests — out of scope per the design spec).

Adapted from the original `dashboard.py`. Fixes: uses `dashboard/data.py` instead of inline path-coupled loading, points the live "submit a review" predictor at the Gradient Boosting model (the original used Logistic Regression, inconsistent with which model actually won in notebook 2), and the "Uncleaned Data" view now uses the small committed `raw_sample.csv` instead of a 100k-row read from a file we don't keep around.

- [ ] **Step 1: Write `dashboard/app.py`**

```python
"""Dash app: browse the cleaned/raw review data, sentiment breakdowns, and
submit a review to get a live sentiment prediction from the winning
Gradient Boosting model."""
import dash
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from dashboard.charts import (
    circle,
    create_graph,
    create_graph2,
    generate_wordcloud,
    time_sentiment,
    time_series,
)
from dashboard.data import load_data, load_models, load_raw_sample, predict_sentiment, save_user_review

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Airbnb Review Sentiment Dashboard"

df_cleaned = load_data()
df_raw_sample = load_raw_sample()
vectorizer, model = load_models()


def _table(dataframe):
    return go.Figure(data=[go.Table(
        header=dict(values=list(dataframe.columns), fill_color="paleturquoise", align="left"),
        cells=dict(values=[dataframe[col] for col in dataframe.columns], fill_color="lavender", align="left"),
    )])


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(
        style={"width": "13%", "float": "left", "height": "100vh", "background-color": "#f8f8f8"},
        children=[
            html.Button("Home", id="home-button", n_clicks=0, style={"width": "90%", "margin": "5px"}),
            html.Button("Uncleaned Data", id="uncleaned-button", n_clicks=0, style={"width": "90%", "margin": "5px"}),
            html.Button("Cleaned Data", id="cleaned-button", n_clicks=0, style={"width": "90%", "margin": "5px"}),
            html.Button("Sentiment Analysis", id="sentimental-button", n_clicks=0, style={"width": "90%", "margin": "5px"}),
            html.Button("Provide Review", id="review-button", n_clicks=0, style={"width": "90%", "margin": "5px"}),
        ],
    ),
    html.Div(style={"width": "80%", "float": "right"}, id="page-content"),
])


@app.callback(
    Output("page-content", "children"),
    [
        Input("home-button", "n_clicks"),
        Input("uncleaned-button", "n_clicks"),
        Input("cleaned-button", "n_clicks"),
        Input("sentimental-button", "n_clicks"),
        Input("review-button", "n_clicks"),
    ],
)
def display_page(*_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([html.H3("Welcome to the Home page.")])

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "uncleaned-button":
        return html.Div([html.H3("Uncleaned Data (sample)"), dcc.Graph(figure=_table(df_raw_sample.head(10)))])
    if button_id == "cleaned-button":
        return html.Div([html.H3("Cleaned Data"), dcc.Graph(figure=_table(df_cleaned.head(10)))])
    if button_id == "sentimental-button":
        return html.Div([
            html.H3("Sentiment Analysis"),
            dcc.Dropdown(
                id="visualization-choices",
                options=[
                    {"label": "Sentiment Distribution (donut)", "value": "circle"},
                    {"label": "Word Cloud", "value": "cloud"},
                    {"label": "Top Reviewers", "value": "bar"},
                    {"label": "Top Reviewers by Sentiment", "value": "bar2"},
                    {"label": "Reviews Over Time", "value": "time"},
                    {"label": "Sentiment Over Time", "value": "time_sentiment"},
                ],
                value="circle",
            ),
            html.Div(id="visualization-output"),
        ])
    if button_id == "review-button":
        return html.Div([
            html.H3("Provide Review"),
            dcc.Input(id="name-input", type="text", placeholder="Enter your name", style={"margin": "10px"}),
            html.Br(), html.Br(),
            dcc.Textarea(id="review-input", placeholder="Enter your review", style={"width": "50%", "height": "200px"}),
            html.Br(), html.Br(),
            html.Button("Submit", id="submit-review", n_clicks=0),
            html.Div(id="review-output"),
        ])
    return html.Div([html.H3("Welcome to the Home page.")])


@app.callback(Output("visualization-output", "children"), [Input("visualization-choices", "value")])
def update_visualization(choice):
    if choice == "circle":
        return dcc.Graph(figure=circle(df_cleaned))
    if choice == "cloud":
        return dcc.Graph(figure=generate_wordcloud(df_cleaned))
    if choice == "bar":
        return dcc.Graph(figure=create_graph(df_cleaned))
    if choice == "bar2":
        return dcc.Graph(figure=create_graph2(df_cleaned))
    if choice == "time":
        return dcc.Graph(figure=time_series(df_cleaned))
    if choice == "time_sentiment":
        return dcc.Graph(figure=time_sentiment(df_cleaned))
    return html.Div()


@app.callback(
    Output("review-output", "children"),
    [Input("submit-review", "n_clicks")],
    [State("name-input", "value"), State("review-input", "value")],
)
def submit_review(n_clicks, name, review):
    if not n_clicks:
        return None
    if not review:
        return "Please enter a review before submitting."
    sentiment = predict_sentiment(review, vectorizer, model)
    save_user_review(name or "Anonymous", review, sentiment)
    return f"Thanks! Predicted sentiment: {sentiment}"


if __name__ == "__main__":
    app.run(debug=True)
```

- [ ] **Step 2: Manual smoke test**

```bash
python -m dashboard.app
```

Expected: server starts on `http://127.0.0.1:8050` with no exceptions. Open it in a browser, click through Home / Uncleaned Data / Cleaned Data / Sentiment Analysis (try all six dropdown options, including "Reviews Over Time" and "Sentiment Over Time") / Provide Review (submit a sample review, confirm it returns a sentiment and appends to `data/user_reviews.csv`). Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "Add dashboard app, wire live prediction to Gradient Boosting model"
```

---

### Task 10: README, final verification, and GitHub push

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: actual results recorded from Tasks 4-6 (classical ML metrics, LSTM accuracy, DistilBERT accuracy) — insert the real numbers observed when those notebooks were executed, not placeholders.

- [ ] **Step 1: Write `README.md`**

```markdown
# Airbnb Review Sentiment Analysis

Classifies Airbnb reviews as negative / neutral / positive, comparing 7
classical ML models against fine-tuned LSTM and DistilBERT approaches, with
an interactive Dash dashboard for exploring the results.

## Credits / Origin

This project completes, debugs, and documents a public tutorial pipeline by
**Taimour Karim** — [tkarim45/Beginner-Data-Science-Projects](https://github.com/tkarim45/Beginner-Data-Science-Projects)
(`NLP/AirBnb reviews Sentimental Analysis/`, MIT licensed). It is not
original research. What this project adds on top of the original:

- Sourced training data independently from [InsideAirbnb](https://insideairbnb.com/)
  (the original repo's raw dataset wasn't available), so the results here are
  self-produced, not copied.
- Found and fixed two real bugs that meant the deep learning and LLM stages
  never actually learned anything (see "What was broken" below).
- Got the full pipeline — preprocessing, classical ML, LSTM, DistilBERT, and
  the dashboard — actually running end to end, with tests.

## Pipeline

1. **`notebooks/1_data_preprocessing.ipynb`** — downloads InsideAirbnb London
   review data, cleans text, labels sentiment via TextBlob polarity, balances
   classes.
2. **`notebooks/2_classical_ml.ipynb`** — TF-IDF features, 7 classifiers
   (Logistic Regression, AdaBoost, KNN, Decision Tree, Random Forest,
   Gradient Boosting, SVC).
3. **`notebooks/3_deep_learning_lstm.ipynb`** — LSTM on tokenized/padded
   sequences.
4. **`notebooks/4_llm_bert.ipynb`** — fine-tuned DistilBERT.
5. **`dashboard/`** — Dash app: browse the data, sentiment breakdowns, and
   submit a review for a live prediction.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | _fill in from notebook 2 output_ | | | |
| AdaBoost | _fill in_ | | | |
| K-Neighbors | _fill in_ | | | |
| Decision Tree | _fill in_ | | | |
| Random Forest | _fill in_ | | | |
| **Gradient Boosting (winner)** | _fill in_ | | | |
| SVC | _fill in_ | | | |
| LSTM | _fill in from notebook 3_ | — | — | — |
| DistilBERT | _fill in from notebook 4_ | — | — | — |

Note: these numbers come from a different InsideAirbnb data snapshot than
Taimour's original run, so they won't exactly match his reported figures —
expected, and arguably more legitimate since they're numbers this project
actually produced.

### Why classical ML beat deep learning here

Once both bugs below were fixed, LSTM and DistilBERT still underperformed
the classical TF-IDF + Gradient Boosting approach. Plausible reasons: the
balanced dataset is small (~12k rows) relative to what deep models typically
need; the underlying reviews are multilingual (German, Spanish, and other
languages observed) despite English-only preprocessing; and the TextBlob
polarity → `pd.cut` equal-width-bin labeling is a fairly crude sentiment
label to begin with. Classical models with high-dimensional TF-IDF features
may partly be picking up surface keyword patterns rather than deeper
semantic signal — worth keeping in mind before trusting the 90%+ accuracy
number too far.

### What was broken (and fixed)

- **LSTM (`3_deep_learning_lstm.ipynb`):** the original capped the tokenizer
  vocabulary at 500 words while the embedding layer was sized for 5000 —
  most tokens ended up out-of-vocabulary, and the model never learned
  (flat loss at `ln(3)`, 32.7% test accuracy — at random-guess baseline).
  Fixed by matching the two.
- **DistilBERT (`4_llm_bert.ipynb`):** the original built a correctly
  configured `Adam(learning_rate=2e-5)` optimizer object, then passed the
  *string* `'adam'` to `compile()` instead — silently using Keras's default
  learning rate (50x too high for BERT-family fine-tuning), with the
  identical failure symptom. Fixed by passing the actual optimizer object.
- Notebook 2 also had a `sns.distplot` call (removed in modern seaborn) and
  a copy-paste bug mislabeling a Logistic Regression confusion matrix as
  "XGBoost" — both fixed.
- `dashboard/charts.py` had two chart functions returning hardcoded fake
  data instead of computing anything from the real data, and one that
  claimed to group by sentiment but actually plotted the same totals three
  times — all three fixed with tests covering the fix.

## Known limitations

- The sentiment labels come from TextBlob polarity cut into 3 equal-width
  bins — a crude heuristic, not verified against human-labeled ground truth.
- The dataset includes non-English reviews despite English-only
  preprocessing (stopwords, lemmatization, stemming).
- Notebook 1 can only be re-run from the InsideAirbnb raw download onward —
  the exact raw file the original pipeline used was never available.

## Setup

```bash
git clone <this-repo-url>
cd airbnb-review-sentiment-analysis
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']]"
```

## Running it

The cleaned dataset and the two models the dashboard needs are already
committed, so you can go straight to:

```bash
python -m dashboard.app
# open http://127.0.0.1:8050
```

To reproduce the full pipeline from raw data (slow — real download +
training time):

```bash
python scripts/download_raw_data.py
jupyter nbconvert --to notebook --execute --inplace notebooks/1_data_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/2_classical_ml.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3_deep_learning_lstm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_llm_bert.ipynb
```

## Tests

```bash
pytest
```

## License

MIT — see `LICENSE`. Portions originally authored by Taimour Karim (2024-present).
```

- [ ] **Step 2: Fill in the actual results table**

Pull the real numbers from the executed notebooks (Tasks 4-6 outputs) and replace every `_fill in..._` cell with the actual observed values. This is not optional — do not leave placeholder text in the committed README.

- [ ] **Step 3: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass (from Tasks 2, 3, 7, 8 — text cleaning, balancing, download decompress, pipeline, charts).

- [ ] **Step 4: Commit the README**

```bash
git add README.md
git commit -m "Add README with credits, results, and setup instructions"
```

- [ ] **Step 5: Create the GitHub repo and push**

Confirm with the user before this step (creates a public, externally-visible artifact under their account).

```bash
gh repo create airbnb-review-sentiment-analysis --public --source=. --remote=origin
git push -u origin main
```

- [ ] **Step 6: Verify**

```bash
gh repo view --web
```

Confirm the repo is visible, README renders correctly on GitHub (check the Credits section is prominent), and `.gitignore` correctly kept `data/raw/` out of the pushed history (`git ls-files data/raw/` should show nothing but `.gitkeep`).
