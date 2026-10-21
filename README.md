# Airbnb Review Sentiment Analysis

Classifies Airbnb reviews as negative / neutral / positive, comparing 7
classical ML models against fine-tuned LSTM and DistilBERT approaches. An
interactive Dash dashboard lets you browse the data, see sentiment
breakdowns, and score a new review live (see "Dashboard" under Running It).

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
- Got the notebook pipeline — preprocessing, classical ML, LSTM, and
  DistilBERT — actually running end to end, and fixed the dashboard built
  on top of it.

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
   submit a review for a live prediction from the Gradient Boosting model.

## Results

Numbers below are from this project's own run, on 19,773 balanced reviews
(6,591 per class) sourced independently from InsideAirbnb — not copied from
Taimour's original ~90.2%, though Gradient Boosting lands close to that by
coincidence. The 7 classical models report accuracy/precision/recall/F1 on a
3,955-row held-out test set; LSTM and DistilBERT only print test
accuracy/loss in their notebooks (LSTM: 0.3596 test loss; DistilBERT: 0.5577
test loss — random-guess baseline for both is `ln(3) ≈ 1.0986`), so their
precision/recall/F1 columns are blank below. "Accuracy" throughout means
agreement with the TextBlob-derived labels, not with human judgment — see
"Known limitations".

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 88.52% | 88.57% | 88.52% | 88.47% |
| AdaBoost | 75.15% | 78.45% | 75.15% | 75.87% |
| K-Neighbors | 64.40% | 70.52% | 64.40% | 64.97% |
| Decision Tree | 87.16% | 87.13% | 87.16% | 87.12% |
| Random Forest | 83.39% | 83.37% | 83.39% | 82.98% |
| **Gradient Boosting (winner)** | 89.89% | 90.02% | 89.89% | 89.93% |
| SVC | 88.19% | 88.28% | 88.19% | 88.14% |
| LSTM | 88.12% | — | — | — |
| DistilBERT | 80.38% | — | — | — |

### Classical ML vs. deep learning here

Not a clean "classical wins" story. Gradient Boosting takes the top spot
(89.89%), but the once-completely-broken LSTM (see "What was broken" below)
comes in a close second at 88.12% — beating 5 of the 7 classical models
outright, including Random Forest. DistilBERT, the most sophisticated model
here, finishes last among the credible models at 80.38%.

The most likely reason DistilBERT underperforms: both notebooks 3 and 4 feed
their models `cleaned_comments` — text that's already had stopwords stripped
and every word stemmed (e.g. "housed" → "hous"). For the LSTM, which builds
its vocabulary from scratch, that's harmless and even shrinks the vocabulary
usefully. For DistilBERT, whose entire advantage comes from a subword
tokenizer and embeddings pretrained on fluent natural English, feeding it
stemmed non-words and stripped sentence structure likely destroys much of
what pretraining gave it — a pretrained LLM this small dataset probably
needed *raw* text to actually pay for itself. Not proven here (would need a
rerun on uncleaned text to confirm), but it's consistent with DistilBERT
losing to both a from-scratch LSTM and to plain TF-IDF logistic regression
despite being the most capable model in the lineup.

Other reasons not to over-read the leaderboard order: the balanced dataset is
still small (~19.8k rows) relative to what deep models typically need; the
underlying reviews are multilingual (German, Spanish, and other languages
observed) despite English-only preprocessing; and the TextBlob polarity →
`pd.cut` equal-width-bin labeling is a fairly crude sentiment label to begin
with. Classical models with high-dimensional TF-IDF features may partly be
picking up surface keyword patterns rather than deeper semantic signal —
worth keeping in mind before trusting any of these accuracy numbers too far.

### What was broken (and fixed)

- **LSTM (`3_deep_learning_lstm.ipynb`):** two independent bugs, both found
  by actually running the notebook rather than just reading the code. (1)
  The original capped the tokenizer vocabulary at 500 words while the
  embedding layer was sized for 5000 — most tokens ended up
  out-of-vocabulary. (2) Sequence length was set to the single *longest*
  review in the dataset (711 tokens), so a typical ~34-word review ended up
  ~95% padding, and the embedding layer had no `mask_zero=True` — the LSTM
  couldn't tell real content from padding, so predictions collapsed to
  nearly the same output regardless of input. Fixing bug (1) alone wasn't
  enough — the model still failed (flat loss at `ln(3)`, 32.7-32.8% test
  accuracy — at random-guess baseline) until bug (2) was also fixed by
  capping sequence length at the 95th percentile and adding `mask_zero=True`.
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
  times — all three fixed.
- The dashboard's live-prediction form fed raw review text to a TF-IDF
  vectorizer that was fitted on stemmed, stopword-stripped text, so most
  words weren't in its vocabulary. Found during smoke testing; fixed by
  running notebook 1's cleaning on the text first (`clean_text` in
  `dashboard/data.py`). On a 1,500-row sample of the dataset, agreement with
  the labels went from 74.7% to 90.1% (English-looking rows: 67.4% to
  85.0%). That sample came from the full dataset, so ~80% of its rows were
  seen in training — read the two figures as a before/after comparison, not
  as test accuracy.

## Known limitations

- The sentiment labels come from TextBlob polarity cut into 3 equal-width
  bins — a crude heuristic, not verified against human-labeled ground truth.
- The dataset includes non-English reviews despite English-only
  preprocessing (stopwords, lemmatization, stemming).
- The labels are confounded with language. In a rough check, only ~26% of
  "negative" rows contain the English words "the" or "and", versus ~61% of
  neutral and ~77% of positive rows, and the first rows of the negative class
  are German, Spanish and Norwegian reviews that read as positive. TextBlob
  uses an English lexicon, so it likely mis-scores foreign text (mechanism
  not verified). Classifier scores on the negative class are therefore partly
  language detection, and the dashboard's live form tends to call clearly
  negative English reviews "neutral".
- Notebook 1 can only be re-run from the InsideAirbnb raw download onward —
  the exact raw file the original pipeline used was never available.
- `scikit-learn` is pinned in `requirements.txt` to the exact version
  `models/*.pkl` were pickled with. Scikit-learn doesn't guarantee pickle
  compatibility across releases — an earlier version of this project upgraded
  scikit-learn without re-pickling and the dashboard failed outright
  (`ModuleNotFoundError: No module named '_loss'`) loading the stale model.
  If you retrain (re-run notebook 2) under a different scikit-learn version,
  update this pin to match.

## Setup

```bash
git clone <this-repo-url>
cd airbnb-review-sentiment-analysis
python -m venv .venv        # or any location you prefer
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']]"
```

(If you hit a Windows "path too long" error installing tensorflow/transformers,
create the venv somewhere with a short path instead, e.g. `C:\venvs\airbnb`,
rather than nested deep inside a cloned-repo path — this is a Windows
`MAX_PATH` limitation, not a project bug.)

## Running it

`data/cleaned/reviews.csv` is already committed, so you can start at notebook 2:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/2_classical_ml.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3_deep_learning_lstm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_llm_bert.ipynb
```

(or open them in Jupyter/VS Code and run cells interactively). Notebook 2
also produces `models/tfidf_vectorizer.pkl` and
`models/gradient_boosting_model.pkl`, used by the dashboard below.

To reproduce from raw data instead (slow — real ~277MB download + a long
CPU processing run):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/1_data_preprocessing.ipynb
```

### Dashboard

Run it from the repo root:

```bash
python -m dashboard.app
```

then open `http://127.0.0.1:8050`. It loads `data/cleaned/reviews.csv`,
`data/raw_sample.csv`, and the committed `models/*.pkl`, so it works straight
after install without re-running any notebook (the first run downloads a few
NLTK resources if they're missing). Pages: Home, Uncleaned Data (1,000-row raw
sample), Cleaned Data, Sentiment Analysis (donut, word treemap, top reviewers,
top reviewers by sentiment, volume over time, sentiment over time), and
Provide Review, which scores the text you submit with the Gradient Boosting
model and appends it to `data/user_reviews.csv`.

## License

MIT — see `LICENSE`. Portions originally authored by Taimour Karim (2024-present).
