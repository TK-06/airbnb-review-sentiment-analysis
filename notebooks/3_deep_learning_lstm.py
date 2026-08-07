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
