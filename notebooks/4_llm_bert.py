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
