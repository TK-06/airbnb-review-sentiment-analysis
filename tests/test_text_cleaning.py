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
