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
