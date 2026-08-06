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
