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
