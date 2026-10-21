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
    try:
        sentiment = predict_sentiment(review, vectorizer, model)
    except ValueError:
        return "Please enter a review with some actual words in it."
    save_user_review(name or "Anonymous", review, sentiment)
    return f"Thanks! Predicted sentiment: {sentiment}"


if __name__ == "__main__":
    app.run(debug=True)
