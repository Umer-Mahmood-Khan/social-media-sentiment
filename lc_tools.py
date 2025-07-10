# lc_tools.py
from typing import ClassVar, Dict, Any
from langchain.tools import BaseTool
import joblib
import pandas as pd
from fetch_tweets import fetch_recent_tweets
from preprocess import clean_text

class FetchTechTweets(BaseTool):
    """
    Fetch recent tweets containing a specified hashtag.
    """
    name: ClassVar[str] = "fetch_tech_tweets"
    description: ClassVar[str] = "Fetch recent tweets containing a given hashtag"

    # Only a single input: the query hashtag
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }

    def _run(self, query: str = "#Tech") -> list[dict]:
        # Fetch tweets with a fixed max_results=100
        df = fetch_recent_tweets(query, max_results=1)
        return df.to_dict(orient="records")(orient="records")

class PredictSentiment(BaseTool):
    """
    Predict sentiment for a list of tweet objects.
    """
    name: ClassVar[str] = "predict_sentiment"
    description: ClassVar[str] = "Predict sentiment (0=neg,1=pos) on a list of tweet objects"
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "tweets": {"type": "array", "items": {"type": "object"}}
        },
        "required": ["tweets"]
    }

    def _run(self, tweets: list[dict]) -> list[dict]:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(tweets)
        # Clean text and predict
        df["clean_text"] = df["text"].astype(str).apply(clean_text)
        vect = joblib.load("tfidf_vect.joblib")
        model = joblib.load("sentiment_lr.joblib")
        X = vect.transform(df["clean_text"])
        df["sentiment"] = model.predict(X)
        df["confidence_pos"] = model.predict_proba(X)[:, 1]
        # Return enriched records as list of dictionaries
        return df.to_dict(orient="records")
