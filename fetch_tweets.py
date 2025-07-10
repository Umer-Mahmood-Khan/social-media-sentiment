import os
from dotenv import load_dotenv
import tweepy
import pandas as pd

# 1) Load .env (if you’re using python-dotenv)
load_dotenv()  

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_recent_tweets(query: str, max_results: int = 1):
    """
    Returns a pandas.DataFrame of recent tweets matching `query`.
    """
    tweets = client.search_recent_tweets(
        query=query + " -is:retweet lang:en",
        max_results=max_results,
        tweet_fields=["created_at", "text", "author_id"]
    ).data or []

    # Build DataFrame
    df = pd.DataFrame([{
        "id": t.id,
        "author_id": t.author_id,
        "created_at": t.created_at,
        "text": t.text
    } for t in tweets])
    return df

if __name__ == "__main__":
    df = fetch_recent_tweets("#Tech", max_results=50)
    print(df.head())
    df.to_csv("raw_tweets.csv", index=False)
