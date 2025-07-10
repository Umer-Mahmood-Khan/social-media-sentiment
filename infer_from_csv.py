# infer_from_csv.py

import joblib
import pandas as pd
from preprocess import clean_text

def run_inference_from_csv(
    input_path: str = "raw_tweets.csv",
    output_path: str = "tech_sentiment_output.csv"
) -> pd.DataFrame:
    """
    1) Loads your pre-scraped tweets from CSV
    2) Cleans text
    3) Vectorizes with your existing TF–IDF
    4) Predicts with your trained LogisticRegression
    5) Saves results to a new CSV
    """
    # 1) Load the CSV (must have at least a 'text' column)
    df = pd.read_csv(input_path)

    # 2) Clean the text
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # 3) Load vectorizer & model
    vect  = joblib.load("tfidf_vect.joblib")
    model = joblib.load("sentiment_lr.joblib")

    # 4) Transform & predict
    X_new = vect.transform(df["clean_text"])
    df["sentiment"]      = model.predict(X_new)
    df["confidence_pos"] = model.predict_proba(X_new)[:, 1]

    # 5) Save enriched results
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df_result = run_inference_from_csv(
        input_path="raw_tweets.csv",          # your existing file
        output_path="tech_sentiment_output.csv"
    )
    print(df_result.head())
