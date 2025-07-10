# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(path="data/training.1600000.processed.noemoticon.csv"):
    # The original CSV has no headers, columns are:
    # [target, id, date, flag, user, text]
    cols = ["target","id","date","flag","user","text"]
    df = pd.read_csv(path, names=cols, encoding="latin-1")

    # Map original targets (0 = negative, 4 = positive) to 0/1
    df["sentiment"] = df["target"].map({0: 0, 4: 1})

    # Drop columns you don’t need
    return df[["text","sentiment"]]

def clean_text(text):
    # simple cleaning: lowercase, strip URLs, mentions, etc.
    import re
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    return text.strip()

def preprocess(df: pd.DataFrame):
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def vectorize(train_texts, test_texts):
    vect = TfidfVectorizer(max_features=5000)
    X_train = vect.fit_transform(train_texts)
    X_test  = vect.transform(test_texts)
    return vect, X_train, X_test

if __name__ == "__main__":
    df = preprocess(load_data())
    # For demonstration, let’s assume you’ve labeled some tweets manually in df.sentiment
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
    vect, X_train, X_test = vectorize(train["clean_text"], test["clean_text"])
    joblib.dump(vect, "tfidf_vect.joblib")
    joblib.dump((X_train, train["sentiment"]), "train_data.joblib")
    joblib.dump((X_test, test["sentiment"]), "test_data.joblib")
