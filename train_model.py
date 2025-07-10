# train_model.py

import joblib
from sklearn.linear_model import LogisticRegression

# Load preprocessed training data
vect = joblib.load("tfidf_vect.joblib")
X_train, y_train = joblib.load("train_data.joblib")

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Persist
joblib.dump(model, "sentiment_lr.joblib")
print("Model trained and saved as sentiment_lr.joblib")
