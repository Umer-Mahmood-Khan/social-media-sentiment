import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1) Load your test split and artifacts
    X_test, y_test = joblib.load("test_data.joblib")
    vect           = joblib.load("tfidf_vect.joblib")    # if you need to re-vectorize, otherwise skip
    model          = joblib.load("sentiment_lr.joblib")

    # 2) (Optional) If X_test is raw text, vectorize:
    # X_test = vect.transform([text for text in X_test_raw])

    # 3) Predict
    y_pred = model.predict(X_test)

    # 4) Compute and print metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", 
          classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
