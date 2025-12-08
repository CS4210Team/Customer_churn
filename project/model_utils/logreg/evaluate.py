import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "log_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "..")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_data.joblib")

def evaluate_logreg():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)
    X, y = joblib.load(FULL_DATA_PATH)

    y_pred = model.predict(X_test)
    print("--- Logistic Regression Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred),
                                               index=["Actual 0", "Actual 1"],
                                               columns=["Pred 0", "Pred 1"]))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print("Cross-Validation Mean Accuracy:", cv_scores.mean())
    print("Cross-Validation Std Dev:", cv_scores.std())

if __name__ == "__main__":
    evaluate_logreg()
