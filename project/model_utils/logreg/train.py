import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ...data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "log_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "..")  # store shared data here
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_data.joblib")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.joblib")  # NEW

os.makedirs(BASE_DIR, exist_ok=True)

def train_logreg():
    # Download and preprocess data
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)

    # Save post-processed feature names
    joblib.dump(X.columns.tolist(), FEATURE_NAMES_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model and shared data
    joblib.dump(model, MODEL_PATH)
    joblib.dump((X_test, y_test), TEST_DATA_PATH)
    joblib.dump((X, y), FULL_DATA_PATH)

    print("Logistic Regression trained and saved.")

if __name__ == "__main__":
    train_logreg()
