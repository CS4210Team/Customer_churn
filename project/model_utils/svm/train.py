import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from ...data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "..")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_data.joblib")

os.makedirs(BASE_DIR, exist_ok=True)

# Train SVM model
def train_svm():
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning with GridSearchCV
    params = {
        "C": [0.1, 1, 5, 10],
        "gamma": ["scale", 0.01, 0.1, 1],
        "kernel": ["rbf"]
    }

    # Initialize GridSearchCV
    grid = GridSearchCV(SVC(), param_grid=params, cv=3, scoring="accuracy", n_jobs=-1)

    # Fit model
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    joblib.dump(model, MODEL_PATH)
    if not os.path.exists(TEST_DATA_PATH):
        joblib.dump((X_test, y_test), TEST_DATA_PATH)
        joblib.dump((X, y), FULL_DATA_PATH)

    print("SVM trained and saved.")

if __name__ == "__main__":
    train_svm()
    