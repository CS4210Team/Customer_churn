import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ...data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knn_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "..")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_data.joblib")

os.makedirs(BASE_DIR, exist_ok=True)

def train_knn():
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    if not os.path.exists(TEST_DATA_PATH):
        joblib.dump((X_test, y_test), TEST_DATA_PATH)
        joblib.dump((X, y), FULL_DATA_PATH)

    print("KNN trained and saved.")

if __name__ == "__main__":
    train_knn()
