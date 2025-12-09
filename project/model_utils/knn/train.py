import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from ...data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

    # Stratified split is usually better for churn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: scale -> KNN
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ]
    )

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 20],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # Manhattan and Euclidean
        "knn__leaf_size": [15, 30, 45] 
    }

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_


    joblib.dump(best_model, MODEL_PATH)

    if not os.path.exists(TEST_DATA_PATH):
        joblib.dump((X_test, y_test), TEST_DATA_PATH)
        joblib.dump((X, y), FULL_DATA_PATH)

    print("KNN trained and saved.")

if __name__ == "__main__":
    train_knn()