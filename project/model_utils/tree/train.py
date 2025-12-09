import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from ...data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tree_model.joblib")
DATA_DIR = os.path.join(BASE_DIR, "..")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")
FULL_DATA_PATH = os.path.join(DATA_DIR, "full_data.joblib")

os.makedirs(BASE_DIR, exist_ok=True)

def train_tree():
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)

    # stratify because of imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tree = DecisionTreeClassifier(random_state=42)

    # grid around what worked
    param_grid = {"max_depth": [4, 5, 6], "min_samples_split": [2, 10, 20], 
                  "min_samples_leaf": [1, 5, 10], "criterion": ["gini", "entropy"]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(tree, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)

    grid.fit(X_train, y_train)
    best_tree = grid.best_estimator_

    joblib.dump(best_tree, MODEL_PATH)

    if not os.path.exists(TEST_DATA_PATH):
        joblib.dump((X_test, y_test), TEST_DATA_PATH)
        joblib.dump((X, y), FULL_DATA_PATH)

    print("Decision Tree trained and saved.")

if __name__ == "__main__":
    train_tree()
