import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from .data_utils import download_kaggle_dataset, unzip_dataset, load_csv, preprocess_data

# Save models inside project/models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Train Models
def train_models():
    # Full pipeline from data download to training

    # Download and prepare data
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # logistic regression model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # decision tree model
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)

    # linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # SVM model
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_model.fit(X_train, y_train)

    # Save all models
    joblib.dump(log_model, os.path.join(MODELS_DIR, "log_model.joblib"))
    joblib.dump(tree_model, os.path.join(MODELS_DIR, "tree_model.joblib"))
    joblib.dump(linear_model, os.path.join(MODELS_DIR, "linear_model.joblib"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.joblib"))

    # Save test data for evaluation
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_data.joblib"))

    print("Training complete. Models and test data saved in the 'models' folder.")