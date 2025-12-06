import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Load models from project/models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Evaluate saved models
def evaluate_models():
    # Load saved test set
    X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "test_data.joblib"))

    log_model = joblib.load(os.path.join(MODELS_DIR, "log_model.joblib"))
    tree_model = joblib.load(os.path.join(MODELS_DIR, "tree_model.joblib"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib"))

    # --- Decision Tree Evaluation ---
    print("\n--- Decision Tree Evaluation ---")
    y_pred_tree = tree_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_tree))
    print("Classification Report:\n", classification_report(y_test, y_pred_tree))
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    print("Confusion Matrix:\n", pd.DataFrame(cm_tree, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    # --- Logistic Regression Evaluation ---
    print("\n--- Logistic Regression Evaluation ---")
    y_pred_log = log_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("Classification Report:\n", classification_report(y_test, y_pred_log))
    cm_log = confusion_matrix(y_test, y_pred_log)
    print("Confusion Matrix:\n", pd.DataFrame(cm_log, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    # --- SVM Evaluation ---
    print("\n--- SVM Evaluation ---")
    y_pred_svm = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    print("Confusion Matrix:\n", pd.DataFrame(cm_svm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    
    # --- K Nearest Neighbors Evaluation ---
    print("\n--- KNN Evaluation ---")
    y_pred_knn = knn_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("Classification Report:\n", classification_report(y_test, y_pred_knn))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print("Confusion Matrix:\n", pd.DataFrame(cm_knn, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))