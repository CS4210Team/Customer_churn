import os
import joblib
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
    linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_model.joblib"))
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib"))

    # --- Decision Tree Evaluation ---
    print("\n--- Decision Tree Evaluation ---")
    y_pred_tree = tree_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_tree))
    print("Classification Report:\n", classification_report(y_test, y_pred_tree))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

    # --- Logistic Regression Evaluation ---
    print("\n--- Logistic Regression Evaluation ---")
    y_pred_log = log_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("Classification Report:\n", classification_report(y_test, y_pred_log))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

    # --- Linear Regression Evaluation ---
    print("\n--- Linear Regression Evaluation ---")
    y_pred_linear = linear_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_linear)
    r2 = r2_score(y_test, y_pred_linear)
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

    # --- SVM Evaluation ---
    print("\n--- SVM Evaluation ---")
    y_pred_svm = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))