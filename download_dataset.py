import os
from dotenv import load_dotenv
import pandas as pd
import zipfile
import subprocess
from sklearn.model_selection import train_test_split #, trait_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load .env
load_dotenv()

# Ensure Kaggle creds exist
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Create required folders
os.makedirs("data", exist_ok=True)

# Download dataset
def download_kaggle_dataset():
    print("Downloading Kaggle dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "blastchar/telco-customer-churn",
        "-p", "data",
        "--force"
    ], check=True)
    print("Download complete.")


# Unzip dataset
def unzip_dataset():
    zip_path = "data/telco-customer-churn.zip"
    print("Unzipping dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")

    print("Unzip done.")


# Load into pandas
def load_csv():
    csv_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print(df.head())
    return df

# Data preprocessing
def preprocess_data(df):
    print("\nPreprocessing data...")
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns = cat_cols, drop_first = True)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    scalar = StandardScaler() # changed it from StandardScalar
    X_scaled = scalar.fit_transform(X)
    
    return X_scaled, y

# Train Models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42  
    )
    
    # logistic regression model
    log_model = LogisticRegression(max_iter = 1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    
    # decision tree model
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    #linear regression model 
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_linear)
    r2 = r2_score(y_test, y_pred_linear)

    #SVM model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # --- Evaluation Section ---
    print("\n--- Decision Tree Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_tree))
    print("Classification Report:\n", classification_report(y_test, y_pred_tree))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

    print("\n--- Logistic Regression Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("Classification Report:\n", classification_report(y_test, y_pred_log))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

    print("\n--- Linear Regression Evaluation ---")
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

    print("\n--- SVM Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    

if __name__ == "__main__":
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
    X, y = preprocess_data(df)
    train_models(X, y)
