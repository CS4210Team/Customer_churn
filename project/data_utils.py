import os
import zipfile
import subprocess
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Ensure Kaggle creds exist
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# BASE DIR = folder where data_utils.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA DIR inside project/
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create project/data directory
os.makedirs(DATA_DIR, exist_ok=True)

# Filenames inside project/data
ZIP_PATH = os.path.join(DATA_DIR, "telco-customer-churn.zip")
CSV_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Download dataset
def download_kaggle_dataset():
    print("Downloading Kaggle dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "blastchar/telco-customer-churn",
        "-p", DATA_DIR,
        "--force"
    ], check=True)
    print(f"Download complete. Saved into: {DATA_DIR}")

# Unzip dataset
def unzip_dataset():
    print("Unzipping dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Unzip done. Extracted to:", DATA_DIR)


# Load into pandas
def load_csv():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
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
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Standard Scaling
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    return X_scaled, y