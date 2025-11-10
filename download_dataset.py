import os
from dotenv import load_dotenv
import pandas as pd
import zipfile
import subprocess

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


if __name__ == "__main__":
    download_kaggle_dataset()
    unzip_dataset()
    df = load_csv()
