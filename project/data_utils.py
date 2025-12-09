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
    # print(df.head())
    return df

def preprocess_data(df):
    print("\nPreprocessing data...")
    df = df.copy()

    # Drop ID
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode all categorical features
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # return unscaled DataFrame
    return X, y

# Data preprocessing
# def preprocess_data(df):
#     print("\nPreprocessing data...")

#     # Drop ID column
#     df = df.drop("customerID", axis=1)

#     # Convert to numeric safely
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#     df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

#     # Encode target
#     df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#     # Categorical columns
#     cat_cols = df.select_dtypes(include=["object"]).columns

#     # Optional: collapse rare categories
#     for col in cat_cols:
#         freq = df[col].value_counts(normalize=True)
#         rare = freq[freq < 0.01].index
#         df[col] = df[col].replace(rare, "Other")

#     # One-hot encode categoricals
#     df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

#     df["ChargePerTenure"] = df["MonthlyCharges"] * (df["tenure"] + 1)
#     df["MonthlyTimesSupport"] = df["MonthlyCharges"] * df["TechSupport_Yes"]
#     df["ShortTenure"] = (df["tenure"] < 12).astype(int)

#     # --- Now split into X and y ---
#     X = df.drop("Churn", axis=1)
#     y = df["Churn"]

#     # Numeric feature columns (now only from X, so no 'Churn')
#     num_cols = X.select_dtypes(include=["int64", "float64"]).columns

#     # Clip outliers on numeric features
#     X[num_cols] = X[num_cols].clip(
#         X[num_cols].quantile(0.01),
#         X[num_cols].quantile(0.99),
#         axis=1
#     )

#     # Standard scale numeric features
#     scaler = StandardScaler()
#     X[num_cols] = scaler.fit_transform(X[num_cols])

#     return X, y


# import os
# import zipfile
# import subprocess
# import pandas as pd
# from sklearn.preprocessing import StandardScaler  # still imported if you need it later
# from dotenv import load_dotenv

# # Load .env
# load_dotenv()

# # Ensure Kaggle creds exist
# os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
# os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# # BASE DIR = folder where data_utils.py lives
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # DATA DIR inside project/
# DATA_DIR = os.path.join(BASE_DIR, "data")

# # Create project/data directory
# os.makedirs(DATA_DIR, exist_ok=True)

# # ---------------- TELCO PATHS (original) ----------------
# TELCO_ZIP_PATH = os.path.join(DATA_DIR, "telco-customer-churn.zip")
# TELCO_CSV_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# # ---------------- BANK PATHS (new) ----------------
# BANK_ZIP_PATH = os.path.join(DATA_DIR, "churn-for-bank-customers.zip")
# BANK_CSV_PATH = os.path.join(DATA_DIR, "churn.csv")  # Kaggle's bank churn CSV


# # ============================================================
# #                DOWNLOAD DATASETS
# # ============================================================

# def download_kaggle_dataset(dataset: str = "bank"):
#     """
#     Download the selected Kaggle dataset into DATA_DIR.

#     dataset: "telco" (default) or "bank"
#     """
#     if dataset == "telco":
#         print("Downloading Telco Customer Churn dataset...")
#         subprocess.run([
#             "kaggle", "datasets", "download",
#             "-d", "blastchar/telco-customer-churn",
#             "-p", DATA_DIR,
#             "--force"
#         ], check=True)
#         print(f"Telco download complete. Saved into: {DATA_DIR}")

#     elif dataset == "bank":
#         print("Downloading Bank Customer Churn dataset...")
#         subprocess.run([
#             "kaggle", "datasets", "download",
#             "-d", "mathchi/churn-for-bank-customers",
#             "-p", DATA_DIR,
#             "--force"
#         ], check=True)
#         print(f"Bank download complete. Saved into: {DATA_DIR}")

#     else:
#         raise ValueError(f"Unknown dataset '{dataset}'. Use 'telco' or 'bank'.")


# # ============================================================
# #                UNZIP DATASETS
# # ============================================================

# def unzip_dataset(dataset: str = "bank"):
#     """
#     Unzip the selected dataset into DATA_DIR.

#     dataset: "telco" (default) or "bank"
#     """
#     if dataset == "telco":
#         zip_path = TELCO_ZIP_PATH
#     elif dataset == "bank":
#         zip_path = BANK_ZIP_PATH
#     else:
#         raise ValueError(f"Unknown dataset '{dataset}'. Use 'telco' or 'bank'.")

#     print(f"Unzipping {dataset} dataset from {zip_path} ...")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(DATA_DIR)
#     print("Unzip done. Extracted to:", DATA_DIR)


# # ============================================================
# #                LOAD CSV
# # ============================================================

# def load_csv(dataset: str = "bank"):
#     """
#     Load the requested dataset as a raw pandas DataFrame.

#     - For 'telco': original Telco CSV
#     - For 'bank': bank churn CSV, with 'Exited' renamed to 'Churn'
#     """
#     print("Loading CSV...")

#     if dataset == "telco":
#         df = pd.read_csv(TELCO_CSV_PATH)
#         return df

#     elif dataset == "bank":
#         df = pd.read_csv(BANK_CSV_PATH)

#         # Rename target column Exited -> Churn so the rest of the pipeline
#         # can consistently use 'Churn'
#         if "Exited" in df.columns and "Churn" not in df.columns:
#             df.rename(columns={"Exited": "Churn"}, inplace=True)

#         return df

#     else:
#         raise ValueError(f"Unknown dataset '{dataset}'. Use 'telco' or 'bank'.")


# # ============================================================
# #                PREPROCESSING
# # ============================================================

# def preprocess_data(df: pd.DataFrame, dataset: str = "telco"):
#     """
#     Preprocesses Telco OR Bank churn datasets into X, y.

#     - TELCO branch is *exactly* your custom pipeline (rare-category collapse,
#       engineered features, clipping, scaling, etc.).
#     - BANK branch is a minimal, separate pipeline that also uses 'Churn'
#       as the target (renamed from Exited).
#     """
#     print("\nPreprocessing data...")
#     df = df.copy()

#     # ----------------------------------------------------------
#     # TELCO PREPROCESSING (your original code, verbatim)
#     # ----------------------------------------------------------
#     if dataset == "telco":
#         # Drop ID column
#         df = df.copy()

#         # Drop ID
#         df.drop("customerID", axis=1, inplace=True)

#         # Fix TotalCharges
#         df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#         df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

#         # Encode target
#         df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#         # One-hot encode all categorical features
#         cat_cols = df.select_dtypes(include=["object"]).columns
#         df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

#         X = df.drop("Churn", axis=1)
#         y = df["Churn"]

#         # return unscaled DataFrame
#         return X, y


#     # ----------------------------------------------------------
#     # BANK PREPROCESSING (separate branch)
#     # ----------------------------------------------------------
#     elif dataset == "bank":
#         # Ensure we have 'Churn' column (from Exited)
#         if "Exited" in df.columns and "Churn" not in df.columns:
#             df.rename(columns={"Exited": "Churn"}, inplace=True)

#         # Drop obvious ID-like columns if present
#         drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
#         if drop_cols:
#             df.drop(drop_cols, axis=1, inplace=True)

#         # Treat string columns as categoricals
#         cat_cols = df.select_dtypes(include=["object"]).columns

#         # One-hot encode categoricals (simple version; no rare-category collapse needed)
#         if len(cat_cols) > 0:
#             df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

#         # Make sure target is numeric 0/1
#         df["Churn"] = df["Churn"].astype(int)

#         # Split into X and y
#         X = df.drop("Churn", axis=1)
#         y = df["Churn"]

#         # Numeric columns
#         num_cols = X.select_dtypes(include=["int64", "float64"]).columns

#         # Optional: use similar clipping & scaling as Telco for consistency
#         if len(num_cols) > 0:
#             X[num_cols] = X[num_cols].clip(
#                 X[num_cols].quantile(0.01),
#                 X[num_cols].quantile(0.99),
#                 axis=1
#             )
#             scaler = StandardScaler()
#             X[num_cols] = scaler.fit_transform(X[num_cols])

#         return X, y

#     else:
#         raise ValueError(f"Unknown dataset '{dataset}'. Use 'telco' or 'bank'.")