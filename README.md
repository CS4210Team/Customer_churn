# Customer_churn

<!DOCTYPE html>
<html>
<body>

<h1>Telco Customer Churn – Local Dataset Setup</h1>

<p>
This project downloads and loads the <strong>Telco Customer Churn</strong> dataset from Kaggle using a Python script, environment variables, and a virtual environment.
</p>

<hr>

<h2>1. Project Structure</h2>

<pre>
CUSTOMER_CHURN/
│
├── project/                 # Main Python package
│   ├── data_utils.py        # Downloading, unzipping, loading, preprocessing
│   ├── train.py             # Model training and saving
│   ├── evaluate.py          # Model loading and evaluation
│   │
│   ├── models/              # SAVED trained models + test data
│   │   ├── log_model.joblib
│   │   ├── tree_model.joblib
│   │   ├── linear_model.joblib
│   │   ├── svm_model.joblib
│   │   └── test_data.joblib
│   │
│   └── data/                # Raw downloaded data goes here
│
├── main.py                  # Single app entrypoint
├── .env                     # Kaggle keys
├── .gitignore
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
</pre>

<hr>

<h2>2. Prerequisites</h2>

<ul>
  <li>Your Kaggle API Key (downloaded as <code>kaggle.json</code>)</li>
  <li>your profile -> settings -> API -> create new token</li>
</ul>

<hr>

<h2>3. Create a Virtual Environment</h2>

<h3>mac:</h3>
<pre>
python3 -m venv venv
source venv/bin/activate
</pre>

<h3>Windows:</h3>
<pre>
python -m venv venv
venv\Scripts\activate
</pre>

<hr>

<h2>4. Install Dependencies</h2>

<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>5. Add Your Kaggle Credentials</h2>

<p>Create a file named <code>.env</code> in the project folder:</p>

<pre>
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
</pre>

<hr>

<h2>6. Run the main file to download the dataset, train the model, and evaluate the model</h2>

<p>Make sure your virtual environment is activated, then run:</p>

<pre>
python main.py
</pre>

<p>This script will:</p>

<ul>
  <li>Authenticate with Kaggle</li>
  <li>Download the dataset ZIP into <code>data/</code></li>
  <li>Extract the files</li>
  <li>Load the CSV using pandas</li>
  <li>Train the models</li>
  <li>Evaluate the models</li>
</ul>

<hr>

<h2>7. After running</h2>

<p>You will see:</p>

<pre>
data/
 ├── telco-customer-churn.zip
 └── WA_Fn-UseC_-Telco-Customer-Churn.csv
</pre>

<pre>
project/
 ├── linear_model.joblib
 ├── log_model.joblib
 ├── svm_model.joblib
 ├── test_data.joblib
 └── tree_model.joblib 
</pre>

<hr>

<h2>8. Deactivate environment</h2>

<pre>deactivate</pre>

<hr>

</body>
</html>
