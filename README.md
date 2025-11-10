# Customer_churn

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Telco Customer Churn – Dataset Setup</title>
</head>
<body>

<h1>Telco Customer Churn – Local Dataset Setup</h1>

<p>
This project downloads and loads the <strong>Telco Customer Churn</strong> dataset from Kaggle using a Python script, environment variables, and a virtual environment.
</p>

<hr>

<h2>✅ 1. Project Structure</h2>

<pre>
customer_churn_project/
│
├── venv/                   # Virtual environment
├── data/                   # Dataset (auto-created)
│
├── .env                    # Kaggle credentials
├── .gitignore
├── requirements.txt
├── download_dataset.py
└── README.html
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

<h2>6. Run the Dataset Downloader</h2>

<p>Make sure your virtual environment is activated, then run:</p>

<pre>
python download_dataset.py
</pre>

<p>This script will:</p>

<ul>
  <li>Authenticate with Kaggle</li>
  <li>Download the dataset ZIP into <code>data/</code></li>
  <li>Extract the files</li>
  <li>Load the CSV using pandas</li>
  <li>Print the first few rows</li>
</ul>

<hr>

<h2>7. After Downloading</h2>

<p>You will see:</p>

<pre>
data/
 ├── telco-customer-churn.zip
 └── WA_Fn-UseC_-Telco-Customer-Churn.csv
</pre>

<hr>

<h2>8. Deactivate environment</h2>

<pre>deactivate</pre>

<hr>

</body>
</html>
