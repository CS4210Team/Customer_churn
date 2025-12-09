# Customer_churn

<!DOCTYPE html>
<html>
<body>

<h1>Telco Customer Churn – Local Dataset Setup</h1>

<p>
This project downloads and loads the <strong>Telco Customer Churn</strong> dataset from Kaggle using a Python script, environment variables, and a virtual environment. It then trains and evaulates 4 different modles, displayingthe results on the terminal and a GUI.
</p>

<hr>

<h2>1. Prerequisites</h2>

<ul>
  <li>Your Kaggle API Key (downloaded as <code>kaggle.json</code>)</li>
  <li>your profile -> settings -> API -> create new token</li>
</ul>

<hr>

<h2>2. Create a Virtual Environment</h2>

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

<h2>3. Install Dependencies</h2>

<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>4. Add Your Kaggle Credentials</h2>

<p>Create a file named <code>.env</code> in the main project folder:</p>

<pre>
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
</pre>

<hr>

<h2>5. Run the main file to download the dataset, train the model, and evaluate the model</h2>
 
<p>You need to run main to train the models at least once before you can run the UI</p>
<p><strong>Make sure your virtual environment is activated, then run:</strong></p>

<pre>
python main.py
streamlit run dashboard.py (for UI) 
</pre>

<hr>

<h2>6. Deactivate environment</h2>

<pre>deactivate</pre>

<hr>

</body>
</html>
