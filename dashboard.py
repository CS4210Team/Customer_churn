# UI, to run --> "streamlit run dashboard.py" 
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="Customer Churn ML Dashboard", layout="wide")
st.title("Customer Churn Report")

# --- Load dataset ---
from project.data_utils import load_csv
df = load_csv()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# --- Load models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "project", "models")

models = {
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "log_model.joblib")),
    "Decision Tree": joblib.load(os.path.join(MODELS_DIR, "tree_model.joblib")),
    "SVM": joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib")),
    "Linear Regression": joblib.load(os.path.join(MODELS_DIR, "linear_model.joblib"))
}

X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "test_data.joblib"))

# --- Eval models ---
st.subheader("Model Performance Comparison")

metrics_data = []

for name, model in models.items():
    if name == "Linear Regression":
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics_data.append({
            "Model": name,
            "Accuracy": "-",
            "Precision": "-",
            "Recall": "-",
            "F1-score": "-",
            "MSE": round(mse, 3),
            "R²": round(r2, 3)
        })
    else:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_data.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(report["weighted avg"]["precision"], 3),
            "Recall": round(report["weighted avg"]["recall"], 3),
            "F1-score": round(report["weighted avg"]["f1-score"], 3),
            "MSE": "-",
            "R²": "-"
        })

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df)


# --- Confusion matrices ---
st.subheader("Confusion Matrices")

classification_models = ["Logistic Regression", "Decision Tree", "SVM"]

cols = st.columns(len(classification_models))  # create columns

for i, name in enumerate(classification_models):
    model = models[name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    with cols[i]:  # put each plot in its column
        st.write(f"**{name}**")
        fig, ax = plt.subplots(figsize=(3, 2.5))  
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 8})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        st.pyplot(fig)
        
# Class distribution
st.subheader("Class Distribution")
class_counts = df['Churn'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="Set2")
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
st.pyplot(fig)