# UI, to run --> "streamlit run dashboard.py" 
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Customer Churn ML Dashboard", layout="wide")
st.title("Customer Churn Report")

# --- Load dataset ---
from project.data_utils import load_csv
df = load_csv()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Overall Dataset Visualizations")

col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    # --- Overall Dataset Graphs ---
    # Class distribution
    fig, ax = plt.subplots()  # small figure
    class_counts = df['Churn'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="Set2")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Numeric feature distributions
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Show up to 3 important numeric columns if they exist
    important_numeric = [col for col in ["tenure", "MonthlyCharges", "TotalCharges"] if col in numeric_cols]

    if important_numeric:
        st.markdown("**Key Numeric Feature Distributions**")
        num_cols = st.columns(len(important_numeric))
        for i, col_name in enumerate(important_numeric):
            with num_cols[i]:
                fig, ax = plt.subplots()
                sns.histplot(df[col_name], kde=True, ax=ax)
                ax.set_title(col_name)
                st.pyplot(fig)

    # Churn rate by Contract type (if column exists)
    if "Contract" in df.columns:
        st.markdown("**Churn Rate by Contract Type**")
        churn_rate_by_contract = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index(name="ChurnRate")
        )

        fig, ax = plt.subplots()
        sns.barplot(data=churn_rate_by_contract, x="Contract", y="ChurnRate", ax=ax)
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    # Monthly Charges vs Churn (if column exists)
    if "MonthlyCharges" in df.columns:
        st.markdown("**Monthly Charges by Churn Status**")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges")
        st.pyplot(fig)

# --- Load models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "project", "models")

models = {
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "log_model.joblib")),
    "Decision Tree": joblib.load(os.path.join(MODELS_DIR, "tree_model.joblib")),
    "SVM": joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib")),
    "KNN": joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
}

X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "test_data.joblib"))

# --- Cross-Validation Section ---
st.subheader("Cross-Validation Scores")

# Option to run CV
run_cv = st.checkbox("Run cross-validation?", value=False)

if run_cv:
    cv_folds = st.slider("Select number of folds:", min_value=2, max_value=10, value=5)
    cv_metrics = []

    # Define scorers for CV
    scoring = {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "Recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "F1-score": make_scorer(f1_score, average="weighted", zero_division=0)
    }

    for name, model in models.items():
        scores_dict = {}
        scores_dict["Model"] = name
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring=scorer)
            scores_dict[f"{metric_name} Mean"] = round(scores.mean(), 3)
            scores_dict[f"{metric_name} Std"] = round(scores.std(), 3)
        cv_metrics.append(scores_dict)

    cv_df = pd.DataFrame(cv_metrics)
    st.dataframe(cv_df)
    
# --- Eval models ---
st.subheader("Model Performance Comparison")

metrics_data = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_data.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(report["weighted avg"]["precision"], 3),
        "Recall": round(report["weighted avg"]["recall"], 3),
        "F1-score": round(report["weighted avg"]["f1-score"], 3),
    })

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df)


# --- Confusion matrices ---
st.subheader("Confusion Matrices")

classification_models = ["Logistic Regression", "Decision Tree", "SVM", "KNN"]

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

# --- Model Metric Graphs ---
st.subheader("Model Metric Visualizations")

metrics_left, metrics_center, metrics_right = st.columns([1, 3, 1])
with metrics_center:
    # Accuracy per model
    if not metrics_df.empty:
        st.markdown("**Accuracy by Model**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=metrics_df, x="Model", y="Accuracy", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

        # F1-score per model
        st.markdown("**F1-Score by Model**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=metrics_df, x="Model", y="F1-score", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-score")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

        # Precision vs Recall as side-by-side bars for each model
        st.markdown("**Precision vs Recall (per Model)**")
        pr_df = metrics_df.melt(id_vars="Model", value_vars=["Precision", "Recall"], var_name="Metric", value_name="Score")

        fig, ax = plt.subplots()
        sns.barplot(data=pr_df, x="Model", y="Score", hue="Metric", ax=ax)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)