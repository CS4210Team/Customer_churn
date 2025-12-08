# UI, to run --> "streamlit run dashboard.py" 
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from project.model_utils.logreg.analysis import get_logreg_weights, get_logreg_probabilities, get_logreg_predictions, get_logreg_score, plot_logreg_decision_boundary, plot_logreg_combined
from project.model_utils.knn.analysis import get_knn_probabilities, get_knn_predictions, get_knn_score, plot_knn_combined

st.set_page_config(page_title="Customer Churn ML Dashboard", layout="wide")
st.title("Customer Churn Report")

# --- Load dataset ---
from project.data_utils import load_csv
df = load_csv()

# --- Load models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to models
MODEL_PATHS = {
    "Logistic Regression": os.path.join(BASE_DIR, "project/model_utils/logreg/log_model.joblib"),
    "Decision Tree": os.path.join(BASE_DIR, "project/model_utils/tree/tree_model.joblib"),
    "KNN": os.path.join(BASE_DIR, "project/model_utils/knn/knn_model.joblib"),
    "SVM": os.path.join(BASE_DIR, "project/model_utils/svm/svm_model.joblib"),
}

# Load all models
models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

# Shared test data path
DATA_DIR = os.path.join(BASE_DIR, "project/model_utils")  # parent folder of all models
X_test, y_test = joblib.load(os.path.join(DATA_DIR, "test_data.joblib"))

# tabs layout
tab_data, tab_overall, tab_logreg, tab_tree, tab_svm, tab_knn = st.tabs(
    ["Data", "Overall Results", "Logistic Regression", "Decision Tree", "SVM", "KNN"]
)

# data tab
with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Overall Dataset Visualizations")

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        # --- Overall Dataset Graphs ---
        # Class distribution
        fig, ax = plt.subplots()  # small figure
        class_counts = df["Churn"].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="Set2")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Numeric feature distributions
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Show up to 3 important numeric columns if they exist
        important_numeric = [
            col for col in ["tenure", "MonthlyCharges", "TotalCharges"] if col in numeric_cols
        ]

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

# overall results tab
with tab_overall:
    # --- Cross-Validation Section ---
    st.subheader("Cross-Validation Scores")

    run_cv = st.checkbox("Run cross-validation?", value=False)

    if run_cv:
        cv_folds = st.slider("Select number of folds:", min_value=2, max_value=10, value=5)
        cv_metrics = []

        # Define scorers for CV
        scoring = {
            "Accuracy": "accuracy",
            "Precision": make_scorer(
                precision_score, average="weighted", zero_division=0
            ),
            "Recall": make_scorer(
                recall_score, average="weighted", zero_division=0
            ),
            "F1-score": make_scorer(
                f1_score, average="weighted", zero_division=0
            ),
        }

        for name, model in models.items():
            scores_dict = {"Model": name}
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
        metrics_data.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Precision": round(report["weighted avg"]["precision"], 3),
                "Recall": round(report["weighted avg"]["recall"], 3),
                "F1-score": round(report["weighted avg"]["f1-score"], 3),
            }
        )

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
        if not metrics_df.empty:
            # Accuracy per model
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
            pr_df = metrics_df.melt(
                id_vars="Model",
                value_vars=["Precision", "Recall"],
                var_name="Metric",
                value_name="Score",
            )

            fig, ax = plt.subplots()
            sns.barplot(data=pr_df, x="Model", y="Score", hue="Metric", ax=ax)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)

# individual model tab

with tab_logreg:
    st.subheader("Logistic Regression Model")

    # Coefficients and intercept
    coef_df, intercept = get_logreg_weights(X_test)
    st.dataframe(coef_df)
    st.write(f"Intercept: {intercept}")

    # Top 10 predicted probabilities
    st.subheader("Top 10 Predicted Probabilities")
    prob_df = get_logreg_probabilities(X_test, top_n=10)
    st.dataframe(prob_df)

    # Predicted classes
    st.subheader("Predicted Classes (Top 10)")
    pred_series = get_logreg_predictions(X_test)
    st.dataframe(pred_series.head(10))

    # Model accuracy
    if 'y_test' in globals():
        accuracy = get_logreg_score(X_test, y_test)
        st.write(f"Model Accuracy: {accuracy:.2%}")
        
    # Decision boundary + probability surface plot
    st.subheader("Decision Boundary & Probability Surface (Top 2 Features)")

    # Generate the figure
    fig = plot_logreg_combined(X_test, y_test if 'y_test' in globals() else np.zeros(len(X_test)))

    # Display it in Streamlit
    st.pyplot(fig)


   
with tab_tree:
    st.subheader("Decision Tree Model")

with tab_svm:
    st.subheader("SVM Model")

with tab_knn:
    st.subheader("K-Nearest Neighbors Model")

    # Top 10 predicted probabilities
    st.subheader("Top 10 Predicted Probabilities")
    knn_prob_df = get_knn_probabilities(X_test, top_n=10)
    st.dataframe(knn_prob_df)

    # Predicted classes
    st.subheader("Predicted Classes (Top 10)")
    knn_pred_series = get_knn_predictions(X_test)
    st.dataframe(knn_pred_series.head(10))

    # Model accuracy 
    knn_accuracy = get_knn_score(X_test, y_test)
    st.write(f"KNN Model Accuracy: {knn_accuracy:.2%}")

    # Decision boundary + probability surface plot
    st.subheader("Decision Boundary & Probability Surface (Top 2 Features)")

    fig_knn = plot_knn_combined(X_test, y_test)
    
    st.pyplot(fig_knn)
