# analysis.py for SVM

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # for type hints only

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVM_MODEL_PATH = os.path.join(BASE_DIR, "svm_model.joblib")


def ensure_svm_dataframe(X):
    """Convert NumPy array to DataFrame if needed, with generic column names."""
    if hasattr(X, "columns"):
        # Already a DataFrame
        return X
    else:
        n_features = X.shape[1]
        return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])


def get_svm_decision_scores(X, top_n=None):
    """Return SVM decision function scores as a DataFrame."""
    X = ensure_svm_dataframe(X)
    model: SVC = joblib.load(SVM_MODEL_PATH)

    # decision_function shape: (n_samples,) for binary classification
    scores = model.decision_function(X)

    # Wrap in DataFrame for consistency with other helpers
    score_df = pd.DataFrame({"Decision_Score": scores})

    if top_n is not None:
        score_df = score_df.head(top_n)

    return score_df


def get_svm_predictions(X):
    """Return predicted class labels as a pandas Series."""
    X = ensure_svm_dataframe(X)
    model: SVC = joblib.load(SVM_MODEL_PATH)
    return pd.Series(model.predict(X), name="Predicted_Class")


def get_svm_score(X, y):
    """Return model accuracy on the given data."""
    X = ensure_svm_dataframe(X)
    model: SVC = joblib.load(SVM_MODEL_PATH)
    return model.score(X, y)


def get_svm_top2_features(X):
    """For visualization, use MonthlyCharges vs TotalCharges."""
    X = ensure_svm_dataframe(X)
    return ["Feature_2", "Feature_3"]


def plot_svm_combined(X, y, class_index=1, resolution=100):
    """Combined figure: "decision boundary" + score surface for top 2 features."""
    X = ensure_svm_dataframe(X)
    model: SVC = joblib.load(SVM_MODEL_PATH)

    top_features = get_svm_top2_features(X)
    X_plot = X[top_features]

    # Prepare grid over the top 2 features
    x_min, x_max = X_plot.iloc[:, 0].min() - 0.5, X_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.5, X_plot.iloc[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Build full feature grid, filling non-plotted features with their mean values
    full_grid = np.tile(X.mean().to_numpy(), (grid_points.shape[0], 1))
    top_idx = [X.columns.get_loc(f) for f in top_features]
    full_grid[:, top_idx] = grid_points

    # decision_function gives continuous scores; reshape for contour plotting
    scores = model.decision_function(full_grid).reshape(xx.shape)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # "Decision Boundary" view (using scores)
    cs1 = axes[0].contourf(xx, yy, scores, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[0].scatter(
        X_plot.iloc[:, 0],
        X_plot.iloc[:, 1],
        c=y,
        cmap="bwr",
        edgecolor="k",
        s=50,
    )
    axes[0].set_xlabel(X_plot.columns[0])
    axes[0].set_ylabel(X_plot.columns[1])
    axes[0].set_title("SVM Decision Function Surface")
    fig.colorbar(cs1, ax=axes[0], label="Decision Score")

    # Second panel: same scores, just a slightly different perspective
    cs2 = axes[1].contourf(xx, yy, scores, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[1].scatter(
        X_plot.iloc[:, 0],
        X_plot.iloc[:, 1],
        c=y,
        cmap="bwr",
        edgecolor="k",
        s=50,
    )
    axes[1].set_xlabel(X_plot.columns[0])
    axes[1].set_ylabel(X_plot.columns[1])
    axes[1].set_title("SVM Decision Surface (Class Separation)")
    fig.colorbar(cs2, ax=axes[1], label="Decision Score")

    plt.tight_layout()
    return fig
