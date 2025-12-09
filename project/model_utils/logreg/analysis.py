# analysis.py

import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# from mlxtend.plotting import plot_decision_regions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "log_model.joblib")


def ensure_dataframe(X):
    """Convert NumPy array to DataFrame if needed, with generic column names."""
    model = joblib.load(MODEL_PATH)
    if hasattr(X, "columns"):
        return X
    else:
        return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(model.coef_.shape[1])])


def get_logreg_weights(X=None):
    """Return logistic regression coefficients and intercept."""
    model = joblib.load(MODEL_PATH)
    
    if X is not None:
        X = ensure_dataframe(X)
        feature_names = X.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(model.coef_.shape[1])]

    coef_df = pd.DataFrame(model.coef_, columns=feature_names)
    intercept = model.intercept_[0]
    return coef_df, intercept


def get_logreg_probabilities(X, top_n=None):
    """Return predicted probabilities for each class. Optionally top N rows."""
    X = ensure_dataframe(X)
    model = joblib.load(MODEL_PATH)
    prob_df = pd.DataFrame(
        model.predict_proba(X),
        columns=[f"Prob_{cls}" for cls in model.classes_]
    )
    if top_n is not None:
        prob_df = prob_df.head(top_n)
    return prob_df


def get_logreg_predictions(X):
    """Return predicted class labels."""
    X = ensure_dataframe(X)
    model = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(X), name="Predicted_Class")


def get_logreg_score(X, y):
    """Return model accuracy on the given data."""
    X = ensure_dataframe(X)
    model = joblib.load(MODEL_PATH)
    return model.score(X, y)


def get_top2_features(X):
    """
    Return top 2 features by absolute coefficient magnitude.
    Handles both pandas DataFrame and NumPy array.
    """
    X = ensure_dataframe(X)
    return list(X.columns[:2])


def plot_logreg_decision_boundary(X, y):
    """
    Return a matplotlib figure of the decision boundary (uses top 2 features).
    Automatically pads remaining features with mean values.
    """
    X = ensure_dataframe(X)
    model = joblib.load(MODEL_PATH)

    top_features = get_top2_features(X)
    X_plot = X[top_features]

    # Create grid for decision boundary
    x_min, x_max = X_plot.iloc[:, 0].min() - 0.5, X_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.5, X_plot.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Fill full feature array with mean values
    full_grid = np.tile(X.mean().to_numpy(), (grid_points.shape[0], 1))
    top_idx = [X.columns.get_loc(f) for f in top_features]
    full_grid[:, top_idx] = grid_points

    # Predict probabilities for coloring
    prob = model.predict_proba(full_grid)[:, 1].reshape(xx.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    ax.set_xlabel(X_plot.columns[0])
    ax.set_ylabel(X_plot.columns[1])
    ax.set_title("Decision Boundary")
    return fig


def plot_logreg_combined(X, y, class_index=1, resolution=100):
    """
    Combined figure: decision boundary + probability surface for top 2 features.
    Automatically pads remaining features with mean values.
    """
    X = ensure_dataframe(X)
    model = joblib.load(MODEL_PATH)

    top_features = get_top2_features(X)
    X_plot = X[top_features]

    # Prepare grid
    x_min, x_max = X_plot.iloc[:, 0].min() - 0.5, X_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.5, X_plot.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Full feature grid
    full_grid = np.tile(X.mean().to_numpy(), (grid_points.shape[0], 1))
    top_idx = [X.columns.get_loc(f) for f in top_features]
    full_grid[:, top_idx] = grid_points

    # Predict probabilities
    prob = model.predict_proba(full_grid)[:, class_index].reshape(xx.shape)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Decision Boundary
    axes[0].contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[0].scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    axes[0].set_xlabel(X_plot.columns[0])
    axes[0].set_ylabel(X_plot.columns[1])
    axes[0].set_title("Decision Boundary")

    # Probability Surface
    c = axes[1].contourf(xx, yy, prob, cmap="RdBu_r", alpha=0.6)
    axes[1].scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    axes[1].set_xlabel(X_plot.columns[0])
    axes[1].set_ylabel(X_plot.columns[1])
    axes[1].set_title(f"Predicted Probability Surface (Class {model.classes_[class_index]})")
    fig.colorbar(c, ax=axes[1], label="Predicted Probability")

    plt.tight_layout()
    return fig
