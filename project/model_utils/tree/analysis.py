import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tree_model.joblib")


def ensure_dataframe(X):
    """
    Convert NumPy array to DataFrame if needed, with generic column names.
    """
    if hasattr(X, "columns"):
        return X
    else:
        # Load model to know feature count
        model: DecisionTreeClassifier = joblib.load(MODEL_PATH)
        n_features = model.n_features_in_
        return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])


def get_tree_importances(X=None, top_n=10):
    """
    Return feature importances from the decision tree.
    
    Parameters:
        X (DataFrame or None): Optional dataset to get feature names.
        top_n (int): Number of top features to return.
    """
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)

    if X is not None:
        X = ensure_dataframe(X)
        feature_names = X.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(model.n_features_in_)]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # Keep only top N important features
    importance_df = importance_df[importance_df["Importance"] > 0]  # remove zero importance
    return importance_df.head(top_n)


def get_tree_probabilities(X, top_n=None):
    """
    Return predicted probabilities for each class.
    """
    X = ensure_dataframe(X)
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)

    prob_df = pd.DataFrame(
        model.predict_proba(X),
        columns=[f"Prob_{cls}" for cls in model.classes_]
    )

    if top_n is not None:
        prob_df = prob_df.head(top_n)

    return prob_df


def get_tree_predictions(X):
    """
    Return predicted class labels.
    """
    X = ensure_dataframe(X)
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)
    return pd.Series(model.predict(X), name="Predicted_Class")


def get_tree_score(X, y):
    """
    Return model accuracy.
    """
    X = ensure_dataframe(X)
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)
    return model.score(X, y)


def get_top2_features(X):
    """
    Return the top 2 most important features according to feature_importances.
    """
    X = ensure_dataframe(X)
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:2]   # top 2 indices

    return [X.columns[i] for i in idx]


def plot_tree_combined(X, y, class_index=1, resolution=150):
    """
    Combined plot: decision boundary + probability surface for the top 2 features.
    """

    X = ensure_dataframe(X)
    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)

    top_features = get_top2_features(X)
    X_plot = X[top_features]

    # Prepare grid
    x_min, x_max = X_plot.iloc[:, 0].min() - 0.5, X_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.5, X_plot.iloc[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Build full feature grid (pad rest w/ means)
    full_grid = np.tile(X.mean().to_numpy(), (grid_points.shape[0], 1))
    top_idx = [X.columns.get_loc(f) for f in top_features]
    full_grid[:, top_idx] = grid_points

    # Get probability for the selected class
    prob = model.predict_proba(full_grid)[:, class_index].reshape(xx.shape)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Decision Boundary
    axes[0].contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    scatter = axes[0].scatter(
        X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap="bwr",
        edgecolor='k', s=50
    )
    axes[0].set_xlabel(top_features[0])
    axes[0].set_ylabel(top_features[1])
    axes[0].set_title("Decision Tree Decision Boundary")

    # Probability Surface
    contour2 = axes[1].contourf(xx, yy, prob, cmap="RdBu_r", alpha=0.6)
    axes[1].scatter(
        X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y,
        cmap="bwr", edgecolor='k', s=50
    )
    axes[1].set_xlabel(top_features[0])
    axes[1].set_ylabel(top_features[1])
    axes[1].set_title(
        f"Decision Tree Predicted Probability Surface (Class {model.classes_[class_index]})"
    )

    fig.colorbar(contour2, ax=axes[1], label="Predicted Probability")

    plt.tight_layout()
    return fig
