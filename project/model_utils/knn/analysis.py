import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_MODEL_PATH = os.path.join(BASE_DIR, "knn_model.joblib")


def ensure_knn_dataframe(X):
    """
    Convert NumPy array to DataFrame if needed, with generic column names.
    """
    if hasattr(X, "columns"):
        return X
    else:
        n_features = X.shape[1]
        return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])


def get_knn_probabilities(X, top_n=None):
    """
    Return predicted probabilities for each class. Optionally top N rows.
    """
    X = ensure_knn_dataframe(X)
    model: KNeighborsClassifier = joblib.load(KNN_MODEL_PATH)

    prob_df = pd.DataFrame(
        model.predict_proba(X),
        columns=[f"Prob_{cls}" for cls in model.classes_]
    )
    if top_n is not None:
        prob_df = prob_df.head(top_n)
    return prob_df


def get_knn_predictions(X):
    """
    Return predicted class labels as a pandas Series.
    """
    X = ensure_knn_dataframe(X)
    model: KNeighborsClassifier = joblib.load(KNN_MODEL_PATH)
    return pd.Series(model.predict(X), name="Predicted_Class")


def get_knn_score(X, y):
    """
    Return model accuracy on the given data.
    """
    X = ensure_knn_dataframe(X)
    model: KNeighborsClassifier = joblib.load(KNN_MODEL_PATH)
    return model.score(X, y)


def get_knn_top2_features(X):
    """
    use the first 2 feature columns since KNN doesnt have coefs.
    """
    X = ensure_knn_dataframe(X)
    return list(X.columns[:2])


def plot_knn_combined(X, y, class_index=1, resolution=100):
    """
    Combined figure: decision boundary + probability surface for top 2 features.

    - Uses first 2 features (via get_knn_top2_features)
    - Pads any remaining features with their mean values
    - Colors by predicted probability for class_index
    """
    X = ensure_knn_dataframe(X)
    model: KNeighborsClassifier = joblib.load(KNN_MODEL_PATH)

    top_features = get_knn_top2_features(X)
    X_plot = X[top_features]

    # Prepare grid over the top 2 features
    x_min, x_max = X_plot.iloc[:, 0].min() - 0.5, X_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.5, X_plot.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Build full feature grid, filling non-plotted features with their mean values
    full_grid = np.tile(X.mean().to_numpy(), (grid_points.shape[0], 1))
    top_idx = [X.columns.get_loc(f) for f in top_features]
    full_grid[:, top_idx] = grid_points

    # Predict probabilities for the chosen class
    prob = model.predict_proba(full_grid)[:, class_index].reshape(xx.shape)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Decision Boundary
    axes[0].contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[0].scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    axes[0].set_xlabel(X_plot.columns[0])
    axes[0].set_ylabel(X_plot.columns[1])
    axes[0].set_title("KNN Decision Boundary")

    # Probability Surface
    contour2 = axes[1].contourf(xx, yy, prob, cmap="RdBu_r", alpha=0.6)
    axes[1].scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    axes[1].set_xlabel(X_plot.columns[0])
    axes[1].set_ylabel(X_plot.columns[1])
    axes[1].set_title(f"KNN Predicted Probability Surface (Class {model.classes_[class_index]})")

    fig.colorbar(contour2, ax=axes[1], label="Predicted Probability")

    plt.tight_layout()
    return fig