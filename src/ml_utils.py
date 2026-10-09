"""
ml_utils.py
===========
Shared helper utilities for the 23CSE301 Machine Learning Capstone Project
(Review 1: full Regression track + Classification Part A).

This module is imported by every notebook in `notebooks/`. Keeping the
reusable logic here means:

* evaluation is computed by exactly the same code for every model,
  so the comparison tables are fair;
* plots share one visual style across all 15 notebooks;
* intermediate results are passed between notebooks through a single,
  well-defined artifact folder instead of being copy-pasted.

Nothing in this file fits a model or touches the test set. All modelling
decisions stay visible inside the notebooks.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# --------------------------------------------------------------------------
# 1. Reproducibility and project paths
# --------------------------------------------------------------------------

# A single fixed seed is reused everywhere (splits, models, searches) so that
# re-running the notebooks reproduces the numbers reported in the README.
RANDOM_STATE = 42

# 80 / 20 hold-out split, as recommended in the capstone guidelines.
TEST_SIZE = 0.20

# Number of folds used for every cross-validation in the project.
CV_FOLDS = 5


def project_root() -> Path:
    """Return the repository root, no matter which folder the notebook runs in.

    We walk upwards from the current working directory until we find the
    folder that contains both `dataset/` and `notebooks/`. This makes the
    notebooks runnable from Jupyter, from VS Code, and from the command line.
    """
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "dataset").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    # Fallback: assume the notebook lives in <root>/notebooks/
    return here.parent


ROOT = project_root()
DATA_DIR = ROOT / "dataset"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"

for _directory in (ARTIFACT_DIR, FIGURE_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 2. Plot styling
# --------------------------------------------------------------------------

def set_plot_style() -> None:
    """Apply one consistent, colourblind-friendly style to every notebook."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": True,
        }
    )


def save_figure(name: str, fig=None) -> Path:
    """Save the current (or given) figure into outputs/figures/ and return its path."""
    fig = fig or plt.gcf()
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------
# 3. Artifact passing between notebooks
# --------------------------------------------------------------------------

def save_artifact(obj, name: str) -> Path:
    """Persist any Python object (DataFrame, pipeline, dict) for the next notebook."""
    path = ARTIFACT_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    print(f"[saved]  {path.relative_to(ROOT)}")
    return path


def load_artifact(name: str):
    """Load an object written by an earlier notebook.

    Raises a clear, actionable error if the notebooks are run out of order.
    """
    path = ARTIFACT_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' not found at {path}.\n"
            "Run the earlier notebooks in numerical order first "
            "(see README.md → 'How to run')."
        )
    obj = joblib.load(path)
    print(f"[loaded] {path.relative_to(ROOT)}")
    return obj


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write a results table to outputs/ as CSV so it can be quoted in the report."""
    path = ROOT / "outputs" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved]  {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------
# 4. Data-quality auditing (shared by both tracks)
# --------------------------------------------------------------------------

def audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column audit: dtype, missing count/%, unique values, sample."""
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Missing": missing,
                "Missing %": round(100 * missing / n, 3) if n else 0.0,
                "Unique": int(df[col].nunique(dropna=True)),
                "Example value": df[col].dropna().iloc[0] if df[col].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Tukey IQR outlier counts for the given numeric columns.

    Outliers are *reported*, never removed here — treatment is a decision that
    belongs in the notebook, with a written justification.
    """
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        rows.append(
            {
                "Feature": col,
                "Q1": round(q1, 3),
                "Q3": round(q3, 3),
                "IQR": round(iqr, 3),
                "Lower fence": round(low, 3),
                "Upper fence": round(high, 3),
                "Outliers": int(mask.sum()),
                "Outlier %": round(100 * mask.mean(), 2),
                "Min": round(s.min(), 3),
                "Max": round(s.max(), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Outlier %", ascending=False).reset_index(drop=True)


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str], ncols: int = 3,
                               title: str = "Distribution of numerical features"):
    """Grid of histograms (with KDE) for a list of numeric columns."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation heatmap",
                             figsize=(11, 8)):
    """Annotated Pearson correlation heatmap for the numeric columns of `df`."""
    corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, annot_kws={"size": 7}, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title)
    fig.tight_layout()
    return fig, corr


# --------------------------------------------------------------------------
# 5. Regression evaluation helpers
# --------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    """R^2, RMSE and MAE computed with one shared implementation."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_regressor(name, model, X_train, y_train, X_test, y_test,
                       fit: bool = True) -> dict:
    """Fit (optionally) a regressor, predict on train and test, return a result row.

    Train metrics are reported alongside test metrics purely as an
    overfitting diagnostic. Model *selection* is always based on the test /
    cross-validated scores, never on the training score.
    """
    if fit:
        model.fit(X_train, y_train)
    train_m = regression_metrics(y_train, model.predict(X_train))
    test_m = regression_metrics(y_test, model.predict(X_test))
    return {
        "Model": name,
        "R2": test_m["R2"],
        "RMSE": test_m["RMSE"],
        "MAE": test_m["MAE"],
        "Train R2": train_m["R2"],
        "Train RMSE": train_m["RMSE"],
        "Overfit gap (Train R2 - Test R2)": train_m["R2"] - test_m["R2"],
    }


def rank_results(results: list[dict] | pd.DataFrame, by: str = "R2",
                 ascending: bool = False) -> pd.DataFrame:
    """Turn a list of result rows into a ranked comparison table."""
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results.copy()
    df = df.sort_values(by, ascending=ascending).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def plot_metric_comparison(results: pd.DataFrame, metric: str, higher_is_better: bool,
                           title: str | None = None, ax=None):
    """Horizontal bar chart comparing one metric across models."""
    df = results.sort_values(metric, ascending=not higher_is_better)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 2))
    colors = sns.color_palette("colorblind", len(df))
    ax.barh(df["Model"], df[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_title(title or f"{metric} by model ({direction})")
    span = max(df[metric].max() - min(df[metric].min(), 0), 1e-9)
    for y, v in enumerate(df[metric]):
        ax.text(v + 0.01 * span, y, f"{v:,.3f}", va="center", fontsize=9)
    ax.margins(x=0.15)
    if created:
        plt.tight_layout()
    return ax


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, ax=None):
    """Scatter of predicted vs actual with the ideal y = x reference line."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=14, edgecolor="none", color="#4C72B0")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction (y = x)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    if created:
        plt.tight_layout()
    return ax


def plot_residuals(y_true, y_pred, model_name: str):
    """Three-panel residual diagnostic: vs fitted, distribution, and Q–Q style."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=14, edgecolor="none", color="#4C72B0")
    axes[0].axhline(0, color="red", ls="--", lw=2)
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residuals vs Predicted")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#55A868")
    axes[1].axvline(0, color="red", ls="--", lw=2)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual distribution")

    from scipy import stats  # imported lazily: only needed for the Q–Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q–Q plot of residuals")

    fig.suptitle(f"Residual diagnostics — {model_name}", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, residuals


# --------------------------------------------------------------------------
# 6. Classification evaluation helpers
# --------------------------------------------------------------------------

def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy plus weighted precision / recall / F1.

    Weighted averaging is used because the Adult target is imbalanced: it
    scores every class and weights each by its support, so the minority
    class cannot be ignored the way plain accuracy allows.
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted")),
        "Weighted Precision": float(precision_score(y_true, y_pred, average="weighted",
                                                    zero_division=0)),
        "Weighted Recall": float(recall_score(y_true, y_pred, average="weighted")),
        "F1 (>50K class)": float(f1_score(y_true, y_pred, pos_label=1)),
        "Recall (>50K class)": float(recall_score(y_true, y_pred, pos_label=1)),
        "Precision (>50K class)": float(precision_score(y_true, y_pred, pos_label=1,
                                                       zero_division=0)),
    }


def evaluate_classifier(name, model, X_train, y_train, X_test, y_test,
                        fit: bool = True) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit a classifier and return (result row, confusion matrix, test predictions)."""
    if fit:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    row = {"Model": name}
    row.update(classification_metrics(y_test, y_pred))
    row["Train Accuracy"] = float(accuracy_score(y_train, model.predict(X_train)))
    row["Overfit gap (Train − Test Acc)"] = row["Train Accuracy"] - row["Accuracy"]
    cm = confusion_matrix(y_test, y_pred)
    return row, cm, y_pred


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, ax=None,
                          normalize: bool = False):
    """Readable, fully labelled confusion-matrix heatmap."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5, 4.2))
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fmt = ".2f" if normalize else "d"
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    if created:
        plt.tight_layout()
    return ax


# --------------------------------------------------------------------------
# 7. Misc
# --------------------------------------------------------------------------

def get_feature_names(fitted_column_transformer) -> np.ndarray:
    """Feature names produced by a fitted ColumnTransformer (post one-hot encoding)."""
    return fitted_column_transformer.get_feature_names_out()


def section(title: str, char: str = "=") -> None:
    """Print a tidy console section header (keeps long notebook outputs readable)."""
    print(char * 78)
    print(title)
    print(char * 78)


def describe_split(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Small table describing the frozen train/test split."""
    return pd.DataFrame(
        {
            "Set": ["Train", "Test"],
            "Rows": [X_train.shape[0], X_test.shape[0]],
            "Columns": [X_train.shape[1], X_test.shape[1]],
            "Share of data": [
                f"{X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%}",
                f"{X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%}",
            ],
            "Target mean": [float(np.mean(y_train)), float(np.mean(y_test))],
        }
    )
