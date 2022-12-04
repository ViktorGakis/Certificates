from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Any, Optional

sns.set(style="darkgrid")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rc('figure', figsize=(12,8))
plt.rc('lines', markersize=4)
plt.rc('font', size=30)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=20)
plt.rcParams["figure.autolayout"] = True

from sklearn.base import BaseEstimator
from sklearn import metrics

def metrics_calc(y_test, y_pred, clsfr:Optional[Any]=None) -> list[pd.DataFrame]:
    """Calculates all classifier relevant metrics, including adjusted ones to imbalanced classes.

    Args:
        y_test: test data
        y_pred: predicted data

    Returns:
        list[pd.DataFrame]: [metrics, metrics per class]
    """
    Metrics = {
        "Accuracy":metrics.accuracy_score(y_test, y_pred),
        "Precision":metrics.precision_score(y_test, y_pred),
        "Recall":metrics.recall_score(y_test, y_pred),
        "f1":metrics.f1_score(y_test, y_pred),
        "Bal Accuracy":metrics.balanced_accuracy_score(y_test, y_pred),
        "Av Precision":metrics.average_precision_score(y_test, y_pred),
        "AUC":metrics.roc_auc_score(y_test, y_pred),
        "Matthews CC":metrics.matthews_corrcoef(y_test, y_pred),
        "Cohen KS":metrics.cohen_kappa_score(y_test, y_pred),
    }
    Metrics = {k:[v] for k,v in Metrics.items()}
    Metrics = pd.DataFrame.from_dict(Metrics).T.rename({0:clsfr}, axis=1).rename_axis('Cls Metrics').round(2)
    Metrics_per_class = pd.DataFrame(metrics.precision_recall_fscore_support(y_test, y_pred)).rename({0:'Precision', 1:'Recall', 2:'f1', 3:'Support'}).rename_axis('Cls Metrics Per Class').round(2)
    Metrics_per_class = pd.concat({clsfr:Metrics_per_class}, axis=1) if clsfr else Metrics_per_class
    return [Metrics, Metrics_per_class]


def metrics_plot(cls:BaseEstimator, X_test:pd.DataFrame, y_test:pd.DataFrame) -> plt.Figure:
    """Plots in a grid, the ROC Curve the Precision-Recall Curve and the Confusion Matrix of a classifier.

    Args:
        cls (BaseEstimator): A trained classifier.
        X_test (pd.DataFrame): Independent test data.
        y_test (pd.DataFrame): Dependent test data.

    Returns:
        plt.Figure: A plt.subplot figure object containing 3 plots(ROC,PRC,Conf Matrix)
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8));
    metrics.RocCurveDisplay.from_estimator(cls, X_test, y_test, ax=axes[0]);
    axes[0].plot([0,1], [0,1], '--r', label='random cls');
    metrics.ConfusionMatrixDisplay.from_estimator(cls, X_test, y_test, ax=axes[1]);
    metrics.PrecisionRecallDisplay.from_estimator(cls, X_test, y_test, ax=axes[2]);
    axes[2].plot([0,1], [0.5,0.5], '--r', label='random cls');
    axes[0].legend();
    axes[2].legend();
    return fig;