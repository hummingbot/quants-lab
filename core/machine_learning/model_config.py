from dataclasses import dataclass
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]
    model_instance: Any
    one_vs_rest: bool = False
    scoring: str = "recall_macro"
    n_iter: int = 100
    cv: int = 4
    verbose: int = 10
    random_state: int = 42
    n_jobs:int = -1


# Random Forest Classifier Configuration
RF_CONFIG = ModelConfig(
    name="Random Forest",
    params={
        'estimator__n_estimators': [1000],  # Or use [int(x) for x in np.linspace(start=100, stop=1000, num=3)]
        'estimator__max_features': ['sqrt'],  # Or ['log2']
        'estimator__max_depth': [20, 55, 100],  # Or use [int(x) for x in np.linspace(10, 100, num=3)]
        'estimator__min_samples_split': [50, 100],
        'estimator__min_samples_leaf': [30, 50],
        'estimator__bootstrap': [True],
        'estimator__class_weight': ['balanced']
    },
    model_instance=RandomForestClassifier(),
    one_vs_rest=True,
)

# Logistic Regression Configuration
LR_CONFIG = ModelConfig(
    name="Logistic Regression",
    params={
        'estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'estimator__dual': [False, True],
        'estimator__tol': [1e-4, 1e-3, 1e-2, 1e-1],
        'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'estimator__fit_intercept': [True, False],
        'estimator__intercept_scaling': [0.001, 0.01, 0.1, 1, 10, 100],
        'estimator__class_weight': ['balanced'],
        'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'estimator__max_iter': [100, 200, 300, 400, 500],
        'estimator__warm_start': [False, True],
        'estimator__n_jobs': [-1]
    },
    model_instance=LogisticRegression(),
    one_vs_rest=False
)

# AdaBoost Classifier Configuration
ADABOOST_CONFIG = ModelConfig(
    name="AdaBoost",
    params={
        'estimator__learning_rate': [0.001, 0.01, 0.1, 1, 10],
        'estimator__algorithm': ['SAMME', 'SAMME.R']
    },
    model_instance=AdaBoostClassifier(),
    one_vs_rest=True
)

# XGBoost Classifier Configuration
XGB_CONFIG = ModelConfig(
    name="XGBoost",
    params={
        'estimator__booster': ['gbtree', 'gblinear', 'dart'],
        'estimator__n_estimators': [1000],  # Or use [int(x) for x in np.linspace(100, 1000, num=3)]
        'estimator__learning_rate': [0.001, 0.01, 0.1, 1, 10],
        'estimator__max_depth': [20, 55, 100],  # Or use [int(x) for x in np.linspace(10, 100, num=3)]
        'estimator__min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'estimator__subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'estimator__colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'estimator__colsample_bylevel': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'estimator__colsample_bynode': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10, 100],
        'estimator__reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10, 100]
    },
    model_instance=XGBClassifier(),
    one_vs_rest=True
)
