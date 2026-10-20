"""Core, testable ML engine for ML Studio.

Dataset-specific choices are adapted from the three original course notebooks.
Any supplied demo files are *synthetic*, not a reproduction of the research data.
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier,
    GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
TRACKS = {"Regression": "RiskScore", "Classification": "stroke", "Clustering": "NObeyesdad"}
DROP_IDS = {"Regression": ["LoanApproved", "ApplicationID", "CustomerID"],
            "Classification": ["id"], "Clustering": []}
EDUCATION = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless features, designed to run *also* on a single prediction record."""

    def __init__(self, track: str):
        self.track = track

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = pd.DataFrame(X).copy()
        if self.track == "Regression":
            if {"LoanAmount", "AnnualIncome"}.issubset(x.columns):
                loan = pd.to_numeric(x["LoanAmount"], errors="coerce")
                income = pd.to_numeric(x["AnnualIncome"], errors="coerce")
                x["LoanToIncomeRatio"] = loan / income.replace(0, np.nan)
            if "ApplicationDate" in x.columns:
                date = pd.to_datetime(x["ApplicationDate"], errors="coerce")
                x["ApplicationYear"] = date.dt.year.astype(float)
                x["ApplicationMonth"] = date.dt.month.astype(float)
                x = x.drop(columns=["ApplicationDate"])
            if "EducationLevel" in x.columns:
                x["EducationLevel"] = x["EducationLevel"].map(EDUCATION).astype(float)
        elif self.track == "Classification":
            if {"age", "avg_glucose_level"}.issubset(x.columns):
                x["age_glucose_risk"] = (
                    pd.to_numeric(x["age"], errors="coerce") *
                    pd.to_numeric(x["avg_glucose_level"], errors="coerce")
                )
        x = x.drop(columns=[c for c in DROP_IDS[self.track] if c in x.columns], errors="ignore")
        x = x.replace([np.inf, -np.inf], np.nan)
        return x


class IQRClipper(BaseEstimator, TransformerMixin):
    """Learn numeric IQR fences on train only; exclude binary columns."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(array, [25, 75], axis=0)
        spread = q3 - q1
        self.low_ = np.where(spread > 0, q1 - 1.5 * spread, -np.inf)
        self.high_ = np.where(spread > 0, q3 + 1.5 * spread, np.inf)
        for i in range(array.shape[1]):
            vals = np.unique(array[~np.isnan(array[:, i]), i])
            if len(vals) <= 2:
                self.low_[i], self.high_[i] = -np.inf, np.inf
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def get_preprocessor(X: pd.DataFrame, track: str) -> ColumnTransformer:
    feature_frame = FeatureBuilder(track).transform(X)
    numeric = feature_frame.select_dtypes(include="number").columns.to_list()
    categorical = [c for c in feature_frame.columns if c not in numeric]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain after excluding IDs/leakage columns.")
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if track in ("Regression", "Classification"):
        steps.append(("clip", IQRClipper()))
    steps.append(("scale", StandardScaler()))
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transforms, sparse_threshold=0)


def model_catalog(track: str):
    if track == "Regression":
        from xgboost import XGBRegressor
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1, max_iter=2500),
            "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2500),
            "Polynomial Regression": Pipeline([
                ("select", SelectKBest(score_func=f_regression, k=8)),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linear", Ridge(alpha=1.0)),
            ]),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=90, n_jobs=2, random_state=RANDOM_STATE),
            "Gradient Boosting (XGBoost)": XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08,
                                                       n_jobs=2, random_state=RANDOM_STATE),
            "Support Vector Regressor": SVR(kernel="rbf"),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=7),
        }
    if track == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
            "SVM (SVC)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                                    n_jobs=2, random_state=RANDOM_STATE),
            "AdaBoost": AdaBoostClassifier(n_estimators=70, random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "Bagging (Decision Tree)": BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
                n_estimators=60, n_jobs=2, random_state=RANDOM_STATE),
            "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE),
        }
    raise ValueError(f"Unknown supervised track: {track}")


def _supervised_metrics(track, y_true, pred, proba=None, classes=None):
    if track == "Regression":
        mse = mean_squared_error(y_true, pred)
        return {"R²": r2_score(y_true, pred), "MAE": mean_absolute_error(y_true, pred),
                "MSE": mse, "RMSE": np.sqrt(mse)}
    labels = list(classes)
    row = {"Accuracy": accuracy_score(y_true, pred),
           "Precision (weighted)": precision_score(y_true, pred, average="weighted", zero_division=0),
           "Recall (weighted)": recall_score(y_true, pred, average="weighted", zero_division=0),
           "F1 (weighted)": f1_score(y_true, pred, average="weighted", zero_division=0),
           "F1 (macro)": f1_score(y_true, pred, average="macro", zero_division=0)}
    if len(labels) == 2:
        positive = labels[-1]
        row["Positive recall"] = recall_score(y_true, pred, pos_label=positive, zero_division=0)
        row["Positive precision"] = precision_score(y_true, pred, pos_label=positive, zero_division=0)
    if proba is not None and len(np.unique(y_true)) > 1:
        try:
            if len(labels) == 2:
                positive = labels[-1]
                row["ROC-AUC"] = roc_auc_score(y_true, proba[:, labels.index(positive)], labels=labels)
            else:
                row["ROC-AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=labels)
        except ValueError:
            row["ROC-AUC"] = np.nan
    return row


def _fit_pipe(pipe, track, y_train, X_train):
    model = pipe.named_steps["model"]
    params = inspect.signature(model.fit).parameters
    if track == "Classification" and "sample_weight" in params and model.__class__.__name__ in (
        "AdaBoostClassifier", "GradientBoostingClassifier", "MLPClassifier"):
        pipe.fit(X_train, y_train, model__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        pipe.fit(X_train, y_train)
    return pipe


@dataclass
class SupervisedRun:
    track: str
    target: str
    source_rows: int
    used_rows: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metrics: pd.DataFrame
    pipelines: dict
    preds: dict
    probabilities: dict
    failures: dict
    cv_scores: dict


def train_supervised(df: pd.DataFrame, track: str, target: str, cap: int | None = 5000,
                     callback: Callable[[int, int, str], None] | None = None) -> SupervisedRun:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing.")
    data = df.dropna(subset=[target]).drop_duplicates().copy()
    if track == "Classification" and target == "stroke" and "gender" in data.columns:
        data = data.loc[data["gender"] != "Other"].copy()
    if cap and len(data) > cap:
        # Random sampling is used *before* the split and is disclosed in the UI.
        data = data.sample(n=cap, random_state=RANDOM_STATE)
    if len(data) < 30:
        raise ValueError("At least 30 complete target rows are needed for a meaningful holdout evaluation.")
    y = data[target]
    X = data.drop(columns=[target, *DROP_IDS[track]], errors="ignore")
    if track == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        good = y.notna()
        y, X = y.loc[good], X.loc[good]
        if y.nunique() < 2:
            raise ValueError("Regression requires a numeric target containing multiple distinct values.")
        stratify = None
        try:
            bins = pd.qcut(y, q=min(10, max(2, len(y) // 20)), duplicates="drop")
            if bins.value_counts().min() >= 2:
                stratify = bins
        except ValueError:
            pass
    else:
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        counts = y.value_counts()
        if counts.min() < 2:
            raise ValueError("Every target class must have at least two records for a stratified split.")
        if y.nunique() > min(20, len(y) // 10):
            raise ValueError("The classification target has too many distinct classes. Choose a categorical target.")
        stratify = y
    if stratify is not None and len(pd.Series(stratify).unique()) > max(1, int(len(y) * 0.2)):
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    # Determine columns from training records only; each model's preprocessing is fitted on train.
    prep = get_preprocessor(X_train, track)
    catalog = model_catalog(track)
    pipelines, preds, probs, results, failures = {}, {}, {}, [], {}
    for i, (name, model) in enumerate(catalog.items(), 1):
        # SelectKBest k cannot exceed number of preprocessed features on small generic datasets.
        if track == "Regression" and name == "Polynomial Regression":
            sample_features = clone(prep).fit_transform(FeatureBuilder(track).transform(X_train))
            model.set_params(select__k=min(8, sample_features.shape[1]))
        pipe = Pipeline([("features", FeatureBuilder(track)), ("preprocess", clone(prep)), ("model", clone(model))])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _fit_pipe(pipe, track, y_train, X_train)
            p = pipe.predict(X_test)
            pr = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
            classes = list(pipe.classes_) if track == "Classification" else None
            scores = _supervised_metrics(track, y_test, p, pr, classes)
            pipelines[name], preds[name], probs[name] = pipe, p, pr
            results.append({"Model": name, **scores})
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
        if callback:
            callback(i, len(catalog), name)
    if not results:
        raise ValueError("All training attempts failed; inspect the target and feature columns.")
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("R²" if track == "Regression" else
                                      ("ROC-AUC" if "ROC-AUC" in result_df.columns else "F1 (macro)"),
                                      ascending=False, na_position="last").reset_index(drop=True)
    return SupervisedRun(track, target, len(df), len(X), X_train, X_test, y_train, y_test,
                         result_df, pipelines, preds, probs, failures, {})


TUNING_GRIDS = {
    "Regression": {
        "Ridge Regression": {"model__alpha": [0.1, 1, 10, 50]},
        "Decision Tree Regressor": {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5]},
        "K-Nearest Neighbors Regressor": {"model__n_neighbors": [5, 9, 15]},
        "Random Forest Regressor": {"model__max_depth": [8, None], "model__min_samples_leaf": [1, 4]},
    },
    "Classification": {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "Decision Tree": {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [1, 5]},
        "Random Forest": {"model__min_samples_leaf": [1, 5, 10], "model__max_depth": [5, None]},
        "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 9]},
    },
}


def tune_supervised(run: SupervisedRun, names: list[str]):
    for name in names:
        if name not in TUNING_GRIDS[run.track]:
            raise ValueError(f"Tuning grid not available for {name}.")
        basic = clone(run.pipelines[name])
        scoring = ("r2" if run.track == "Regression" else
                   "roc_auc" if run.y_train.nunique() == 2 else "f1_macro")
        cv = GridSearchCV(basic, TUNING_GRIDS[run.track][name], scoring=scoring, cv=3, n_jobs=1)
        cv.fit(run.X_train, run.y_train)
        best = cv.best_estimator_
        pred = best.predict(run.X_test)
        proba = best.predict_proba(run.X_test) if hasattr(best, "predict_proba") else None
        classes = list(best.classes_) if run.track == "Classification" else None
        tag = f"{name} (tuned)"
        run.pipelines[tag], run.preds[tag], run.probabilities[tag] = best, pred, proba
        row = {"Model": tag, **_supervised_metrics(run.track, run.y_test, pred, proba, classes)}
        run.metrics = pd.concat([run.metrics, pd.DataFrame([row])], ignore_index=True)
        run.cv_scores[tag] = {"CV score": float(cv.best_score_), "Best parameters": cv.best_params_}
    run.metrics = run.metrics.sort_values("R²" if run.track == "Regression" else
                                          "ROC-AUC" if "ROC-AUC" in run.metrics else "F1 (macro)",
                                          ascending=False, na_position="last").reset_index(drop=True)
    return run


@dataclass
class ClusteringRun:
    data: pd.DataFrame
    target: str | None
    preprocessor: Pipeline
    transformed: np.ndarray
    kmeans: KMeans
    agglomerative: AgglomerativeClustering
    predictions: dict
    metrics: pd.DataFrame
    elbow: pd.DataFrame
    linkage_table: pd.DataFrame
    pca_points: np.ndarray
    explained_ratio: np.ndarray
    dendro_linkage: np.ndarray
    dendro_n: int


def _internal_scores(x, labels):
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan,
                "Smallest cluster": counts.min()}
    return {"Silhouette": silhouette_score(x, labels, sample_size=min(700, len(x)), random_state=RANDOM_STATE),
            "Davies-Bouldin": davies_bouldin_score(x, labels),
            "Calinski-Harabasz": calinski_harabasz_score(x, labels),
            "Smallest cluster": int(counts.min())}


def train_clustering(df: pd.DataFrame, target: str | None, n_clusters: int = 4,
                     linkage_method: str = "ward", cap: int | None = 2100,
                     callback=None) -> ClusteringRun:
    data = df.drop_duplicates().copy()
    if cap and len(data) > cap:
        data = data.sample(n=cap, random_state=RANDOM_STATE)
    data = data.reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("Clustering requires at least 10 rows.")
    if n_clusters >= len(data) or n_clusters < 2:
        raise ValueError("Number of clusters must be between 2 and n-1.")
    X_raw = data.drop(columns=[target] if target and target in data else [], errors="ignore")
    prep = Pipeline([("features", FeatureBuilder("Clustering")),
                     ("preprocess", get_preprocessor(X_raw, "Clustering"))])
    X = np.asarray(prep.fit_transform(X_raw), dtype=float)
    elbow = []
    for k in range(2, min(10, len(data)-1)+1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        elbow.append({"k": k, "Inertia": model.inertia_,
                      "Silhouette": _internal_scores(X, model.labels_)["Silhouette"]})
    if callback:
        callback("Elbow curve complete")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE).fit(X)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method).fit(X)
    preds = {"K-Means": km.labels_, f"Agglomerative ({linkage_method})": agg.labels_}
    metrics = pd.DataFrame([{"Model": name, **_internal_scores(X, labels)} for name, labels in preds.items()])
    link_rows = []
    for method in ["ward", "complete", "average", "single"]:
        candidate = agg if method == linkage_method else AgglomerativeClustering(
            n_clusters=n_clusters, linkage=method).fit(X)
        link_rows.append({"Linkage": method, **_internal_scores(X, candidate.labels_)})
    if callback:
        callback("Cluster and linkage metrics complete")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = pca.fit_transform(X)
    rng = np.random.default_rng(RANDOM_STATE)
    ids = np.sort(rng.choice(len(X), size=min(350, len(X)), replace=False))
    dendro_matrix = linkage(X[ids], method=linkage_method)
    return ClusteringRun(data, target, prep, X, km, agg, preds, metrics,
                         pd.DataFrame(elbow), pd.DataFrame(link_rows), emb,
                         pca.explained_variance_ratio_, dendro_matrix, len(ids))
