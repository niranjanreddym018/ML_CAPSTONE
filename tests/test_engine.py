"""Integration tests against small, explicitly synthetic input files."""
from pathlib import Path

import numpy as np
import pandas as pd

from ml_studio.engine import FeatureBuilder, TRACKS, train_clustering, train_supervised, tune_supervised

ROOT = Path(__file__).resolve().parents[1]


def demo(filename):
    return pd.read_csv(ROOT / "sample_data" / filename)


def test_regression_models_tuning_and_fresh_prediction():
    raw = demo("demo_credit_risk.csv")
    run = train_supervised(raw, "Regression", TRACKS["Regression"], cap=250)
    assert len(run.pipelines) == 10, run.failures
    assert set(("R²", "MAE", "MSE", "RMSE")).issubset(run.metrics.columns)
    assert "LoanApproved" not in run.pipelines["Linear Regression"].named_steps["features"].transform(run.X_train)
    assert "LoanToIncomeRatio" in FeatureBuilder("Regression").transform(run.X_train)
    run = tune_supervised(run, ["Ridge Regression", "Decision Tree Regressor"])
    assert len(run.cv_scores) == 2
    assert np.isfinite(run.pipelines["Ridge Regression (tuned)"].predict(run.X_test.iloc[:3])).all()


def test_classification_models_and_positive_recall():
    raw = demo("demo_stroke.csv")
    run = train_supervised(raw, "Classification", TRACKS["Classification"], cap=350)
    assert len(run.pipelines) == 10, run.failures
    assert "Positive recall" in run.metrics.columns
    assert "ROC-AUC" in run.metrics.columns
    assert run.pipelines["Logistic Regression"].predict(run.X_test.iloc[:1]).shape == (1,)
    run = tune_supervised(run, ["Logistic Regression"])
    assert "Logistic Regression (tuned)" in run.pipelines


def test_cluster_models_plots_and_single_record_prediction():
    raw = demo("demo_obesity.csv")
    run = train_clustering(raw, "NObeyesdad", n_clusters=4, cap=None)
    assert len(run.predictions) == 2
    assert len(run.metrics) == 2
    assert len(run.elbow) >= 3
    assert len(run.linkage_table) == 4
    assert run.pca_points.shape == (len(run.data), 2)
    assert "NObeyesdad" not in run.preprocessor.named_steps["features"].transform(
        run.data.drop(columns=["NObeyesdad"]))
    single = run.data.drop(columns=["NObeyesdad"]).iloc[:1]
    cluster = run.kmeans.predict(run.preprocessor.transform(single))
    assert len(cluster) == 1


def test_feature_engineering_unseen_categories_and_missing_values():
    frame = pd.DataFrame({"LoanAmount":[40000], "AnnualIncome":[0],
                          "ApplicationDate":["bad-date"], "EducationLevel":["unknown"],
                          "LoanApproved":[1]})
    engineered = FeatureBuilder("Regression").transform(frame)
    assert pd.isna(engineered["LoanToIncomeRatio"].iloc[0])
    assert "ApplicationMonth" in engineered
    assert "LoanApproved" not in engineered
