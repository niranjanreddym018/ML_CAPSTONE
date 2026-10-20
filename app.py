"""ML Studio — Streamlit Community Cloud entry point.

Run: streamlit run app.py
Original datasets are not included; use CSV upload, or synthetic demo mode.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score, classification_report, confusion_matrix,
    normalized_mutual_info_score, precision_recall_curve, roc_curve,
)

from ml_studio.preprocessing_view import preprocessing_ui
from ml_studio.engine import (
    RANDOM_STATE, TRACKS, TUNING_GRIDS, train_clustering,
    train_supervised, tune_supervised,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="ML Studio | Interactive ML Workbench", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {--accent: #2563eb;}
.block-container {max-width: 1500px; padding-top: 1.4rem;}
[data-testid="stMetric"] {background:#f6f8fc;border:1px solid #e7edf6; padding:14px 16px;border-radius:12px;}
@media (prefers-color-scheme: dark) {
 [data-testid="stMetric"] {background:#1b2636; border-color:#3b4b62;}
}
.hero {padding: 1.3rem 1.55rem; background: linear-gradient(105deg,#152a54,#294f94 65%,#0e7685);
       border-radius: 18px; color:#fff; margin-bottom:1.1rem;}
.hero h1 {font-size:2.0rem; margin:0 0 0.35rem; letter-spacing:-.045em; color:#fff;}
.hero p {margin:0; color:#d6e5ff; line-height:1.5;}
.eyebrow {font-size:.75rem; letter-spacing:.15em; text-transform:uppercase;
          font-weight:700; color:#9ad3ff; margin-bottom:.4rem;}
.subtle {color:#6f7b91; font-size:.94rem;}
</style>
""", unsafe_allow_html=True)


def heading(text, detail=None):
    st.subheader(text)
    if detail:
        st.caption(detail)


def download_frame(df: pd.DataFrame, name: str, label: str):
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv")


def load_input(track: str):
    upload = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"], key=f"upload_{track}")
    demo = st.sidebar.checkbox("Try synthetic demo data", key=f"demo_{track}",
                               help="Demonstration values only — not the course's original datasets.")
    if upload is not None:
        raw = upload.getvalue()
        origin = upload.name
        try:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")
            return None, None, None
        key = hashlib.sha256(raw).hexdigest()
    elif demo:
        filename = {"Regression": "demo_credit_risk.csv", "Classification": "demo_stroke.csv",
                    "Clustering": "demo_obesity.csv"}[track]
        raw = (ROOT / "sample_data" / filename).read_bytes()
        df = pd.read_csv(io.BytesIO(raw))
        origin = filename + " (synthetic)"
        key = hashlib.sha256(raw).hexdigest()
    else:
        return None, None, None
    if df.empty or df.shape[1] < 2:
        st.error("CSV needs at least two columns and one data row.")
        return None, None, None
    st.sidebar.caption(f"**Loaded:** {origin} · {len(df):,} rows × {df.shape[1]} columns")
    return df, key, origin


def overview(track):
    st.markdown('''<div class="hero"><div class="eyebrow">23CSE301 · Machine Learning Capstone</div>
      <h1>ML Studio</h1><p>Upload a dataset. Explore and clean it. Compare models. Explain the results.<br>
      Three projects, one interactive machine learning workspace.</p></div>''', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, number, title, sub in zip(cols, ["10", "10", "2"],
                                       ["Regression", "Classification", "Clustering"],
                                       ["RiskScore · Loan", "Stroke detection", "Obesity lifestyle patterns"]):
        with col:
            st.metric(title, number + " algorithms" if number != "2" else "2 algorithms")
            st.caption(sub)
    st.info(f"Current track: **{track}**. Upload its original CSV in the sidebar, or select the clearly labeled synthetic demo.")
    st.markdown("**Your original notebook files are included in the project ZIP** under `notebooks/`. The raw CSVs were not attached, so real review results require the actual datasets.")
    st.markdown("**Project flow:** Dataset & EDA → Preprocessing → Training & Comparison → Detailed Evaluation → Prediction → Exports")


def explorer(df, track, target, origin):
    heading("Dataset & Exploratory Analysis", "Review source data and patterns before training.")
    if "synthetic" in origin:
        st.warning("Synthetic example dataset. Any reported metrics are a UI demonstration, not assignment results.")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    with st.expander("Column types, missing values and summary statistics", expanded=False):
        audit = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values,
                              "Missing": df.isna().sum().values,
                              "Unique": df.nunique(dropna=True).values})
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.dataframe(df.describe(include="all").T.fillna(""), use_container_width=True)
    heading("Explore distributions")
    a, b = st.columns([1, 2])
    col = a.selectbox("Column", df.columns.tolist(), key=f"eda_col_{track}")
    if pd.api.types.is_numeric_dtype(df[col]):
        fig = px.histogram(df, x=col, nbins=35, marginal="box",
                           color=target if target and target in df and target != col and track != "Regression" else None,
                           template="plotly_white", title=f"Distribution of {col}")
    else:
        counts = df[col].fillna("Missing").astype(str).value_counts().head(25)
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": col, "y": "Count"},
                     template="plotly_white", title=f"Category counts: {col}")
    b.plotly_chart(fig, use_container_width=True)
    nums = df.select_dtypes(include="number").columns.tolist()
    if len(nums) >= 2:
        left, right = st.columns(2)
        with left:
            heading("Correlation heatmap")
            use_cols = st.multiselect("Numeric columns", nums,
                                      default=nums[:min(12, len(nums))], key=f"heat_{track}")
            if len(use_cols) >= 2:
                corr = df[use_cols].corr()
                st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                          aspect="auto", title="Pearson correlation"), use_container_width=True)
        with right:
            heading("Feature relationships")
            x = st.selectbox("X-axis", nums, index=0, key=f"scatterx_{track}")
            y = st.selectbox("Y-axis", nums, index=1, key=f"scattery_{track}")
            view = df.sample(n=min(len(df), 2500), random_state=RANDOM_STATE)
            fig = px.scatter(view, x=x, y=y,
                             color=target if target and track != "Regression" and target in view else None,
                             opacity=0.6, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    if target and target in df:
        heading("Target inspection")
        if track == "Regression" or (track == "Clustering" and pd.api.types.is_numeric_dtype(df[target])):
            st.plotly_chart(px.histogram(df, x=target, nbins=35, template="plotly_white"),
                            use_container_width=True)
        else:
            counts = df[target].value_counts(dropna=False).rename_axis("Class").reset_index(name="Rows")
            st.plotly_chart(px.bar(counts, x="Class", y="Rows", template="plotly_white"),
                            use_container_width=True)
    download_frame(df.head(1000), "dataset_preview.csv", "Download data preview")


def training_ui(df, track, target, key, origin):
    heading("Train & Compare", "Training is started by you — no model is silently retrained on each page refresh.")
    if "synthetic" in origin:
        st.warning("Synthetic dataset selected. Switch to the original CSV before reporting your capstone metrics.")
    cols = st.columns([1.5, 1, 1])
    with cols[0]:
        n_label = st.selectbox("Experiment size", ["Fast · 2,000 rows", "Standard · 5,000 rows", "Full dataset"],
                               index=1 if track == "Regression" else 2, key=f"cap_{track}")
        cap = 2000 if n_label.startswith("Fast") else 5000 if n_label.startswith("Standard") else None
    with cols[1]:
        if track == "Clustering":
            k = st.number_input("Number of clusters (k)", min_value=2, max_value=min(10, len(df)-1),
                                value=min(4, len(df)-1), key=f"k_{track}")
        else:
            k = 4
            st.caption("80% training · 20% held-out test")
    with cols[2]:
        if track == "Clustering":
            method = st.selectbox("Hierarchical linkage", ["ward", "complete", "average", "single"])
        else:
            method = "ward"
            st.caption("Preprocessing fitted only on the training data")
    signature = (track, key, target, cap, k, method)
    state_key = f"run_{track}"
    if st.session_state.get(f"sig_{track}") != signature:
        st.session_state.pop(state_key, None)
    if st.button("▶ Train and evaluate", type="primary", use_container_width=True):
        progress = st.progress(0, text="Preparing the experiment...")
        try:
            if track == "Clustering":
                def update(message):
                    progress.progress(75, text=message)
                with st.spinner("Running K-Means, hierarchical clustering, and PCA..."):
                    run = train_clustering(df, target, int(k), method, cap, update)
            else:
                def update(i, total, model):
                    progress.progress(int(100*i/total), text=f"Training {i}/{total}: {model}")
                with st.spinner("Training all 10 models on the same split..."):
                    run = train_supervised(df, track, target, cap, update)
            st.session_state[state_key] = run
            st.session_state[f"sig_{track}"] = signature
            progress.empty()
            st.success("Training finished. Open the Evaluation, Prediction, or Exports section.")
        except Exception as exc:
            progress.empty()
            st.error(f"Training failed: {type(exc).__name__}: {exc}")
    run = st.session_state.get(state_key)
    if run is None:
        st.info("Click **Train and evaluate** to generate your comparison table and visualizations.")
        return
    if track == "Clustering":
        st.dataframe(run.metrics.round(4), hide_index=True, use_container_width=True)
        st.caption("Silhouette and Calinski-Harabasz: higher is better. Davies-Bouldin: lower is better. "
                   "Silhouette is sampled for speed on larger datasets.")
    else:
        a, b, c = st.columns(3)
        a.metric("Rows used", f"{run.used_rows:,}")
        b.metric("Train rows", f"{len(run.X_train):,}")
        c.metric("Test rows", f"{len(run.X_test):,}")
        st.dataframe(run.metrics.round(4), use_container_width=True, hide_index=True)
        score = "R²" if track == "Regression" else "ROC-AUC" if "ROC-AUC" in run.metrics else "F1 (macro)"
        plot = px.bar(run.metrics, x=score, y="Model", orientation="h", template="plotly_white",
                      title=f"Model comparison by {score}")
        plot.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
        st.plotly_chart(plot, use_container_width=True)
        if run.failures:
            with st.expander(f"{len(run.failures)} model(s) could not train"):
                st.json(run.failures)
        heading("Optional hyperparameter tuning (3-fold CV)")
        st.caption("Two preset algorithms can be tuned on the training split only; test rows remain held out. "
                   "Tuned results appear as separate rows so you can compare baseline versus tuned.")
        options = [x for x in TUNING_GRIDS[track] if x in run.pipelines and f"{x} (tuned)" not in run.pipelines]
        defaults = ["Ridge Regression", "Decision Tree Regressor"] if track == "Regression" else ["Logistic Regression", "Random Forest"]
        selected = st.multiselect("Choose models to tune", options,
                                  default=[s for s in defaults if s in options], key=f"tune_names_{track}")
        if st.button("Tune selected models", disabled=not selected):
            try:
                with st.spinner("Searching hyperparameters using training-only cross-validation..."):
                    st.session_state[state_key] = tune_supervised(run, selected)
                st.success("Tuned results added to the comparison.")
                st.rerun()
            except Exception as exc:
                st.error(f"Tuning failed: {exc}")
        if run.cv_scores:
            st.json(run.cv_scores)


def visual_supervised(run, track):
    heading("Detailed Model Evaluation")
    model_name = st.selectbox("Model", run.metrics["Model"].tolist(), key=f"detail_model_{track}")
    if model_name not in run.pipelines:
        st.warning("Please re-train this model.")
        return
    pred = run.preds[model_name]
    if track == "Regression":
        residual = np.asarray(run.y_test) - pred
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(x=np.asarray(run.y_test), y=pred, labels={"x":"Actual RiskScore", "y":"Predicted RiskScore"},
                             title="Predicted vs actual", template="plotly_white", opacity=0.55)
            m1 = min(np.min(pred), run.y_test.min()); m2 = max(np.max(pred), run.y_test.max())
            fig.add_trace(go.Scatter(x=[m1, m2], y=[m1, m2], mode="lines", name="Ideal", line=dict(dash="dash")))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(x=pred, y=residual, labels={"x":"Predicted", "y":"Actual - predicted"},
                                       title="Residual plot", template="plotly_white", opacity=0.55),
                            use_container_width=True)
        st.caption("Residuals closer to zero indicate smaller prediction errors. R² can be negative for a model worse than predicting the test-set mean.")
    else:
        classes = list(run.pipelines[model_name].classes_)
        cm = confusion_matrix(run.y_test, pred, labels=classes)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(cm, x=[str(x) for x in classes], y=[str(x) for x in classes],
                            text_auto=True, color_continuous_scale="Blues", template="plotly_white",
                            labels=dict(x="Predicted", y="Actual"), title="Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            report = classification_report(run.y_test, pred, labels=classes, output_dict=True, zero_division=0)
            st.markdown("**Per-class precision, recall and F1**")
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
        proba = run.probabilities.get(model_name)
        if proba is not None and len(classes) == 2:
            positive = classes[-1]
            prob = proba[:, classes.index(positive)]
            fpr, tpr, _ = roc_curve(run.y_test, prob, pos_label=positive)
            prec, rec, _ = precision_recall_curve((run.y_test == positive).astype(int), prob)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.area(x=fpr, y=tpr, labels={"x":"False positive rate", "y":"True positive rate"},
                              title=f"ROC curve · positive={positive}", template="plotly_white")
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash"), name="Chance"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.plotly_chart(px.line(x=rec, y=prec, labels={"x":"Recall", "y":"Precision"},
                                        title="Precision–recall curve", template="plotly_white"),
                                use_container_width=True)
            if run.target == "stroke":
                st.warning("Stroke classes are highly imbalanced in the original dataset. A high accuracy or weighted F1 "
                           "does not mean the model detects positive stroke cases well. Examine positive recall and ROC-AUC. "
                           "Predictions are educational and not medical advice.")
    pipe = run.pipelines[model_name]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        try:
            features = pipe.named_steps["preprocess"].get_feature_names_out()
            values = (model.feature_importances_ if hasattr(model, "feature_importances_") else
                      np.abs(np.asarray(model.coef_)).mean(axis=0))
            if len(features) == len(values):
                feat = pd.DataFrame({"Feature": features, "Importance": values}).nlargest(18, "Importance")
                st.plotly_chart(px.bar(feat, x="Importance", y="Feature", orientation="h",
                                       template="plotly_white", title="Important model features"),
                                use_container_width=True)
        except Exception:
            pass


def visual_clustering(run):
    heading("Clustering Visualizations", "Both methods use the same prepared features; the true obesity label is never used to fit clusters.")
    st.markdown("**Elbow curve and silhouette vs. k**")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.line(run.elbow, x="k", y="Inertia", markers=True, template="plotly_white",
                                title="K-Means elbow · Inertia"), use_container_width=True)
    with b:
        st.plotly_chart(px.line(run.elbow, x="k", y="Silhouette", markers=True, template="plotly_white",
                                title="Silhouette vs. k"), use_container_width=True)
    st.caption("An elbow is a heuristic, not an automatic proof of the best number of clusters.")
    st.markdown("**Hierarchical linkage strategy comparison**")
    st.dataframe(run.linkage_table.round(4), use_container_width=True, hide_index=True)
    st.caption("Single/average linkage can isolate tiny outlier clusters. Check 'Smallest cluster' before choosing the highest silhouette.")
    st.markdown("**Hierarchical dendrogram (representative 350-row sample maximum)**")
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(run.dendro_linkage, truncate_mode="lastp", p=25, no_labels=True, ax=ax)
    ax.set(xlabel="Merged cluster groups", ylabel="Merge distance", title="Truncated dendrogram")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Dendrogram built from a random sample of {run.dendro_n} rows for responsiveness; clusters and scores use all experiment rows.")
    st.markdown("**PCA projections of both algorithms (2 principal components)**")
    st.caption("Explained variance PC1 + PC2: " + f"{sum(run.explained_ratio):.1%}" +
               ". Colors identify arbitrary cluster IDs and are not matched across algorithms.")
    cols = st.columns(2)
    for (name, labels), col in zip(run.predictions.items(), cols):
        d = pd.DataFrame({"PC1": run.pca_points[:, 0], "PC2": run.pca_points[:, 1],
                          "Cluster": pd.Series(labels).astype(str)})
        with col:
            st.plotly_chart(px.scatter(d, x="PC1", y="PC2", color="Cluster", opacity=0.7,
                                       template="plotly_white", title=name), use_container_width=True)
    with st.expander("Optional: t-SNE embedding (may take a minute)"):
        st.caption("t-SNE is for exploration only. The underlying clustering models were fitted in the full feature space.")
        if st.button("Compute t-SNE", key="tsne_button"):
            n = min(1000, len(run.data))
            rng = np.random.default_rng(RANDOM_STATE)
            indices = np.sort(rng.choice(len(run.data), size=n, replace=False))
            with st.spinner("Computing t-SNE..."):
                embedding = TSNE(n_components=2, perplexity=min(30, n-1), init="pca",
                                 random_state=RANDOM_STATE, max_iter=600).fit_transform(run.transformed[indices])
            st.session_state["tsne_points"] = (id(run), indices, embedding)
        cache = st.session_state.get("tsne_points")
        if cache and cache[0] == id(run):
            _, indices, emb = cache
            cols = st.columns(2)
            for (name, labels), col in zip(run.predictions.items(), cols):
                d = pd.DataFrame({"tSNE1":emb[:,0], "tSNE2":emb[:,1],
                                  "Cluster":pd.Series(labels[indices]).astype(str)})
                with col:
                    st.plotly_chart(px.scatter(d, x="tSNE1", y="tSNE2", color="Cluster",
                                               template="plotly_white", title=f"t-SNE · {name}"),
                                    use_container_width=True)
    if run.target and run.target in run.data:
        heading("Post-hoc label comparison", "True classes are only used here, after clustering.")
        rows = []
        cols = st.columns(2)
        for (name, labels), col in zip(run.predictions.items(), cols):
            rows.append({"Model": name,
                         "Adjusted Rand Index": adjusted_rand_score(run.data[run.target], labels),
                         "Normalized Mutual Information": normalized_mutual_info_score(run.data[run.target], labels)})
            tab = pd.crosstab(pd.Series(labels, name="Cluster"), run.data[run.target])
            with col:
                st.plotly_chart(px.imshow(tab, text_auto=True, aspect="auto", title=name,
                                          labels={"x":"Actual category", "y":"Cluster"}), use_container_width=True)
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    numeric = run.data.select_dtypes(include="number").columns.tolist()
    if numeric:
        heading("K-Means cluster profiles")
        prof = run.data[numeric].copy()
        if {"Height", "Weight"}.issubset(prof.columns):
            prof["BMI (interpretation only)"] = prof["Weight"] / prof["Height"].replace(0, np.nan).pow(2)
        prof["Cluster"] = run.kmeans.labels_
        sizes = prof.groupby("Cluster").size().rename("Size")
        st.dataframe(prof.groupby("Cluster").mean(numeric_only=True).round(2).join(sizes),
                     use_container_width=True)


def _default_record(x_train):
    output = {}
    for c in x_train:
        s = x_train[c]
        if pd.api.types.is_numeric_dtype(s):
            output[c] = float(s.median()) if s.notna().any() else 0.0
        else:
            mode = s.dropna().mode()
            output[c] = str(mode.iloc[0]) if len(mode) else ""
    return output


def predict_ui(run, track):
    heading("Make Predictions", "Use values for the original input features. The saved fitted preprocessing is applied automatically.")
    if track == "Clustering":
        models = ["K-Means"]
        x_train = run.data.drop(columns=[run.target] if run.target else [], errors="ignore")
        st.info("Agglomerative hierarchical clustering has no native `.predict()` method. New-row cluster assignment uses K-Means only.")
    else:
        models = list(run.pipelines)
        x_train = run.X_train
    name = st.selectbox("Trained model", models, key=f"pred_model_{track}")
    record = _default_record(x_train)
    st.caption(f"{len(record)} input fields. Defaults are training-data medians or most frequent values. Review every value before predicting.")
    with st.form(f"prediction_form_{track}"):
        cols = st.columns(3)
        for i, (field, default) in enumerate(record.items()):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(x_train[field]):
                    record[field] = st.number_input(field, value=float(default), format="%.4f",
                                                     key=f"v_{track}_{field}")
                else:
                    options = x_train[field].dropna().astype(str).value_counts().index.to_list()
                    if field == "ApplicationDate":
                        record[field] = st.text_input(field + " (YYYY-MM-DD)", value=str(default),
                                                      key=f"v_{track}_{field}")
                    elif len(options) <= 50:
                        record[field] = st.selectbox(field, options, key=f"v_{track}_{field}")
                    else:
                        record[field] = st.text_input(field, value=str(default), key=f"v_{track}_{field}")
        submit = st.form_submit_button("Predict for this record", type="primary")
    if submit:
        try:
            single = pd.DataFrame([record])
            if track == "Clustering":
                transformed = run.preprocessor.transform(single)
                result = run.kmeans.predict(transformed)[0]
                st.success(f"Assigned to K-Means **Cluster {result}**")
            else:
                pipe = run.pipelines[name]
                result = pipe.predict(single)[0]
                if track == "Regression":
                    st.success(f"Predicted {run.target}: **{float(result):.3f}**")
                else:
                    st.success(f"Predicted {run.target}: **{result}**")
                    if hasattr(pipe, "predict_proba"):
                        probabilities = pipe.predict_proba(single)[0]
                        st.dataframe(pd.DataFrame({"Class":pipe.classes_, "Estimated probability": probabilities}),
                                     hide_index=True, use_container_width=True)
                        st.caption("Probabilities are model outputs, not guarantees or a calibrated clinical assessment.")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
    heading("Batch CSV predictions")
    st.caption("Upload unlabeled rows with exactly the input columns shown above (extra columns are ignored).")
    batch = st.file_uploader("Batch prediction CSV", type="csv", key=f"batch_{track}")
    if batch is not None:
        try:
            newdata = pd.read_csv(batch)
            missing = [col for col in x_train if col not in newdata]
            if missing:
                st.error("Missing input columns: " + ", ".join(missing))
            else:
                new_x = newdata[x_train.columns]
                if track == "Clustering":
                    pred = run.kmeans.predict(run.preprocessor.transform(new_x))
                else:
                    pred = run.pipelines[name].predict(new_x)
                output = newdata.copy()
                output["ML_Studio_Prediction"] = pred
                st.dataframe(output.head(20), use_container_width=True)
                download_frame(output, f"{track.lower()}_batch_predictions.csv", "Download batch predictions")
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
    if track in ["Regression", "Classification"]:
        st.warning("Outputs are for an academic demonstration, not financial lending decisions or medical diagnosis.")


def export_ui(run, track):
    heading("Download Experiment Results")
    download_frame(run.metrics, f"{track.lower()}_model_comparison.csv", "Download metrics table (CSV)")
    if track == "Clustering":
        outcome = run.data.copy()
        for label, clusters in run.predictions.items():
            outcome[label + "_cluster"] = clusters
        download_frame(outcome, "clustering_assignments.csv", "Download cluster assignments (CSV)")
        download_frame(run.elbow, "kmeans_elbow.csv", "Download elbow curve data (CSV)")
        download_frame(run.linkage_table, "hierarchical_linkages.csv", "Download linkage comparison (CSV)")
        artifact = {"preprocessor":run.preprocessor, "kmeans":run.kmeans,
                    "description":"For cluster inference call kmeans.predict(preprocessor.transform(new_rows))."}
        artifact_name = "ml_studio_kmeans.joblib"
    else:
        model_name = st.selectbox("Export model", list(run.pipelines), key=f"export_model_{track}")
        held_out = run.X_test.copy()
        held_out[f"actual_{run.target}"] = np.asarray(run.y_test)
        held_out[f"predicted_{run.target}"] = np.asarray(run.preds[model_name])
        if track == "Regression":
            held_out["residual"] = held_out[f"actual_{run.target}"] - held_out[f"predicted_{run.target}"]
        elif run.probabilities[model_name] is not None:
            for i, label in enumerate(run.pipelines[model_name].classes_):
                held_out[f"P({label})"] = run.probabilities[model_name][:,i]
        download_frame(held_out, f"{track.lower()}_test_predictions.csv", "Download held-out predictions (CSV)")
        artifact = run.pipelines[model_name]
        artifact_name = f"ml_studio_{track.lower()}_pipeline.joblib"
    # Only saved models trained here, not arbitrary user-uploaded pickles.
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=2)
    st.download_button("Download fitted model (joblib)", data=buffer.getvalue(), file_name=artifact_name,
                       mime="application/octet-stream")
    st.caption("Only open joblib files from trusted sources. Downloaded models are for the Python environment, not for browser execution.")
    meta = {"track":track, "random_state": RANDOM_STATE, "comparison":run.metrics.replace({np.nan:None}).to_dict("records")}
    if track != "Clustering":
        meta.update({"target": run.target, "training_rows": len(run.X_train),
                     "test_rows": len(run.X_test), "source_rows":run.source_rows,
                     "experiment_rows":run.used_rows, "tuning": run.cv_scores})
    else:
        meta.update({"ground_truth_label": run.target, "experiment_rows":len(run.data),
                     "pca_variance_ratio":run.explained_ratio.tolist()})
    st.download_button("Download run summary (JSON)",
                       data=json.dumps(meta, indent=2, default=str).encode(),
                       file_name=f"ml_studio_{track.lower()}_run.json", mime="application/json")


def main():
    st.sidebar.title("🧪 ML Studio")
    track = st.sidebar.radio("Select a machine learning track", list(TRACKS), key="main_track")
    st.sidebar.divider()
    df, key, origin = load_input(track)
    st.sidebar.divider()
    pages = ["Overview", "Dataset & EDA", "Preprocessing", "Train & Compare", "Evaluation", "Prediction", "Downloads"]
    page = st.sidebar.radio("Workspace", pages, key=f"page_{track}")
    st.sidebar.caption("Notebook-aligned capstone · Python + scikit-learn + Streamlit")
    if page == "Overview":
        overview(track)
        return
    if df is None:
        st.title(page)
        st.info("Upload your dataset in the left sidebar, or turn on synthetic demo data.")
        st.code({"Regression":"Loan.csv", "Classification":"healthcare-dataset-stroke-data.csv",
                 "Clustering":"ObesityDataSet_raw_and_data_sinthetic.csv"}[track])
        return
    suggested = TRACKS[track]
    choices = list(df.columns)
    if track == "Clustering":
        choices = ["(No reference label)"] + choices
        initial = choices.index(suggested) if suggested in choices else 0
        chosen = st.sidebar.selectbox("Reference label (excluded from training)", choices,
                                      index=initial, key=f"target_{track}")
        target = None if chosen == "(No reference label)" else chosen
    else:
        initial = choices.index(suggested) if suggested in choices else len(choices)-1
        target = st.sidebar.selectbox("Target column", choices, index=initial, key=f"target_{track}")
    if page == "Dataset & EDA":
        explorer(df, track, target, origin)
        return
    if page == "Preprocessing":
        preprocessing_ui(df, track, target)
        return
    if page == "Train & Compare":
        training_ui(df, track, target, key, origin)
        return
    run = st.session_state.get(f"run_{track}")
    sig = st.session_state.get(f"sig_{track}")
    if run is None or sig is None or sig[1] != key or sig[2] != target:
        st.title(page)
        st.info("Open **Train & Compare**, select your experiment settings, and train the models first.")
        return
    if page == "Evaluation":
        visual_clustering(run) if track == "Clustering" else visual_supervised(run, track)
    elif page == "Prediction":
        predict_ui(run, track)
    else:
        export_ui(run, track)


if __name__ == "__main__":
    main()
