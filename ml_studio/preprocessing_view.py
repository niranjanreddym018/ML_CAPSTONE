"""Interactive, read-only inspector for notebook-aligned preprocessing."""
import pandas as pd

from ml_studio.engine import FeatureBuilder, get_preprocessor


def preprocessing_ui(df, track, target):
    import streamlit as st

    st.subheader("Preprocessing Inspector")
    st.caption("See exactly which features are created or excluded before model training.")
    if track == "Regression":
        st.write("Remove duplicates, exclude `LoanApproved` and other IDs, engineer loan/income ratio and date parts, ordinal-encode education; median-impute, clip numeric outliers, standardize and one-hot encode other categories.")
    elif track == "Classification":
        st.write("Remove duplicates, exclude patient `id`, derive `age_glucose_risk`, impute BMI/other missing values and scale/encode feature columns. Use stratified train/test splitting.")
    else:
        st.write("Remove duplicates; keep `NObeyesdad` separate as an interpretation-only label. Median-impute and scale numeric attributes and one-hot encode categorical attributes. No train/test split.")
    clean = df.drop_duplicates().copy()
    if track == "Classification" and target == "stroke" and "gender" in clean:
        clean = clean.loc[clean["gender"] != "Other"].copy()
    if track != "Clustering" and target:
        clean = clean.dropna(subset=[target])
    raw_features = clean.drop(columns=[target] if target else [], errors="ignore")
    a, b, c = st.columns(3)
    a.metric("Original rows", len(df))
    b.metric("Usable rows", len(clean))
    c.metric("Removed rows", len(df) - len(clean))
    engineered = FeatureBuilder(track).transform(raw_features)
    numeric = engineered.select_dtypes(include="number").columns.tolist()
    category = [name for name in engineered if name not in numeric]
    a, b, c = st.columns(3)
    a.metric("Numeric features", len(numeric))
    b.metric("Categorical features", len(category))
    c.metric("Engineered features", len(set(engineered) - set(raw_features)))
    st.write("**Generated:**", ", ".join(sorted(set(engineered) - set(raw_features))) or "none")
    st.write("**Excluded/consumed:**", ", ".join(sorted(set(raw_features) - set(engineered))) or "none")
    st.dataframe(engineered.head(15), hide_index=True, use_container_width=True)
    with st.expander("Preview scaled and one-hot encoded features"):
        st.caption("This preview is fitted on up to 500 rows for display only. Actual training uses a fresh preprocessing pipeline fitted on training rows only, preventing test-data leakage.")
        if st.button("Generate processed sample", key=f"sample_transform_{track}"):
            try:
                sample = raw_features.head(500)
                prep = get_preprocessor(sample, track)
                arr = prep.fit_transform(FeatureBuilder(track).transform(sample))
                result = pd.DataFrame(arr, columns=prep.get_feature_names_out())
                st.metric("Output features after encoding", result.shape[1])
                st.dataframe(result.head(20).round(3), hide_index=True, use_container_width=True)
                st.download_button("Download this processed preview", result.to_csv(index=False),
                                   file_name="processed_preview.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Preprocessing preview failed: {exc}")
    st.info("This screen does not change the CSV. Choose Train & Compare to fit actual models and their preprocessing pipelines.")
