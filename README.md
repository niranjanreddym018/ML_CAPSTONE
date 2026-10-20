# ML Studio · 23CSE301 Machine Learning Capstone

An interactive, browser-based ML workbench built with **Python + Streamlit**, combining the three uploaded capstone notebooks into one application. The original `.ipynb` source files are preserved in `notebooks/`.

## What is included

| Track | Original CSV expected | Target | Algorithms |
|---|---|---|---|
| Regression | `Loan.csv` | `RiskScore` | Linear, Ridge, Lasso, ElasticNet, Polynomial (8 top features + degree 2), Decision Tree, Random Forest, XGBoost, SVR, KNN |
| Classification | `healthcare-dataset-stroke-data.csv` | `stroke` | Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM, Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP |
| Clustering | `ObesityDataSet_raw_and_data_sinthetic.csv` | `NObeyesdad` (reference only) | K-Means, Agglomerative Hierarchical |

**Review 2 coverage:** Interactive GUI, upload/EDA, missing-value checks and duplicate counts, preprocessing inspection and transformed previews, side-by-side model results, required charts, train/test evaluation, cluster evaluation and plots, prediction screen, results/model downloads, and deployment-ready source. Optional 3-fold training-only tuning is available for supervised models.

Clustering follows the rubric's **two required algorithms**, not ten clustering algorithms. The ten-model requirement is implemented for *each supervised track*.

## Run locally

Python **3.11 or 3.12** recommended.

```bash
# Open the ML_Studio project folder
cd ML_Studio

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address, typically `http://localhost:8501`. Open it in a browser. You can then upload your *real* CSV datasets with the sidebar uploader.

### How to use

1. Choose **Regression**, **Classification**, or **Clustering** from the sidebar.
2. Upload the corresponding original CSV. The original dataset files are **not** bundled because only the three notebooks were attached in this chat.
3. In **Dataset & EDA**, inspect records, nulls, duplicates, distributions, scatter plots and correlation heatmaps.
4. In **Preprocessing**, inspect derived, excluded, imputed, scaled and encoded features via a separate preview.
5. In **Train & Compare**, choose *Fast (2,000 rows)*, *Standard (5,000)*, or *Full dataset* and click **Train and evaluate**. The number of used rows is shown; final review metrics should use the full data if your machine/cloud allocation permits.
6. Compare all models. If needed, select two supervised algorithms under **Optional hyperparameter tuning**; the best hyperparameters are selected with 3-fold CV on training data only.
7. In **Evaluation**, show required charts and model diagnostics.
8. In **Prediction**, enter feature values or upload a batch CSV with the same columns as the training inputs.
9. In **Downloads**, export the metrics CSV, predictions CSV, run JSON, or a locally trained fitted model.

**Quick demo:** Turn on **Try synthetic demo data** in the sidebar. These small datasets are marked *synthetic* everywhere and are for checking that the interface works. **Do not report these scores as real capstone results.**

## Preprocessing / technical notes

- **Supervised:** duplicate rows are removed, target-null records discarded, and the same 80:20 train/test split is used across all ten models for a track. Numeric values are median-imputed, IQR-clipped when applicable and scaled; categorical values are imputed and one-hot encoded. Transformers are fitted on the *training split only*, inside a scikit-learn pipeline. The saved fitted pipeline handles new prediction records without re-fitting.
- **Regression-specific:** prevents label leakage by dropping `LoanApproved`, generates `LoanToIncomeRatio` from loan amount and annual income, extracts year/month from `ApplicationDate`, ordinal-encodes `EducationLevel`. For the notebook-aligned risk data, the regression target is `RiskScore`, which is **not** used as an input.
- **Stroke-specific:** excludes patient `id`, drops the one-off `gender='Other'` row if it exists, creates `age_glucose_risk`, uses stratified splits and balanced weights or class weights where supported. **Positive-class recall and ROC-AUC are more useful than accuracy for the original imbalanced stroke dataset.**
- **Classification evaluation:** accuracy, weighted precision, weighted recall, weighted F1, macro F1, positive-class recall/precision for binary classification, ROC-AUC, confusion matrix and ROC/precision-recall plots.
- **Regression evaluation:** R², MAE, MSE, RMSE, residual plot, actual-versus-predicted plot, and feature importance when supported.
- **Clustering:** drop duplicates, hold `NObeyesdad` outside the feature matrix, impute/scale numeric inputs and one-hot encode categories. Explore an elbow curve (inertia), k vs silhouette, four hierarchical linkages, a sampled dendrogram, PCA (2 components), optional t-SNE, cluster profiles, internal metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz), and optional post-hoc ARI/NMI/crosstabs against known obesity labels. No train/test split is used. Silhouette scores may use a reproducible 700-point sample for speed.
- **Predicting clusters:** K-Means supports new-record predictions; standard agglomerative clustering does not support `predict()`, so it is not presented as a predictor.
- **Generic uploaded datasets:** supervised targets can be changed via the sidebar. The notebook's derived features are activated only when their named inputs exist. Non-numeric regression targets and insufficient class counts are rejected with readable errors.
- **Real-time training:** your browser session trains the models when you click the button; **no results are hardcoded** from notebooks. The full 20,000-row regression experiment and XGBoost/SVM/neural networks can take time on Community Cloud. This tool is not a clinically validated stroke predictor nor a lending decision engine.

### Note on notebook equivalence

This application is an integrated **reimplementation** of the three tracks, not a literal cell-by-cell notebook runner. For responsiveness it uses several smaller estimator defaults, a single shared training pipeline, limited optional grids and sampled silhouette/dendrogram visualizations. Accordingly, its evaluation numbers may differ from the original notebooks even on the same CSV. It does not imply a reviewer will award a particular grade.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository, e.g. `ml-studio-capstone`.
2. Put the **contents** of this folder in the repo root (`app.py`, `requirements.txt`, `ml_studio/`, `sample_data/`, `notebooks/`, `.streamlit/`). Keep the original raw course datasets *out of Git* if they contain sensitive or non-redistributable records.
3. Push to GitHub:

   ```bash
   git init
   git add .
   git commit -m "Initial ML Studio Streamlit application"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ml-studio-capstone.git
   git push -u origin main
   ```

4. Visit **[Streamlit Community Cloud](https://share.streamlit.io/)** and sign in with your GitHub account.
5. Select **Create app**, your repo and branch (`main`), and main file path **`app.py`**. Set Python to 3.11 or 3.12 if the advanced settings offer it. No API keys are necessary.
6. Click **Deploy**. After the build finishes, Community Cloud assigns a **public shareable URL**. Test all three tracks with the included *synthetic* demo files, then upload actual datasets for review.

**Deployment isn't performed automatically by this ZIP**: pushing to your GitHub account and creating the Streamlit app requires your account access. This repository is prepared to deploy.

For the official process, see [Streamlit's Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## Project structure

```text
ML_Studio/
├── app.py                         # Streamlit pages, plots, predictions, downloads
├── ml_studio/
│   ├── __init__.py
│   ├── engine.py
│   └── preprocessing_view.py                   # ML preprocessing, models, metrics, CV, clustering
├── notebooks/
│   ├── credit_risk_regression.ipynb
│   ├── classification_stroke_final.ipynb
│   └── clustering_obesity_review2.ipynb
├── sample_data/
│   ├── demo_credit_risk.csv        # Synthetic only
│   ├── demo_stroke.csv             # Synthetic only
│   ├── demo_obesity.csv            # Synthetic only
│   └── generate_demo_data.py
├── tests/
│   └── test_engine.py
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Developer checks

```bash
pip install pytest
python -m pytest -q
python -m py_compile app.py ml_studio/engine.py
```

For reproducibility, random seed = **42**. Test results in the Streamlit UI always come from the currently uploaded file, selected target and training settings.
