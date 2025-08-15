# =========================================================
# AUTO ML PIPELINE (EDA -> CLEAN -> TRAIN -> EVAL -> SAVE)
# + SHAP Explainability, Feature Importances, MLflow Logging
# =========================================================
import os, json, warnings, math, uuid
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             f1_score, accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

from joblib import dump

# Optional: MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "imbalance": True,               # classification only
    "smote_k_neighbors": 5,
    "do_variance_threshold": True,
    "scaling": True,
    "encoding": "onehot",
    "do_outlier_cap": True,
    "n_splits_cv": 5,
    "n_iter_search": 30,
    "generate_eda_html": True,
    "html_report_name": "EDA_Report.html",
    "save_dir": "artifacts",
    # Explainability
    "shap_sample_rows": 2000,        # sample for SHAP (speed)
    "top_k_importance": 30           # plots top-K features
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ---------- Utils ----------
def quick_dtype_buckets(df, target_col):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target_col in num_cols: num_cols.remove(target_col)
    if target_col in cat_cols: cat_cols.remove(target_col)
    return num_cols, cat_cols

def is_classification(y):
    return (pd.Series(y).dtype.kind in ("O","b")) or (pd.Series(y).nunique() <= 20)

def iqr_cap(df, cols):
    for col in cols:
        if df[col].notna().sum() == 0: continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lo, hi)
    return df

def generate_eda_report(df, save_as="EDA_Report.html"):
    path = os.path.join(CONFIG["save_dir"], save_as)
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title="Automated EDA Report", explorative=True)
        profile.to_file(path)
        print(f"📄 EDA HTML saved -> {path}")
    except Exception as e:
        print(f"ℹ️ ydata-profiling not available or failed ({e}). Generating light EDA HTML...")
        html = []
        html.append("<h1>Light EDA Report</h1>")
        html.append(f"<p>Shape: {df.shape}</p>")
        html.append(df.dtypes.to_frame("dtype").to_html())
        miss = (df.isna().mean()*100).sort_values(ascending=False).to_frame("missing_%")
        html.append("<h2>Missing (%)</h2>")
        html.append(miss.to_html())
        html.append("<h2>Describe (numeric)</h2>")
        html.append(df.describe(include="number").to_html())
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"📄 Light EDA HTML saved -> {path}")

def build_preprocessor(df, target_col):
    num_cols, cat_cols = quick_dtype_buckets(df, target_col)
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    steps_num = [("imputer", num_imputer)]
    if CONFIG["scaling"]: steps_num.append(("scaler", StandardScaler()))
    steps_cat = [("imputer", cat_imputer)]
    if CONFIG["encoding"] == "onehot":
        steps_cat.append(("encoder", OneHotEncoder(handle_unknown="ignore")))

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps_num), num_cols),
                      ("cat", Pipeline(steps_cat), cat_cols)],
        remainder="drop"
    )

    selectors = []
    if CONFIG["do_variance_threshold"]:
        selectors.append(("varth", VarianceThreshold(0.0)))
    return preprocessor, selectors, num_cols, cat_cols

def get_model_search_spaces(task):
    if task == "classification":
        models = {
            "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                       {"model__C": np.logspace(-3, 2, 20), "model__penalty": ["l2"], "model__solver": ["lbfgs","liblinear"]}),
            "rf": (RandomForestClassifier(class_weight="balanced", random_state=CONFIG["random_state"]),
                   {"model__n_estimators":[100,200,400,600], "model__max_depth":[None,6,10,16,24],
                    "model__min_samples_split":[2,5,10], "model__min_samples_leaf":[1,2,4], "model__max_features":["sqrt","log2",None]}),
            "hgb": (HistGradientBoostingClassifier(random_state=CONFIG["random_state"]),
                    {"model__max_depth":[None,3,5,7], "model__learning_rate":np.logspace(-3,-0.3,15),
                     "model__max_leaf_nodes":[None,15,31,63], "model__min_samples_leaf":[10,20,30,50]})
        }
        scorer = "f1_macro"
    else:
        models = {
            "linreg": (LinearRegression(), {}),
            "rf": (RandomForestRegressor(random_state=CONFIG["random_state"]),
                   {"model__n_estimators":[100,200,400,600], "model__max_depth":[None,6,10,16,24],
                    "model__min_samples_split":[2,5,10], "model__min_samples_leaf":[1,2,4],
                    "model__max_features":["sqrt","log2",0.8,1.0]}),
            "hgb": (HistGradientBoostingRegressor(random_state=CONFIG["random_state"]),
                    {"model__max_depth":[None,3,5,7], "model__learning_rate":np.logspace(-3,-0.3,15),
                     "model__max_leaf_nodes":[None,15,31,63], "model__min_samples_leaf":[10,20,30,50]})
        }
        scorer = "neg_root_mean_squared_error"
    return models, scorer

def threshold_tune(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2*precision[:-1]*recall[:-1] / (precision[:-1]+recall[:-1] + 1e-12)
    i = int(np.argmax(f1s))
    return float(thresholds[i]), float(f1s[i])

def get_feature_names(preprocessor, num_cols, cat_cols, X_sample):
    """
    Reconstruct transformed feature names after ColumnTransformer.
    Works for num (after scaler) + cat (after OHE).
    """
    feature_names = []
    # numeric pipeline — names are same as num_cols
    feature_names.extend(num_cols)
    # categorical OHE names
    try:
        ohe = None
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                # unpack pipeline -> encoder
                try:
                    ohe = trans.named_steps.get("encoder", None)
                except Exception:
                    ohe = None
                if ohe is not None:
                    cats = ohe.get_feature_names_out(cols)
                    feature_names.extend(cats.tolist())
                else:
                    feature_names.extend(list(cols))
    except Exception:
        # fallback length match later
        pass

    # Fallback length fix if mismatch:
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            pass
    return feature_names

def safe_plot_bar(values, names, title, outpath, top_k=30):
    idx = np.argsort(values)[::-1][:min(top_k, len(values))]
    vals = np.array(values)[idx]
    nams = np.array(names)[idx]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), nams, rotation=75, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def log_mlflow_start(run_name):
    if not MLFLOW_AVAILABLE: 
        print("ℹ️ MLflow not installed — skipping MLflow logging.")
        return None
    try:
        mlflow.set_experiment("AutoML_Pipeline")
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: v for k, v in CONFIG.items() if not isinstance(v, (list, dict))})
        return run
    except Exception as e:
        print(f"ℹ️ MLflow issue: {e} — proceeding without MLflow.")
        return None

def log_mlflow_metrics(metrics):
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            pass

def log_mlflow_artifact(path):
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(path)
        except Exception:
            pass

# --------------- MAIN ---------------
def run_end_to_end(dataset_path, target_col):
    run_id = str(uuid.uuid4())[:8]
    mlflow_run = log_mlflow_start(run_name=f"run_{run_id}")

    print("🚀 Loading data...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded: {df.shape}")

    # Basic cleaning
    df.replace([-1, 999, "?", "NA", "na", "null", "None", ""], np.nan, inplace=True)
    dups = df.duplicated().sum()
    if dups:
        df = df.drop_duplicates()
        print(f"🧹 Dropped duplicates: {dups}")

    # Task detection
    task = "classification" if is_classification(df[target_col]) else "regression"
    print(f"🧭 Task: {task}")

    # Outlier cap
    if CONFIG["do_outlier_cap"]:
        num_cols_cap, _ = quick_dtype_buckets(df, target_col)
        df = iqr_cap(df, num_cols_cap)

    # EDA
    if CONFIG["generate_eda_html"]:
        generate_eda_report(df, save_as=CONFIG["html_report_name"])
        log_mlflow_artifact(os.path.join(CONFIG["save_dir"], CONFIG["html_report_name"]))

    # Target encode if classification
    label_encoder = None
    y_raw = df[target_col]
    y = y_raw
    if task == "classification" and y_raw.dtype.kind in ("O","b"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        print("🏷️ LabelEncoded target")

    X = df.drop(columns=[target_col])

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=stratify
    )
    print(f"📦 Split: Train {X_train.shape}, Test {X_test.shape}")

    preprocessor, selectors, num_cols, cat_cols = build_preprocessor(df, target_col)
    base_steps = [("preprocess", preprocessor)]
    for nm, sel in selectors:
        base_steps.append((nm, sel))
    if CONFIG["imbalance"] and task == "classification":
        base_steps.append(("smote", SMOTE(random_state=CONFIG["random_state"], k_neighbors=CONFIG["smote_k_neighbors"])))

    models, scorer = get_model_search_spaces(task)
    cv = StratifiedKFold(n_splits=CONFIG["n_splits_cv"], shuffle=True, random_state=CONFIG["random_state"]) \
        if task == "classification" else KFold(n_splits=CONFIG["n_splits_cv"], shuffle=True, random_state=CONFIG["random_state"])

    # Hyperparam search
    print("🔎 Hyperparameter search...")
    best_name, best_cv, best_obj = None, -np.inf, None
    for name, (estimator, space) in models.items():
        pipe_cls = ImbPipeline if (CONFIG["imbalance"] and task == "classification") else Pipeline
        pipe = pipe_cls(base_steps + [("model", estimator)])
        if space:
            search = RandomizedSearchCV(pipe, space, n_iter=CONFIG["n_iter_search"], scoring=scorer,
                                        n_jobs=-1, cv=cv, random_state=CONFIG["random_state"], verbose=0)
            search.fit(X_train, y_train)
            score_val = search.best_score_
            obj = search
        else:
            score_val = np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1))
            obj = pipe.fit(X_train, y_train)
        print(f"   • {name}: CV={score_val:.4f}")
        if score_val > best_cv:
            best_cv, best_name, best_obj = score_val, name, obj

    print(f"🏆 Best model: {best_name} | CV={best_cv:.4f}")
    best_estimator = best_obj.best_estimator_ if hasattr(best_obj, "best_estimator_") else best_obj
    best_estimator.fit(X_train, y_train)

    # ===== Evaluation =====
    metrics_out = {}
    if task == "classification":
        y_pred = best_estimator.predict(X_test)
        metrics_out["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics_out["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))

        # probability/score
        y_scores = None
        if hasattr(best_estimator, "predict_proba"):
            proba = best_estimator.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_scores = proba[:, 1]
        elif hasattr(best_estimator, "decision_function"):
            y_scores = best_estimator.decision_function(X_test)

        if y_scores is not None:
            try:
                metrics_out["roc_auc"] = float(roc_auc_score(y_test, y_scores))
                metrics_out["pr_auc"] = float(average_precision_score(y_test, y_scores))
                th, f1b = threshold_tune(y_test, y_scores)
                metrics_out["best_threshold_f1"] = th
                metrics_out["best_f1_at_th"] = f1b
            except Exception:
                pass

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    else:
        y_pred = best_estimator.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        adj_r2 = 1 - ((1 - r2) * (n - 1) / max(n - p - 1, 1))
        metrics_out.update({"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "Adj_R2": float(adj_r2)})

    print("📈 Metrics:", json.dumps(metrics_out, indent=2))
    if mlflow_run is not None: log_mlflow_metrics(metrics_out)

    # ===== Save artifacts =====
    model_path = os.path.join(CONFIG["save_dir"], f"best_model_{best_name}.joblib")
    dump(best_estimator, model_path)
    print(f"💾 Model saved -> {model_path}")
    if mlflow_run is not None: log_mlflow_artifact(model_path)

    if isinstance(label_encoder, LabelEncoder):
        le_path = os.path.join(CONFIG["save_dir"], "label_encoder.joblib")
        dump(label_encoder, le_path)
        print(f"💾 Label encoder saved -> {le_path}")
        if mlflow_run is not None: log_mlflow_artifact(le_path)

    manifest = {"task": task, "best_model": best_name, "cv_best_score": float(best_cv), "config": CONFIG}
    with open(os.path.join(CONFIG["save_dir"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("🧾 manifest.json saved")
    if mlflow_run is not None:
        mf = os.path.join(CONFIG["save_dir"], "manifest.json")
        log_mlflow_artifact(mf)

    # ===== Feature Names after preprocess =====
    # Take a small fit on train to extract fitted preprocessor
    fitted_pre = best_estimator.named_steps.get("preprocess", None)
    if fitted_pre is None:
        # If inside ImbPipeline, preprocess is still accessible
        for step_name, step in best_estimator.named_steps.items():
            if isinstance(step, ColumnTransformer):
                fitted_pre = step
                break

    # Prepare a small transformed sample for explainability
    X_train_sample = X_train.sample(min(len(X_train), CONFIG["shap_sample_rows"]), random_state=CONFIG["random_state"]) \
                     if hasattr(X_train, "sample") else X_train
    # Transform to matrix
    Xt_sample = fitted_pre.transform(X_train_sample)
    # Get feature names
    num_cols, cat_cols = quick_dtype_buckets(df, target_col)
    feat_names = get_feature_names(fitted_pre, num_cols, cat_cols, X_train_sample)
    if len(feat_names) != getattr(Xt_sample, "shape", [0,0])[1]:
        # fallback generic names
        feat_names = [f"f_{i}" for i in range(Xt_sample.shape[1])]

    # ===== Feature Importances =====
    print("🧮 Computing feature importances...")
    # Try model native importances
    model = best_estimator.named_steps.get("model", best_estimator)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef if coef.ndim == 1 else coef[0])
    else:
        # permutation importance as fallback
        try:
            pi = permutation_importance(best_estimator, X_test, y_test, n_repeats=5, random_state=CONFIG["random_state"], n_jobs=-1)
            importances = pi.importances_mean
        except Exception:
            pass

    if importances is not None:
        imp_path = os.path.join(CONFIG["save_dir"], "feature_importance_topK.png")
        safe_plot_bar(importances, feat_names, "Feature Importance (Top-K)", imp_path, top_k=CONFIG["top_k_importance"])
        print(f"🖼️ Feature importance plot saved -> {imp_path}")
        if mlflow_run is not None: log_mlflow_artifact(imp_path)

    # ===== SHAP Explainability =====
    print("🔍 Running SHAP explainability (best-effort)...")
    try:
        import shap
        # Get model inside pipeline
        model = best_estimator.named_steps.get("model", best_estimator)
        # SHAP background (use small sample)
        bg_idx = np.random.RandomState(CONFIG["random_state"]).choice(Xt_sample.shape[0], min(200, Xt_sample.shape[0]), replace=False)
        background = Xt_sample[bg_idx]

        # Choose explainer
        explainer = None
        if model.__class__.__name__.endswith(("Classifier","Regressor")) and hasattr(model, "predict"):
            try:
                explainer = shap.Explainer(model, background)
            except Exception:
                explainer = shap.KernelExplainer(model.predict if hasattr(model, "predict") else model.predict_proba, background)

        if explainer is not None:
            # Pick a small subset for plotting
            test_sample = X_test.sample(min(len(X_test), 500), random_state=CONFIG["random_state"]) \
                          if hasattr(X_test, "sample") else X_test
            Xt_test = fitted_pre.transform(test_sample)
            shap_values = explainer(Xt_test)

            # Summary plot (bar)
            shap_bar = os.path.join(CONFIG["save_dir"], "shap_summary_bar.png")
            plt.figure()
            shap.plots.bar(shap_values, show=False, max_display=CONFIG["top_k_importance"])
            plt.tight_layout(); plt.savefig(shap_bar, dpi=150); plt.close()
            print(f"🖼️ SHAP bar saved -> {shap_bar}")
            if mlflow_run is not None: log_mlflow_artifact(shap_bar)

            # Beeswarm plot
            shap_bee = os.path.join(CONFIG["save_dir"], "shap_beeswarm.png")
            plt.figure()
            shap.plots.beeswarm(shap_values, show=False, max_display=CONFIG["top_k_importance"])
            plt.tight_layout(); plt.savefig(shap_bee, dpi=150); plt.close()
            print(f"🖼️ SHAP beeswarm saved -> {shap_bee}")
            if mlflow_run is not None: log_mlflow_artifact(shap_bee)
        else:
            print("ℹ️ SHAP explainer could not be created, skipping plots.")
    except Exception as e:
        print(f"ℹ️ SHAP failed/skipped: {e}")

    if mlflow_run is not None:
        try:
            mlflow.end_run()
        except Exception:
            pass

    print("\n✅ Done: EDA, training, evaluation, explainability & logging complete.")
    return best_estimator

# ================= HOW TO RUN =================
# best_model = run_end_to_end("your_dataset.csv", target_col="label")
