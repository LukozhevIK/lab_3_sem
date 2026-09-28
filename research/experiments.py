"""Reusable code for lab 2; fitted preprocessing stays inside each Pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
SEED = 42
EXPERIMENT_NAME = "Heart_Disease_Lab2"
MODEL_NAME = "HeartDiseaseClassifier"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "age_group"]
POLY_FEATURES = ["age", "thalach", "oldpeak"]
BIN_FEATURES = ["age", "thalach", "oldpeak"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Fixed BEFORE looking at lab-2 scores, based on lab-1 EDA.
# 13 of 26 engineered columns = 50%, within the required 20–70% range.
EXPERT_FEATURES = [
    "numeric__age", "numeric__thalach", "numeric__oldpeak", "numeric__ca",
    "categorical__sex", "categorical__cp", "categorical__exang", "categorical__slope",
    "categorical__thal", "polynomial__age thalach", "polynomial__thalach oldpeak",
    "bins__age", "bins__thalach",
]


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data():
    path = ROOT / "data" / "clean_dataset.pkl"
    data = pd.read_pickle(path)
    assert data.shape == (303, 15), "Unexpected lab-1 dataset"
    assert not data.isna().any().any()
    X = data.drop(columns="target").copy()
    # Stable serving schema: JSON numbers and strings, rather than pandas categories.
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype("float64")
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)
    y = data["target"].astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return X, y, X_train, X_test, y_train, y_test, digest


def configure_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient(), experiment.experiment_id


def make_transform(engineered=False):
    steps = [
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", TargetEncoder(target_type="binary", cv=5, random_state=SEED), CATEGORICAL_FEATURES),
    ]
    if engineered:
        steps += [
            ("polynomial", Pipeline([
                ("expand", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
            ]), POLY_FEATURES),
            ("bins", Pipeline([
                ("discretize", KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile", subsample=None)),
                ("scale", StandardScaler()),
            ]), BIN_FEATURES),
        ]
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=True)


def make_pipeline(kind="baseline", params=None, selected_indices=None):
    steps = [("transform", make_transform(engineered=kind != "baseline"))]
    if kind == "expert_selected":
        if selected_indices is None:
            raise ValueError("Expert-selection indices must be provided")
        steps.append(("selection", ColumnTransformer(
            [("selected", "passthrough", list(selected_indices))], remainder="drop",
        )))
    defaults = {"n_estimators": 100, "random_state": SEED, "n_jobs": 2}
    defaults.update(params or {})
    steps.append(("classification", RandomForestClassifier(**defaults)))
    return Pipeline(steps)


def feature_names(pipeline):
    names = pipeline.named_steps["transform"].get_feature_names_out().tolist()
    if "selection" in pipeline.named_steps:
        indices = pipeline.named_steps["selection"].transformers_[0][2]
        names = [names[index] for index in indices]
    return names


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


def inference_requirements():
    import cloudpickle
    import scipy
    return [
        "mlflow==2.16.0", "numpy==1.26.4", "pandas==2.3.3", "scikit-learn==1.5.2",
        f"scipy=={scipy.__version__}", f"cloudpickle=={cloudpickle.__version__}",
    ]


def log_model_artifacts(model, X_example, digest, full_data=False):
    names = feature_names(model)
    mlflow.log_dict({"columns": names}, "features/transformed_columns.json")
    mlflow.log_dict({"columns": X_example.columns.tolist()}, "features/input_columns.json")
    if "selection" in model.named_steps:
        indices = [int(i) for i in model.named_steps["selection"].transformers_[0][2]]
        mlflow.log_dict({"indices": indices}, "features/selected_indices.json")
        mlflow.log_dict({"columns": names}, "features/selected_columns.json")
    mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="environment")
    mlflow.log_artifact(str(Path(__file__)), artifact_path="source")
    mlflow.set_tags({
        "dataset_sha256": digest, "selection_metric": "5-fold training CV F1",
        "data_scope": "all_303_rows" if full_data else "train_227_rows",
    })
    return mlflow.sklearn.log_model(
        model, artifact_path="model", input_example=X_example,
        signature=infer_signature(X_example, model.predict(X_example)),
        pip_requirements=inference_requirements(),
    )


def train_and_log(kind, X_train, y_train, X_test, y_test, digest,
                  selected_indices=None, params=None, run_name=None, cv_scores=None,
                  register=False):
    template = make_pipeline(kind, params=params, selected_indices=selected_indices)
    start = time.monotonic()
    if cv_scores is None:
        cv_scores = cross_val_score(template, X_train, y_train, cv=CV, scoring="f1", n_jobs=1)
    fitted = clone(template).fit(X_train, y_train)
    metrics = evaluate(fitted, X_test, y_test)
    metrics.update(cv_f1_mean=float(np.mean(cv_scores)), cv_f1_std=float(np.std(cv_scores)))
    with mlflow.start_run(run_name=run_name or kind) as run:
        run_id = run.info.run_id
        rf_params = fitted.named_steps["classification"].get_params()
        mlflow.log_params({
            "feature_set": kind, "train_rows": len(X_train), "test_rows": len(X_test),
            "random_state": SEED, "cv_folds": 5, "transformed_features": len(feature_names(fitted)),
            **{key: rf_params[key] for key in ["n_estimators", "max_depth", "max_features", "min_samples_leaf"]},
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_seconds", time.monotonic() - start)
        model_info = log_model_artifacts(fitted, X_train.head(5), digest)
    version = None
    if register:
        version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
    result = {"name": run_name or kind, "kind": kind, "run_id": run_id, **metrics}
    return result, fitted, version


def select_expert_columns(X_train, y_train):
    transform = make_transform(engineered=True)
    # Diagnostic only: each actual model/CV fold fits a fresh transform.
    X_train_fe_sklearn = X_train.copy()
    expanded = transform.fit_transform(X_train_fe_sklearn, y_train)
    names = transform.get_feature_names_out().tolist()
    X_train_fe_sklearn = pd.DataFrame(expanded, index=X_train.index, columns=names)
    indices = [names.index(name) for name in EXPERT_FEATURES]
    assert 0.2 <= len(indices) / len(names) <= 0.7
    save_json(RESEARCH / "engineered_columns.json", {"columns": names})
    save_json(RESEARCH / "selected_columns.json", {"columns": EXPERT_FEATURES})
    save_json(RESEARCH / "selected_indices.json", {"indices": indices})
    return X_train_fe_sklearn, names, indices


def tune_best(best_kind, X_train, y_train, selected_indices, n_trials=12):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    scores_by_trial = {}
    with mlflow.start_run(run_name="optuna_search") as parent:
        parent_id = parent.info.run_id
        mlflow.log_params({"feature_set": best_kind, "objective": "cv_f1", "direction": "maximize", "n_trials": n_trials})

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            }
            pipeline = make_pipeline(best_kind, params=params, selected_indices=selected_indices)
            scores = cross_val_score(pipeline, X_train, y_train, cv=CV, scoring="f1", n_jobs=1)
            scores_by_trial[trial.number] = scores
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number:02d}", nested=True):
                mlflow.log_params({**params, "feature_set": best_kind, "trial_number": trial.number})
                mlflow.log_metrics({"cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())})
            return float(scores.mean())

        study.optimize(objective, n_trials=n_trials, timeout=600)
        completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        assert completed >= 10, "At least ten completed trials are required"
        study.trials_dataframe().to_json(RESEARCH / "optuna_trials.json", orient="records", indent=2)
        mlflow.log_artifact(str(RESEARCH / "optuna_trials.json"))
        mlflow.log_metrics({"best_cv_f1": study.best_value, "completed_trials": completed})
        mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
    return study, scores_by_trial[study.best_trial.number], parent_id


def comparison_plot(results):
    table = pd.DataFrame(results)
    save_json(RESEARCH / "results.json", results)
    fig, ax = plt.subplots(figsize=(10, 5))
    table.set_index("name")[["precision", "recall", "f1", "roc_auc"]].plot.bar(ax=ax, rot=15)
    ax.set(title="Сравнение моделей на тестовых 25%", ylim=(0, 1.05), ylabel="Значение метрики", xlabel="")
    ax.legend(loc="lower left", ncol=4)
    fig.tight_layout()
    fig.savefig(RESEARCH / "test_metrics.png", dpi=160)
    plt.show()
    return table


def train_production(best_result, best_fitted, X, y, digest, client):
    production = clone(best_fitted).fit(X, y)
    with mlflow.start_run(run_name="production_full_data") as run:
        production_run_id = run.info.run_id
        mlflow.log_params({
            "feature_set": best_result["kind"], "train_rows": len(X), "test_rows": 0,
            "source_run_id": best_result["run_id"],
            **{key: production.named_steps["classification"].get_params()[key]
               for key in ["n_estimators", "max_depth", "max_features", "min_samples_leaf", "random_state"]},
        })
        mlflow.set_tag("role", "Production")
        model_info = log_model_artifacts(production, X.head(5), digest, full_data=True)
    version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
    client.set_model_version_tag(MODEL_NAME, version, "Production", "true")
    client.set_model_version_tag(MODEL_NAME, version, "role", "Production")
    client.set_registered_model_alias(MODEL_NAME, "Production", version)
    model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_info.model_uri))
    shutil.copy2(model_dir / "MLmodel", RESEARCH / "MLmodel")
    restored = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Production")
    np.testing.assert_array_equal(production.predict(X.head(10)), restored.predict(X.head(10)))
    np.testing.assert_allclose(production.predict_proba(X.head(10)), restored.predict_proba(X.head(10)))
    metadata = {
        "experiment_name": EXPERIMENT_NAME, "model_name": MODEL_NAME,
        "production_run_id": production_run_id, "production_version": str(version),
        "production_uri": f"models:/{MODEL_NAME}@Production",
        "source_run_id": best_result["run_id"], "source_name": best_result["name"],
        "feature_set": best_result["kind"], "train_rows": len(X),
        "cv_f1_mean": best_result["cv_f1_mean"],
        "test_metrics_of_source_model": {key: best_result[key] for key in ["precision", "recall", "f1", "roc_auc"]},
        "parameters": {key: production.named_steps["classification"].get_params()[key]
                       for key in ["n_estimators", "max_depth", "max_features", "min_samples_leaf", "random_state"]},
        "input_columns": X.columns.tolist(), "transformed_columns": feature_names(production),
        "dataset_sha256": digest,
    }
    save_json(RESEARCH / "production.json", metadata)
    return production, metadata
