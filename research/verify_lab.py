"""Read-only integration checks of completed mandatory lab-2 artifacts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import nbformat
import numpy as np
from research.experiments import configure_mlflow, load_data, feature_names


def main():
    client, experiment_id = configure_mlflow()
    directory = ROOT / "research"
    results = json.loads((directory / "results.json").read_text())
    metadata = json.loads((directory / "production.json").read_text())
    trials = json.loads((directory / "optuna_trials.json").read_text())
    assert len(trials) >= 10 and all(t["state"] == "COMPLETE" for t in trials)
    runs = client.search_runs([experiment_id], max_results=1000)
    assert all(r.info.status == "FINISHED" for r in runs)
    X, y, X_train, X_test, y_train, y_test, digest = load_data()
    assert len(X_train) == 227 and len(X_test) == 76
    assert set(X_train.index).isdisjoint(X_test.index)
    assert digest == metadata["dataset_sha256"]
    for result in results:
        run = client.get_run(result["run_id"])
        for metric in ["precision", "recall", "f1", "roc_auc", "cv_f1_mean"]:
            assert 0 <= result[metric] <= 1
            assert np.isclose(result[metric], run.data.metrics[metric])
        for folder in ["model", "features", "environment"]:
            assert client.list_artifacts(result["run_id"], folder)
    final = client.get_run(metadata["production_run_id"])
    assert final.data.metrics == {}, "Final full-data run must not claim test metrics"
    assert final.data.params["train_rows"] == "303"
    version = client.get_model_version_by_alias(metadata["model_name"], "Production")
    assert version.version == metadata["production_version"]
    assert version.tags["Production"] == "true"
    model = mlflow.sklearn.load_model(metadata["production_uri"])
    assert feature_names(model) == metadata["transformed_columns"]
    assert len(feature_names(model)) == 13
    assert model.predict(X).shape == (303,)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    notebook = nbformat.read(directory / "research.ipynb", as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    assert (directory / "MLmodel").read_text().find(final.info.run_id) >= 0
    print(f"OK: {len(runs)} finished runs; {len(trials)} trials; "
          f"Production v{version.version}; 303 predictions; executed notebook without errors")


if __name__ == "__main__":
    main()
