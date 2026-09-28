"""Make a consistent SQLite snapshot plus all local model artifacts and data."""
from datetime import datetime
from pathlib import Path
import sqlite3
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
backup_dir = ROOT / "backups"
backup_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
archive = backup_dir / f"lab2_mlflow_{timestamp}.tar.gz"
with tempfile.TemporaryDirectory(prefix="lab2_backup_") as temp:
    snapshot = Path(temp) / "mlruns.db"
    with sqlite3.connect(f"file:{ROOT / 'mlflow' / 'mlruns.db'}?mode=ro", uri=True) as source:
        with sqlite3.connect(snapshot) as destination:
            source.backup(destination)
            assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with tarfile.open(archive, "w:gz") as output:
        output.add(snapshot, arcname="mlflow/mlruns.db")
        output.add(ROOT / "mlflow" / "start_mlflow.sh", arcname="mlflow/start_mlflow.sh")
        output.add(ROOT / "mlflow" / "mlartifacts", arcname="mlflow/mlartifacts")
        output.add(ROOT / "data" / "clean_dataset.pkl", arcname="data/clean_dataset.pkl")
        output.add(ROOT / "data" / "heart_disease.csv", arcname="data/heart_disease.csv")
with tarfile.open(archive) as verification:
    names = verification.getnames()
    assert "mlflow/mlruns.db" in names
    assert any(name.endswith("model.pkl") for name in names)
print(archive)
print(f"Verified {len(names)} entries; {archive.stat().st_size / 1024 / 1024:.2f} MiB")
