import json
import os
import subprocess
import sys
from pathlib import Path


def test_train_ready_missing_environment_is_privacy_safe(tmp_path):
    report = tmp_path / "readiness.json"
    environment = os.environ.copy()
    environment.pop("RYNNWORLD4D_COLMAP_ROOT", None)
    environment.pop("RYNNWORLD4D_LATENT_ROOT", None)
    environment["RYNNWORLD4D_READINESS_REPORT"] = str(report)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_ready"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    rendered = json.dumps(payload)
    assert result.returncode == 1
    assert payload["ready"] is False
    assert "environment" in payload["failures"]
    assert str(tmp_path) not in rendered
    assert "/home/" not in rendered
