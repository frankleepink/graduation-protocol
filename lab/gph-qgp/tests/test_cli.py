"""Smoke tests for the CLI: every experiment runs end-to-end."""

import os
import subprocess
import sys


def _run_cli(experiment, out_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    result = subprocess.run(
        [sys.executable, "-m", "gph_qgp", "run",
         "--experiment", experiment, "--out", out_dir],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_cli_dispersion(tmp_path):
    out = _run_cli("dispersion", str(tmp_path))
    assert (tmp_path / "dispersion.csv").exists()
    assert (tmp_path / "dispersion.png").exists()
    assert "eta" in out


def test_cli_foam(tmp_path):
    out = _run_cli("foam", str(tmp_path))
    assert (tmp_path / "foam.csv").exists()
    assert (tmp_path / "foam.png").exists()
    assert "holographic" in out


def test_cli_collapse(tmp_path):
    out = _run_cli("collapse", str(tmp_path))
    assert (tmp_path / "collapse_dp.csv").exists()
    assert (tmp_path / "collapse_grw.json").exists()
    assert (tmp_path / "collapse.png").exists()
    assert "NOT simulated" in out


def test_cli_pilot(tmp_path):
    out = _run_cli("pilot", str(tmp_path))
    assert (tmp_path / "observers.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "pilot_speckle.png").exists()
    assert "Krippendorff" in out
