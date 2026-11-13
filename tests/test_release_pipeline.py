from pathlib import Path

import pytest
import yaml

from main import load_config
from postprocess import analyze_saved_case
from preflight import validate_project_config
from run_all import main as run_scan


ROOT = Path(__file__).resolve().parents[1]


def _test_config(tmp_path):
    config = load_config(ROOT / "config_test.yaml")
    config["wave"]["m_modes"] = [0]
    config["output"]["results_dir"] = str(tmp_path / "results")
    config["output"]["save_scan_fields"] = True
    target = tmp_path / "test_config.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    return target


def test_release_default_grid_and_direct_solver():
    cfg = load_config(ROOT / "config.yaml")
    report = validate_project_config(cfg)
    assert report.unknowns == 240_000
    assert report.method == "direct"
    assert report.run_count == 20


def test_preflight_rejects_antenna_overlap():
    cfg = load_config(ROOT / "config.yaml")
    cfg["antenna"]["radius_m"] = 0.034
    with pytest.raises(ValueError, match="overlaps the PML"):
        validate_project_config(cfg)


def test_dry_run_does_not_create_results(tmp_path):
    path = _test_config(tmp_path)
    rows = run_scan(str(path), dry_run=True)
    assert rows == []
    assert not (tmp_path / "results").exists()


def test_scan_checkpoint_resume_and_postprocess(tmp_path, monkeypatch):
    path = _test_config(tmp_path)
    rows = run_scan(str(path), plots=False)
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    case = tmp_path / "results" / "f9.8MHz_m0"
    for filename in ("fields.npz", "case_config.yaml", "selected_mode.npz", "result.json"):
        assert (case / filename).is_file()
    assert (tmp_path / "results" / "dispersion_scan.csv").is_file()
    report = analyze_saved_case(case)
    assert report["m_mode"] == 0
    assert (case / "wavefit_postprocessed.json").is_file()

    def forbidden(*args, **kwargs):
        raise AssertionError("A resumed case must not construct another solver")

    monkeypatch.setattr("run_all.HeliconSolver", forbidden)
    reused = run_scan(str(path), plots=False, resume=True)
    assert len(reused) == 1
    assert reused[0]["status"] == "ok"
    assert reused[0]["relative_linear_residual"] == rows[0]["relative_linear_residual"]


def test_scan_failure_recorded_before_exit(tmp_path, monkeypatch):
    path = _test_config(tmp_path)

    def failed(*args, **kwargs):
        raise RuntimeError("diagnostic test failure")

    monkeypatch.setattr("run_all.HeliconSolver", failed)
    rows = run_scan(str(path), plots=False, fail_fast=True)
    assert rows[0]["status"].startswith("error:")
    root = tmp_path / "results"
    assert (root / "dispersion_scan.csv").is_file()
    assert (root / "f9.8MHz_m0" / "failure.json").is_file()
