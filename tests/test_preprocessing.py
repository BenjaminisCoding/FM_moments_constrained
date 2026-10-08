"""Source integrity, output preservation, and optical quadrature checks."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cfm_project.preprocessing import common


def test_checked_download_and_corruption(tmp_path, monkeypatch):
    origin = tmp_path / "original.txt"
    origin.write_bytes(b"raw assay")
    config = tmp_path / "configs/preprocessing"
    config.mkdir(parents=True)
    expected = hashlib.sha256(origin.read_bytes()).hexdigest()
    (config / "sources.json").write_text(
        json.dumps(
            {
                "example": [
                    {
                        "filename": "assay.txt",
                        "url": origin.as_uri(),
                        "sha256": expected,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(common, "ROOT", tmp_path)
    raw = tmp_path / "raw"
    with pytest.raises(FileNotFoundError, match="--download"):
        common.sources("example", raw)
    common.sources("example", raw, download=True)
    assert (raw / "assay.txt").read_bytes() == origin.read_bytes()
    (raw / "assay.txt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        common.sources("example", raw, download=True)
    assert (raw / "assay.txt").read_bytes() == b"corrupted"
    (raw / "assay.txt").unlink()
    origin.write_bytes(b"changed upstream")
    with pytest.raises(ValueError, match="checksum mismatch"):
        common.sources("example", raw, download=True)
    assert list(raw.iterdir()) == []


def test_manifest_preserves_other_benchmarks(tmp_path):
    for name in ["semrau", "aerosol"]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "train.bin").write_bytes(name.encode())
        common.register(tmp_path, folder)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["files"]) == {"semrau/train.bin", "aerosol/train.bin"}
    for name, record in manifest["files"].items():
        assert record["sha256"] == common.digest(tmp_path / name)


def test_cli_refuses_overwrite(tmp_path):
    folder = tmp_path / "semrau"
    folder.mkdir()
    sentinel = folder / "train.npz"
    sentinel.write_bytes(b"frozen")
    script = Path(__file__).resolve().parents[1] / "scripts/preprocess_data.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "semrau",
            "--raw",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert sentinel.read_bytes() == b"frozen"


def test_histogram_quadrature_with_empty_bins():
    pytest.importorskip("xarray")
    from cfm_project.preprocessing.aerosol import (
        endpoint_quadrature,
        histogram_quantile,
    )

    w0, w1 = np.array([0.0, 0.3, 0.7]), np.array([0.6, 0.4, 0.0])
    edges = np.array([0.0, 1.0, 3.0, 5.0])
    u, mass = endpoint_quadrature(w0, w1)
    assert np.all(mass > 0)
    np.testing.assert_allclose(mass.sum(), 1.0)
    for weights in [w0, w1]:
        values = histogram_quantile(u, weights, edges)
        assert np.isfinite(values).all()
        # Uniform density within each bin has a known analytic mean.
        np.testing.assert_allclose(
            values @ mass, weights @ ((edges[:-1] + edges[1:]) / 2)
        )


def test_mie_small_particle_limit():
    pytest.importorskip("xarray")
    from cfm_project.preprocessing.aerosol import scattering_cross_section

    diameter = np.array([1.0, 2.0, 3.0])
    x = np.pi * diameter / 550
    refractive = 1.5
    rayleigh = (8 / 3) * x**4 * ((refractive**2 - 1) / (refractive**2 + 2)) ** 2
    rayleigh *= np.pi * (diameter / 2) ** 2 * 1e-6
    np.testing.assert_allclose(
        scattering_cross_section(diameter, 550, refractive), rayleigh, rtol=0.001
    )
