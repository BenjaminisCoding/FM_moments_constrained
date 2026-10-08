"""Empirical moments of a fixed observation map and frozen training inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@torch.no_grad()
def empirical_feature_mean(
    feature_map: Callable[[torch.Tensor], torch.Tensor],
    samples: torch.Tensor,
    *,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Average the actual feature map over every supplied cell, accumulating in float64.

    The result is an empirical observation, not a population-expectation guarantee.
    Temperature, class projection and any other transformation belong inside
    ``feature_map`` so the observation and training constraint have identical meaning.
    """
    if samples.ndim != 2 or len(samples) == 0 or not torch.isfinite(samples).all():
        raise ValueError("Moment samples must be a nonempty finite (N, d) tensor.")
    if batch_size < 1:
        raise ValueError("Moment batch_size must be positive.")
    total = None
    for batch in samples.split(batch_size):
        values = feature_map(batch)
        if (
            values.ndim != 2
            or len(values) != len(batch)
            or values.shape[1] == 0
            or not torch.isfinite(values).all()
        ):
            raise ValueError("The observation map must return finite (N, K) features.")
        # CPU float64 also supports classifiers evaluated on devices without float64.
        subtotal = values.detach().to(device="cpu", dtype=torch.float64).sum(dim=0)
        if total is not None and subtotal.shape != total.shape:
            raise ValueError("Observation dimension changed between batches.")
        total = subtotal if total is None else total + subtotal
    return total / len(samples)


def load_classifier_moment_inputs(
    prepared_dir: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[dict[str, np.ndarray], Callable, dict]:
    """Load an audited endpoint/aggregate bundle without reading intermediate cells.

    Fitting and diagnostics use the same frozen target and observation map.
    Training-file hashes are verified before the arrays and classifier load.
    """
    prepared = Path(prepared_dir)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if protocol.get("schema") != "empirical_classifier_moment_v1":
        raise ValueError("Unsupported classifier moment input schema.")
    if protocol.get("target_kind") != "posterior_mean":
        raise ValueError("Expected an empirical posterior-mean target.")
    for name in ("training_inputs.npz", "classifier.pt", "endpoint_coupling.npz"):
        actual = hashlib.sha256((prepared / name).read_bytes()).hexdigest()
        if actual != protocol["training_sha256"][name]:
            raise ValueError(f"Frozen classifier moment input changed: {name}")
    with np.load(prepared / "training_inputs.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if set(data) != {"x0", "x1", "target", "tau", "sigma"}:
        raise ValueError("Training inputs must contain only endpoints and declared aggregates.")
    if not np.array_equal(data["target"], np.asarray(protocol["target"])):
        raise ValueError("Target does not match its declared observation.")
    if float(data["tau"]) != float(protocol["tau"]):
        raise ValueError("Constraint time does not match its declared observation.")
    if not np.isfinite(data["target"]).all() or np.any(data["target"] < 0):
        raise ValueError("Classifier moment target must be finite and nonnegative.")
    saved = torch.load(prepared / "classifier.pt", map_location=device, weights_only=False)
    model, temperature = saved["model"], float(saved["temperature"])
    if temperature != float(protocol["temperature"]) or temperature <= 0:
        raise ValueError("Classifier temperature does not match its declared observation.")
    model.eval().requires_grad_(False)

    def posterior(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(x.to(device)) / temperature, dim=1).to(x.device)

    posterior.log_prob = lambda x: torch.log_softmax(
        model(x.to(device)) / temperature, dim=1
    ).to(x.device)
    return data, posterior, protocol
