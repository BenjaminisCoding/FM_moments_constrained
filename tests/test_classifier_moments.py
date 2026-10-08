from __future__ import annotations

from dataclasses import fields
import hashlib
import json

import numpy as np
import pytest
import torch

from cfm_project.classifier_moments import empirical_feature_mean, load_classifier_moment_inputs
from cfm_project.pipeline import _build_problem_and_targets
from cfm_project.pseudo_labels import PseudoLabelPreparedData
from cfm_project import single_cell_data as sc


def test_empirical_mean_weights_cells_not_batches():
    x = torch.arange(5, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    result = empirical_feature_mean(lambda z: torch.cat([z, z.square()], 1), x, batch_size=2)
    torch.testing.assert_close(result, torch.tensor([2., 6.], dtype=torch.float64))
    assert not result.requires_grad


@pytest.mark.parametrize("bad", [torch.empty(0, 2), torch.tensor([[float('nan'), 1.]])])
def test_empirical_mean_rejects_invalid_observations(bad):
    with pytest.raises(ValueError, match="Moment samples"):
        empirical_feature_mean(lambda x: x, bad)


def test_empirical_mean_rejects_invalid_map():
    with pytest.raises(ValueError, match="observation map"):
        empirical_feature_mean(lambda x: x * float('nan'), torch.ones(2, 1))


@pytest.fixture
def classifier_dataset(monkeypatch, tmp_path):
    x = np.tile(np.arange(4, dtype=np.float32), 4).reshape(-1, 1)
    x = np.concatenate([x, np.zeros_like(x)], axis=1)
    times = np.repeat(np.arange(4), 4)
    labels = np.tile(np.array(["A", "A", "A", "B"]), 4)
    monkeypatch.setattr(sc, "_load_single_cell_dataset", lambda data_cfg: (
        x.copy(), times.copy(), {"cell_type": labels.copy()}, None
    ))

    def fixed_classifier(**kwargs):
        assert kwargs["features_np"].shape == (4, 2)
        assert kwargs["features_np"][:, 0].max() < 10  # No evaluation-day fit data.
        temperature = kwargs["single_cfg"]["pseudo_labels"]["posterior_temperature"]

        def posterior(z):
            # Deliberately imperfect: means differ from the 75%/25% hard counts.
            p = .1 + .2 * z[:, :1] / temperature
            return torch.cat([p, 1-p], 1)

        values = {field.name: None for field in fields(PseudoLabelPreparedData)}
        values.update(
            posterior=posterior, selected_k=2,
            method=kwargs["single_cfg"]["pseudo_labels"]["method"],
            posterior_temperature=temperature, class_labels=["A", "B"],
            bic_by_k={}, stability_by_k={}, cache_hit=False,
        )
        return PseudoLabelPreparedData(**values)

    monkeypatch.setattr(sc, "prepare_pseudo_labels", fixed_classifier)
    cfg = dict(
        label="moment_consistency", family="single_cell", dim=2, constraint_times=[.2],
        constraint_time_policy="observed_nonendpoint_all",
        single_cell=dict(
            path=str(tmp_path / "fixed.h5ad"), max_dim=2, whiten=False,
            normalized_time_values=[0., .2, .4, 1.], constraint_times_normalized=[.2],
            eval_times_normalized=[.4], pseudo_labels=dict(
                enabled=True, method="supervised_mlp", supervised_label_key="cell_type",
                posterior_temperature=2., fit_times_normalized=[.2], target_kind="posterior_mean",
            ),
        ),
    )
    return cfg, x


def build(cfg):
    return sc.prepare_single_cell_problem_and_targets(
        data_cfg=cfg, experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"), dtype=torch.float32,
    )


@pytest.mark.parametrize("method", ["supervised_mlp", "supervised_logreg"])
@pytest.mark.parametrize("classes, mode, expected", [
    (None, "independent", [.25, .75]),
    (["B"], "independent", [.75]),
    (["A", "B"], "sum", [1.]),
])
def test_target_uses_actual_temperature_and_projected_map(classifier_dataset, method, classes, mode, expected):
    cfg, x = classifier_dataset
    cfg["single_cell"]["pseudo_labels"].update(
        method=method, constraint_classes=classes, constraint_class_mode=mode,
    )
    p = build(cfg)
    torch.testing.assert_close(p.pseudo_targets[.2], torch.tensor(expected))
    actual = p.pseudo_posterior(p.target_samples_by_time[.2]).mean(0)
    torch.testing.assert_close(p.pseudo_targets[.2], actual)
    assert p.pseudo_target_kind == "posterior_mean"
    assert p.pseudo_target_source == "observed_constraint_posterior_mean"
    x[8:12, 0] = 1000  # Changing the unconstrained day cannot change this observation.
    torch.testing.assert_close(build(cfg).pseudo_targets[.2], p.pseudo_targets[.2])


@pytest.mark.parametrize("kind", ["auto", "label_proportions"])
def test_label_frequency_targets_are_available(classifier_dataset, kind):
    cfg, _ = classifier_dataset
    cfg["single_cell"]["pseudo_labels"]["target_kind"] = kind
    p = build(cfg)
    torch.testing.assert_close(p.pseudo_targets[.2], torch.tensor([.75, .25]))
    assert p.pseudo_target_kind == "label_proportions"


def test_overrides_and_provenance_remain_explicit(classifier_dataset):
    cfg, _ = classifier_dataset
    cfg["single_cell"]["pseudo_labels"].update(
        target_overrides={"0.2": [.4, .6]}, target_override_source="Externally measured moment",
    )
    p = build(cfg)
    torch.testing.assert_close(p.pseudo_targets[.2], torch.tensor([.4, .6]))
    assert p.pseudo_target_kind == "provided"
    assert p.pseudo_target_source == "Externally measured moment"


def test_pipeline_reports_target_kind(classifier_dataset):
    cfg, _ = classifier_dataset
    built = _build_problem_and_targets(
        {"data": cfg, "experiment": {"protocol": "no_leaveout"}},
        torch.device("cpu"), torch.float32,
    )
    for key in ("data_build_meta", "pseudo_summary_meta"):
        assert built[key]["pseudo_target_kind"] == "posterior_mean"


def test_unknown_target_kind_rejected(classifier_dataset):
    cfg, _ = classifier_dataset
    cfg["single_cell"]["pseudo_labels"]["target_kind"] = "posterior_meen"
    with pytest.raises(ValueError, match="target_kind"):
        build(cfg)


def test_frozen_bundle_loads_without_cells_and_rejects_changed_classifier(tmp_path):
    model = torch.nn.Linear(2, 2)
    torch.save(dict(model=model, temperature=1.), tmp_path / "classifier.pt")
    target = np.array([.4, .6])
    np.savez(tmp_path / "training_inputs.npz", x0=np.zeros((3, 2)), x1=np.ones((3, 2)),
             target=target, tau=.2, sigma=.1)
    np.savez(tmp_path / "endpoint_coupling.npz", src=np.arange(3), tgt=np.arange(3), mass=np.ones(3)/3)
    protocol = dict(
        schema="empirical_classifier_moment_v1", target_kind="posterior_mean",
        target=target.tolist(), tau=.2, temperature=1.,
        training_sha256={name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                        for name in ("training_inputs.npz", "classifier.pt", "endpoint_coupling.npz")},
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    data, q, _ = load_classifier_moment_inputs(tmp_path)
    np.testing.assert_array_equal(data["target"], target)
    x = torch.ones(2, 2, requires_grad=True)
    assert torch.isfinite(torch.autograd.grad(q(x)[:, 0].sum(), x)[0]).all()
    (tmp_path / "classifier.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="classifier.pt"):
        load_classifier_moment_inputs(tmp_path)
