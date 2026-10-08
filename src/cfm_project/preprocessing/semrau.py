"""GEO UMI tables -> endpoint-fitted representation and fixed bulk moments."""

import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from cfm_project.generalized_moment_sb import balance_log_kernel
from cfm_project.semrau_data import (
    load_bulk,
    load_scrb,
    safe_shared_genes,
    split_endpoint_cells,
)

from .common import sources, write_json

MARKERS = ["Sox2", "Tbx3", "Nanog", "Ppia"]
CALIBRATION = {"Sox2": [36, 96], "Tbx3": [24, 60, 96]}


def prepare(raw, out, download=False):
    provenance = sources("semrau", raw, download)
    frames = {}
    for hour in [0, 6, 12, 24, 36, 60, 96]:
        frame, _ = load_scrb(raw, hour)
        frames[hour] = frame.loc[:, frame.sum(axis=0) >= 2000]
    names = frames[0].index.to_numpy()
    if any(not np.array_equal(f.index, names) for f in frames.values()):
        raise ValueError("Inconsistent gene order across raw marginals")
    splits = {
        h: split_endpoint_cells(frames[h].columns, seed)
        for h, seed in [(0, 79578), (36, 795782536)]
    }

    def counts(hour, split=None):
        frame = frames[hour]
        return (
            (frame.loc[:, splits[hour][split]] if split else frame)
            .to_numpy(dtype=float)
            .T
        )

    marker_indices = [list(names).index(gene) for gene in MARKERS]
    joined = np.concatenate([counts(0, "train"), counts(36, "train")])
    log = np.log1p(10000 * joined / joined.sum(1, keepdims=True))
    variance = log.var(0)
    variance[marker_indices] = -1
    selected = np.argsort(-variance, kind="stable")[:1000]
    pca = PCA(n_components=8, svd_solver="full").fit(log[:, selected])
    pc_scale = np.sqrt(pca.explained_variance_)
    marker_scale = np.sqrt(np.maximum(joined[:, marker_indices].mean(0), 0.05))

    def transform(x):
        log = np.log1p(10000 * x / x.sum(1, keepdims=True))
        return np.c_[
            pca.transform(log[:, selected]) / pc_scale,
            np.sqrt(x[:, marker_indices]) / marker_scale,
        ]

    # Fixed per-gene bulk-to-UMI calibration subsets.
    bulk = load_bulk(raw)
    bulk = bulk.loc[safe_shared_genes(frames[0], bulk)]
    observations, target = [], []
    for gene, hours in CALIBRATION.items():

        def abundance(hour, gene=gene):
            column = bulk[f"{hour}h"]
            return float(1e6 * column.loc[gene] / column.sum())

        x = np.array([abundance(h) for h in hours])
        y = np.array([frames[h].loc[gene].mean() for h in hours])
        prediction = float((x @ y / (x @ x)) * abundance(6))
        observations.append(
            {"gene": gene, "calibration_hours": hours, "target_raw_UMI": prediction}
        )
        target.append(prediction / marker_scale[MARKERS.index(gene)] ** 2)
    x0, x1 = transform(counts(0, "train")), transform(counts(36, "train"))
    a, b = np.full(len(x0), 1 / len(x0)), np.full(len(x1), 1 / len(x1))
    cost = cdist(x0, x1, "sqeuclidean")
    exact, log = ot.emd(a, b, cost, log=True, numItermax=1000000)
    if log.get("warning"):
        raise RuntimeError(log["warning"])
    entropic = balance_log_kernel(
        -cost / 64, a, b, tolerance=2e-9, max_iterations=10000, newton_refine=True
    ).coupling
    matrix = np.zeros((2, 12))
    matrix[0, 8] = matrix[1, 9] = 1
    for name, coupling in [("exact", exact), ("entropic", entropic)]:
        folder = out / name
        folder.mkdir(parents=True)
        np.testing.assert_allclose(coupling.sum(1), a, atol=2e-9)
        np.testing.assert_allclose(coupling.sum(0), b, atol=2e-9)
        i, j = np.where(coupling > 1e-14)
        np.savez_compressed(
            folder / "train.npz",
            x0=x0,
            x1=x1,
            pair_i=i,
            pair_j=j,
            pair_weights=coupling[i, j],
            cell_ids0=splits[0]["train"],
            cell_ids1=splits[36]["train"],
        )
        write_json(
            folder / "metadata.json",
            {
                "setup": "sox2_tbx3_6h",
                "endpoints": [0, 36],
                "hours": [0, 6, 12, 24, 36],
                "constraint_hour": 6,
                "unconstrained_hours": [12, 24],
                "tau": 1 / 6,
                "markers": MARKERS,
                "matrix": matrix.tolist(),
                "target": target,
                "normalization": target,
                "observations": observations,
                "marker_scale": marker_scale.tolist(),
                "endpoint_cells": [len(x0), len(x1)],
                "support": len(i),
                "coupling": name,
                "epsilon": 64 if name == "entropic" else None,
                "representation": "8 endpoint-trained PCs + 4 square-root raw UMI marker coordinates",
                "sources": provenance,
                "interpretation": "Bulk-calibrated marker moments. Calibration uses evaluation-day and endpoint-validation cells; reported scores are calibration-dependent evaluations.",
                "calibration_selection_pool": "Constraint-time single-cell means",
            },
        )
    np.savez_compressed(
        out / "representation.npz",
        selected_genes=names[selected].astype(str),
        selected_indices=selected,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        pc_scale=pc_scale,
        marker_scale=marker_scale,
        markers=MARKERS,
    )
    evaluation = {}
    for hour in [0, 6, 12, 24, 36]:
        split = "evaluation" if hour in splits else None
        evaluation[f"x_{hour}"] = transform(counts(hour, split))
        evaluation[f"ids_{hour}"] = np.array(
            splits[hour][split] if split else frames[hour].columns
        ).astype(str)
    np.savez_compressed(out / "evaluation.npz", **evaluation)
    np.savez_compressed(
        out / "endpoint_validation.npz",
        x_0=transform(counts(0, "validation")),
        x_1=transform(counts(36, "validation")),
    )
    write_json(out / "endpoint_splits.json", splits)
