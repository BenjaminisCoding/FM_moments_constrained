"""Public Mendeley H5AD PCA coordinates -> endpoint scaling, OT and classifier moments.

The supplied PCA coordinates and bundled, constraint-day-trained classifiers are
frozen study inputs. This does not recreate upstream PCA from sequencing counts
or refit the observation classifier. Constraint-day cells are used only to
measure its empirical posterior mean; LAND references remain endpoint-only.
"""

import hashlib
import json
import shutil

import anndata as ad
import numpy as np
import ot
import torch
from scipy.spatial.distance import cdist

from cfm_project.classifier_moments import empirical_feature_mean

from .common import ROOT, digest, sources, write_json


def prepare(raw, out, download=False):
    raw_directory = raw.parent if raw.suffix == ".h5ad" else raw
    provenance = sources("multi", raw_directory, download)
    path = raw_directory / provenance[0]["filename"]
    if raw.suffix == ".h5ad" and raw.name != path.name:
        raise ValueError(f"Expected the published dataset filename: {path.name}")
    source = ad.read_h5ad(path, backed="r")
    try:
        if (
            "X_pca" not in source.obsm
            or "day" not in source.obs
            or "cell_type" not in source.obs
        ):
            raise ValueError(
                "Multi requires obsm['X_pca'] and obs['day'], obs['cell_type']"
            )
        embedding = np.asarray(source.obsm["X_pca"][:, :100], dtype=np.float32)
        if embedding.shape[1] != 100 or not np.isfinite(embedding).all():
            raise ValueError("Expected 100 finite PCA coordinates per cell")
        days = np.asarray(source.obs["day"], dtype=float)
        by_day = {d: np.flatnonzero(days == d) for d in [2, 3, 4, 7]}

        def subset(day, offset):
            return np.sort(
                np.random.default_rng(20260801 + offset).choice(
                    by_day[day], 2000, replace=False
                )
            )

        endpoints = {d: subset(d, d) for d in [2, 7]}
        validation = {d: subset(d, 100 + d) for d in [3, 4]}
        joined = embedding[np.concatenate(list(endpoints.values()))]
        mean = joined.mean(0, dtype=np.float64)
        scale = np.maximum(joined.std(0, dtype=np.float64), 1e-8)
        embedding = ((embedding.astype(np.float64) - mean) / scale).astype(np.float32)
    finally:
        source.file.close()
    x0, x1 = [embedding[endpoints[d]] for d in [2, 7]]
    sigma = np.sqrt(
        0.5 * (x0.var(0, dtype=np.float64).mean() + x1.var(0, dtype=np.float64).mean())
    )
    mass, log = ot.emd(
        np.full(2000, 1 / 2000),
        np.full(2000, 1 / 2000),
        cdist(x0, x1, "sqeuclidean"),
        log=True,
        numItermax=1000000,
    )
    if log.get("warning"):
        raise RuntimeError(log["warning"])
    i, j = np.where(mass > 1e-12)
    source_hash = digest(path)
    endpoint_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in [x0, x1]]
    for constraint, evaluation in [(3, 4), (4, 3)]:
        name = f"constraint_day{constraint}_eval_day{evaluation}"
        folder = out / name
        folder.mkdir()
        frozen = ROOT / "benchmark_data/multi" / name
        protocol = json.loads((frozen / "protocol.json").read_text())
        if endpoint_hashes != protocol["endpoint_hashes"]:
            raise ValueError(
                "Source PCA/subsets differ from the study: the frozen classifier cannot be reused."
            )
        classifier = frozen / "classifier.pt"
        if digest(classifier) != protocol["training_sha256"]["classifier.pt"]:
            raise ValueError("Bundled observation classifier checksum mismatch")
        saved = torch.load(classifier, map_location="cpu", weights_only=False)
        model = saved["model"].eval().requires_grad_(False)
        temperature = float(saved["temperature"])

        def posterior(x, model=model, temperature=temperature):
            return torch.softmax(model(x) / temperature, dim=1)

        target = empirical_feature_mean(
            posterior, torch.from_numpy(embedding[by_day[constraint]])
        ).numpy()
        tau = (constraint - 2) / 5
        np.savez_compressed(
            folder / "training_inputs.npz",
            x0=x0,
            x1=x1,
            target=target,
            tau=tau,
            sigma=sigma,
        )
        np.savez_compressed(
            folder / "endpoint_coupling.npz", src=i, tgt=j, mass=mass[i, j]
        )
        shutil.copyfile(classifier, folder / "classifier.pt")
        pools = {
            **endpoints,
            constraint: by_day[constraint],
            evaluation: validation[evaluation],
        }
        np.savez_compressed(
            folder / "evaluation_only.npz",
            **{f"t_{(d - 2) / 5:g}": embedding[pools[d]] for d in [2, 3, 4, 7]},
        )
        np.save(folder / "validation.npy", embedding[validation[evaluation]])
        protocol.update(
            target=target.tolist(),
            sigma=float(sigma),
            preprocessing="Source PCA100; fixed uniform subsets; endpoint-only standardization.",
            source_sha256=source_hash,
            sources=provenance,
            source_filename=path.name,
            target_source="Full raw constraint-day empirical mean of the bundled frozen classifier",
            observation_sample_count=len(by_day[constraint]),
            endpoint_hashes=endpoint_hashes,
        )
        protocol["training_sha256"] = {
            n: digest(folder / n)
            for n in ["training_inputs.npz", "classifier.pt", "endpoint_coupling.npz"]
        }
        write_json(folder / "protocol.json", protocol)
    np.savez_compressed(
        out / "representation.npz",
        mean=mean,
        scale=scale,
        endpoint_day2_indices=endpoints[2],
        endpoint_day7_indices=endpoints[7],
        validation_day3_indices=validation[3],
        validation_day4_indices=validation[4],
    )
