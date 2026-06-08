#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import CategoricalDtype

import cellrank as cr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Schiebinger serum pilot .h5ad with selected days, HVG-only features, "
            "and PCA embedding."
        )
    )
    parser.add_argument(
        "--days",
        type=float,
        nargs="+",
        default=[10.0, 10.5, 11.0],
        help="Observed day labels to keep from the serum subset.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of PCA components to compute into obsm['X_pca'].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/schiebinger_serum_d10_d10p5_d11_hvg_pca50.h5ad"),
        help="Output .h5ad path.",
    )
    return parser.parse_args()


def _require_columns(adata: ad.AnnData, columns: list[str]) -> None:
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")


def _remove_unused_categories(obs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = obs.copy()
    for col in columns:
        series = out[col]
        if isinstance(series.dtype, CategoricalDtype):
            out[col] = series.cat.remove_unused_categories()
    return out


def main() -> None:
    args = _parse_args()
    if int(args.n_pcs) <= 0:
        raise ValueError(f"--n-pcs must be positive, got {args.n_pcs}")

    selected_days = [str(float(day)) for day in args.days]
    if len(set(selected_days)) != len(selected_days):
        raise ValueError(f"--days contains duplicates: {selected_days}")

    adata = cr.datasets.reprogramming_schiebinger(subset_to_serum=True)
    _require_columns(adata, ["day", "cell_sets"])

    day_str = adata.obs["day"].astype(str)
    keep_mask = day_str.isin(selected_days).to_numpy()
    if not np.any(keep_mask):
        raise ValueError(
            "No cells matched requested days. "
            f"Requested={selected_days}, available={sorted(day_str.unique().tolist())}"
        )
    adata = adata[keep_mask].copy()

    if "highly_variable" not in adata.var.columns:
        raise KeyError(
            "Input adata.var is missing 'highly_variable'. "
            "Expected Schiebinger dataset metadata to include this column."
        )
    hvg_mask = adata.var["highly_variable"].astype(bool).to_numpy()
    if int(np.sum(hvg_mask)) <= 0:
        raise ValueError("No HVGs selected by adata.var['highly_variable'].")
    adata = adata[:, hvg_mask].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=int(args.n_pcs), zero_center=True, svd_solver="arpack")

    obs_out = _remove_unused_categories(adata.obs[["day", "cell_sets"]], ["day", "cell_sets"])
    out = ad.AnnData(X=adata.X.copy(), obs=obs_out, var=adata.var.copy())
    out.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path)

    unique_days = sorted(obs_out["day"].astype(str).unique().tolist(), key=float)
    print(f"[done] wrote: {output_path}")
    print(f"[info] n_cells={out.n_obs} n_genes_hvg={out.n_vars} n_pcs={out.obsm['X_pca'].shape[1]}")
    print(f"[info] days={unique_days}")
    print(
        "[info] cell_sets=",
        sorted(obs_out["cell_sets"].astype(str).unique().tolist()),
    )


if __name__ == "__main__":
    main()
