"""Public Semrau data, with assay provenance kept separate from model targets.

Loading an assay does not assert that it measures an expectation in another
assay's coordinates. In particular, this module never turns bulk CPM into a
single-cell mean target. Shared-gene calibration excludes synthetic ERCC measurements.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd


HOURS = (0, 6, 12, 24, 36, 48, 60, 72, 96)


def read_counts(stream, *, bulk: bool = False) -> pd.DataFrame:
    """Preserve date-damaged symbols and parse decimal-comma scientific numbers.

    Duplicate bulk symbols are deliberately not merged: Mar-01 can be either
    Marc1 or March1. Neither guessing nor summing is a defensible gene mapping.
    """
    frame = pd.read_csv(stream, sep="\t", index_col=0, keep_default_na=False,
                        decimal="," if bulk else ".")
    frame = frame.apply(pd.to_numeric, errors="raise")
    value = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(value).all() or (value < 0).any() or (value != np.floor(value)).any():
        raise ValueError("Expected finite, nonnegative integer UMI counts")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate sample/well identifiers within an assay")
    if not bulk and frame.index.duplicated().any():
        raise ValueError("Single-cell gene symbols must be unique")
    return frame.astype(np.int64)


def load_bulk(raw: Path) -> pd.DataFrame:
    with gzip.open(raw / "GSE79578_bulkRNAseq.txt.gz", "rt") as f:
        return read_counts(f, bulk=True)


def load_scrb(raw: Path, hour: int) -> tuple[pd.DataFrame, str]:
    if hour not in HOURS:
        raise ValueError(f"No SCRB-seq marginal at {hour} hours")
    suffix = f"_scrbseq_{'2i' if hour == 0 else str(hour) + 'h'}.txt.gz"
    with tarfile.open(raw / "GSE79578_RAW.tar") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(suffix)]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f"Expected one regular archive member for {hour}h")
        member = members[0]
        # Read in memory; never extract paths provided by a remote archive.
        with gzip.GzipFile(fileobj=archive.extractfile(member)) as compressed:
            frame = read_counts(io.TextIOWrapper(compressed))
    accession = member.name.split("_")[0]
    frame.columns = [f"{accession}:{well}" for well in frame.columns]
    return frame, member.name


def synthetic_ercc_mask(genes) -> np.ndarray:
    # Mouse DNA repair genes Ercc1,...,Ercc8 are NOT synthetic spike-ins.
    return np.array([bool(re.match(r"^ERCC[-_.]\d+$", str(g), re.I)) for g in genes])


def safe_shared_genes(single: pd.DataFrame, bulk: pd.DataFrame) -> list[str]:
    """Declared denominator: common unambiguous non-rRNA gene symbols.

    Excluding rRNA is also essential for the corrupted/extraordinary Rn45s
    entries in bulk. Values are preserved in the raw data, never guessed.
    """
    duplicated = set(bulk.index[bulk.index.duplicated(keep=False)])
    return [g for g in single.index if g in bulk.index and g not in duplicated
            and not re.fullmatch(r"(?:Mar|Sep)-\d\d", g)
            and not re.match(r"^Rn\d", g) and not synthetic_ercc_mask([g])[0]]


def split_endpoint_cells(ids, seed: int, train_fraction=.70, validation_fraction=.15) -> dict:
    ids = np.asarray(ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    a, b = int(len(ids) * train_fraction), int(len(ids) * (train_fraction + validation_fraction))
    return {"train": ids[perm[:a]].tolist(), "validation": ids[perm[a:b]].tolist(),
            "evaluation": ids[perm[b:]].tolist()}
