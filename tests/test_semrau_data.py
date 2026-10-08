"""Scientific data-contract checks; no network or benchmark data required."""
import io

import pytest

from cfm_project.semrau_data import (
    read_counts,
    safe_shared_genes, split_endpoint_cells, synthetic_ercc_mask,
)


def test_decimal_comma_and_ambiguous_gene_names_are_not_silently_repaired():
    bulk = read_counts(io.StringIO("\t0h\t48h\nRn45s\t1\t1,07E+09\nMar-01\t2\t3\nMar-01\t4\t5\nNanog\t6\t7\n"), bulk=True)
    assert bulk.loc["Rn45s", "48h"] == 1070000000
    assert bulk.index.tolist().count("Mar-01") == 2
    cell = read_counts(io.StringIO("\tA1\nRn45s\t4\nNanog\t5\nMarch1\t2\n"))
    assert safe_shared_genes(cell, bulk) == ["Nanog"]


def test_endogenous_ercc_genes_are_not_spike_ins():
    assert synthetic_ercc_mask(["Ercc1", "Ercc6l", "ERCC-00002", "ERCC_00171", "Ercc8"]).tolist() == [False, False, True, True, False]


def test_splits_are_disjoint_exhaustive_and_reproducible():
    ids = [f"GSM:A{i}" for i in range(137)]
    split = split_endpoint_cells(ids,79578)
    assert split == split_endpoint_cells(ids,79578)
    assert len(set(sum(split.values(),[]))) == len(ids)
    assert set(sum(split.values(),[])) == set(ids)
    assert set(split["train"]).isdisjoint(split["evaluation"])


@pytest.mark.parametrize("value", ["-1", "nan", "0.5", "oops"])
def test_invalid_umi_values_are_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        read_counts(io.StringIO(f"\tA1\nNanog\t{value}\n"))
