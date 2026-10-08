import torch
import pytest

from cfm_project.training import (
    _active_moment_blocks_for_step,
    _block_lambdas_for_moment_blocks,
    _merge_block_lambdas,
    _parse_moment_block_schedule,
    _targets_for_moment_blocks,
)


def test_parse_moment_block_schedule_uses_fraction_boundaries() -> None:
    schedule = _parse_moment_block_schedule(
        raw_schedule=[
            {"until_fraction": 0.5, "blocks": ["covariance"]},
            {"until_fraction": 1.0, "blocks": ["covariance", "y_fourth_central"]},
        ],
        total_steps=10,
        moment_feature_blocks=("covariance", "y_fourth_central"),
        moment_block_normalization="sqrt_dim",
    )

    assert [phase["end_step"] for phase in schedule] == [5, 10]
    assert _active_moment_blocks_for_step(schedule, 0) == ("covariance",)
    assert _active_moment_blocks_for_step(schedule, 4) == ("covariance",)
    assert _active_moment_blocks_for_step(schedule, 5) == (
        "covariance",
        "y_fourth_central",
    )


def test_moment_block_schedule_requires_independent_block_lambdas() -> None:
    with pytest.raises(ValueError, match="independent Lagrange multipliers"):
        _parse_moment_block_schedule(
            raw_schedule=[{"until_fraction": 1.0, "blocks": ["covariance"]}],
            total_steps=10,
            moment_feature_blocks=("covariance", "y_fourth_central"),
            moment_block_normalization="none",
        )


def test_targets_for_moment_blocks_selects_active_blocks_from_full_targets() -> None:
    targets = {0.5: torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0])}

    covariance_targets = _targets_for_moment_blocks(
        targets=targets,
        dim=2,
        source_blocks=("covariance", "y_fourth_central"),
        selected_blocks=("covariance",),
    )
    combined_targets = _targets_for_moment_blocks(
        targets=targets,
        dim=2,
        source_blocks=("covariance", "y_fourth_central"),
        selected_blocks=("covariance", "y_fourth_central"),
    )

    assert torch.allclose(covariance_targets[0.5], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert combined_targets is targets


def test_active_block_lambda_update_can_be_merged_without_touching_inactive_blocks() -> None:
    lambdas = {
        0.5: {
            "covariance": torch.zeros(4),
            "y_fourth_central": torch.ones(1),
        }
    }

    active = _block_lambdas_for_moment_blocks(lambdas, ("covariance",))
    updated = {0.5: {"covariance": torch.full((4,), 2.0)}}
    merged = _merge_block_lambdas(active, updated)

    assert "y_fourth_central" not in merged[0.5]

    merged_full = _merge_block_lambdas(lambdas, updated)
    assert torch.allclose(merged_full[0.5]["covariance"], torch.full((4,), 2.0))
    assert torch.allclose(merged_full[0.5]["y_fourth_central"], torch.ones(1))
