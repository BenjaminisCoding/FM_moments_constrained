from __future__ import annotations

import torch

from cfm_project.multimarginal import (
    build_persistent_minibatch_ot_proposal,
    build_adjacent_ot_segments,
    local_to_global_time_and_velocity,
)


def test_persistent_minibatch_ot_proposal_covers_all_endpoints_each_pass() -> None:
    source = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    target = torch.arange(10, dtype=torch.float32).reshape(5, 2) + 20.0
    proposal = build_persistent_minibatch_ot_proposal(
        source=source,
        target=target,
        time=0.25,
        batch_size=4,
        passes=2,
        generator=torch.Generator().manual_seed(41),
    )

    assert tuple(proposal.samples.shape) == (16, 2)
    assert int(proposal.source_coverage.min()) >= 2
    assert int(proposal.target_coverage.min()) >= 2
    expected = (
        0.75 * source[proposal.source_indices]
        + 0.25 * target[proposal.target_indices]
    )
    assert torch.allclose(proposal.samples, expected)


def test_local_half_interval_velocity_is_rescaled_to_global_time() -> None:
    local_time = torch.tensor([[0.0], [0.5], [1.0]])
    local_velocity = torch.ones(3, 2)
    global_time, global_velocity = local_to_global_time_and_velocity(
        local_time=local_time,
        local_velocity=local_velocity,
        start_time=0.5,
        end_time=1.0,
    )
    assert torch.allclose(global_time, torch.tensor([[0.5], [0.75], [1.0]]))
    assert torch.allclose(global_velocity, torch.full((3, 2), 2.0))


def test_balanced_adjacent_ot_supports_unequal_pool_sizes(tmp_path) -> None:
    torch.manual_seed(23)
    p0 = torch.randn(7, 2)
    pm = torch.randn(9, 2) + 0.5
    p1 = torch.randn(8, 2) + 1.0
    segments = build_adjacent_ot_segments(
        snapshot_times=[0.0, 0.5, 1.0],
        snapshot_pools=[p0, pm, p1],
        label_prefix="unequal",
        ot_method="balanced_lp",
        balanced_ot_cache_dir=tmp_path,
    )
    assert len(segments) == 2
    assert all(segment.problem.has_global_ot_support for segment in segments)
    assert all(torch.isclose(segment.problem.global_ot_mass.sum(), torch.tensor(1.0)) for segment in segments)
    cached = list(tmp_path.glob("*.pt"))
    assert len(cached) == 2


def test_balanced_adjacent_ot_preserves_nonuniform_midpoint_masses(tmp_path) -> None:
    torch.manual_seed(29)
    p0 = torch.randn(5, 2)
    pm = torch.randn(7, 2) + 0.5
    p1 = torch.randn(6, 2) + 1.0
    midpoint_weights = torch.tensor([0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03])
    segments = build_adjacent_ot_segments(
        snapshot_times=[0.0, 0.5, 1.0],
        snapshot_pools=[p0, pm, p1],
        snapshot_weights=[None, midpoint_weights, None],
        label_prefix="weighted_midpoint",
        ot_method="balanced_lp",
        balanced_ot_cache_dir=tmp_path,
    )

    incoming = torch.zeros_like(midpoint_weights)
    incoming.scatter_add_(
        0,
        segments[0].problem.global_ot_tgt_idx,
        segments[0].problem.global_ot_mass,
    )
    outgoing = torch.zeros_like(midpoint_weights)
    outgoing.scatter_add_(
        0,
        segments[1].problem.global_ot_src_idx,
        segments[1].problem.global_ot_mass,
    )
    assert torch.allclose(incoming, midpoint_weights, atol=1.0e-6)
    assert torch.allclose(outgoing, midpoint_weights, atol=1.0e-6)
