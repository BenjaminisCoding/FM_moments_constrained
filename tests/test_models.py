import torch

from cfm_project.models import PathCorrection


def test_path_correction_zero_init_output_starts_at_zero() -> None:
    model = PathCorrection(
        state_dim=3,
        hidden_dims=[8, 8],
        activation="silu",
        zero_init_output=True,
    )
    t = torch.rand(5, 1)
    x0 = torch.randn(5, 3)
    x1 = torch.randn(5, 3)

    out = model(t, x0, x1)

    assert torch.allclose(out, torch.zeros_like(out))
