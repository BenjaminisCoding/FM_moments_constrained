from __future__ import annotations

import torch

from cfm_project.training import _cfm_loss


class _CaptureVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_x: torch.Tensor | None = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x.detach().clone()
        return torch.zeros_like(x)


def test_generic_velocity_input_noise_is_reproducible_and_target_only_stays_clean() -> None:
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    clean_model = _CaptureVelocity()
    noisy_model = _CaptureVelocity()
    clean_time = torch.Generator().manual_seed(123)
    noisy_time = torch.Generator().manual_seed(123)
    noise_generator = torch.Generator().manual_seed(456)

    clean_loss, clean_energy = _cfm_loss(
        mode="baseline",
        v_model=clean_model,
        g_model=None,
        x0=x0,
        x1=x1,
        time_generator=clean_time,
    )
    noisy_loss, noisy_energy = _cfm_loss(
        mode="baseline",
        v_model=noisy_model,
        g_model=None,
        x0=x0,
        x1=x1,
        time_generator=noisy_time,
        velocity_input_noise_sigma=0.1,
        noise_generator=noise_generator,
    )

    assert clean_model.last_x is not None
    assert noisy_model.last_x is not None
    expected_generator = torch.Generator().manual_seed(456)
    expected_noise = 0.1 * torch.randn(x0.shape, generator=expected_generator)
    assert torch.allclose(noisy_model.last_x - clean_model.last_x, expected_noise)
    # The model predicts zero, so both losses depend only on the unchanged target x1-x0.
    assert torch.allclose(noisy_loss, clean_loss)
    assert noisy_energy == clean_energy


def test_generic_velocity_input_noise_rejects_negative_sigma() -> None:
    model = _CaptureVelocity()
    try:
        _cfm_loss(
            mode="baseline",
            v_model=model,
            g_model=None,
            x0=torch.zeros(2, 1),
            x1=torch.ones(2, 1),
            velocity_input_noise_sigma=-0.1,
        )
    except ValueError as error:
        assert "velocity_input_noise_sigma must be non-negative" in str(error)
    else:
        raise AssertionError("negative velocity input noise sigma was accepted")


def test_generic_velocity_time_sampling_applies_beta_alpha_one_transform() -> None:
    x0 = torch.zeros(8, 2)
    x1 = torch.ones(8, 2)
    model = _CaptureVelocity()
    actual_generator = torch.Generator().manual_seed(321)
    expected_generator = torch.Generator().manual_seed(321)
    expected_t = torch.rand(8, 1, generator=expected_generator).pow(1.0 / 0.5)

    _cfm_loss(
        mode="baseline",
        v_model=model,
        g_model=None,
        x0=x0,
        x1=x1,
        time_generator=actual_generator,
        time_sampling_alpha=0.5,
    )

    assert model.last_x is not None
    assert torch.allclose(model.last_x, expected_t.expand_as(model.last_x))


def test_generic_velocity_time_sampling_rejects_nonpositive_alpha() -> None:
    model = _CaptureVelocity()
    try:
        _cfm_loss(
            mode="baseline",
            v_model=model,
            g_model=None,
            x0=torch.zeros(2, 1),
            x1=torch.ones(2, 1),
            time_sampling_alpha=0.0,
        )
    except ValueError as error:
        assert "time_sampling_alpha must be positive" in str(error)
    else:
        raise AssertionError("nonpositive velocity time-sampling alpha was accepted")
