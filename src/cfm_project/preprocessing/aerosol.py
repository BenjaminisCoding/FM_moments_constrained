"""EBAS DMPS/CPC/nephelometer files -> calibrated June aerosol benchmark."""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.special import spherical_jn, spherical_yn

from .common import sources, write_json


def scattering_cross_section(diameter_nm, wavelength_nm, refractive_index):
    """Homogeneous-sphere Mie scattering cross section in square micrometers."""
    x = np.pi * np.asarray(diameter_nm, dtype=float) / wavelength_nm
    n = np.arange(1, int(np.ceil(x.max() + 4 * x.max() ** (1 / 3) + 2)) + 1)[:, None]
    z = x[None, :]
    mz = refractive_index * z
    psi = z * spherical_jn(n, z)
    dpsi = spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    psim = mz * spherical_jn(n, mz)
    dpsim = spherical_jn(n, mz) + mz * spherical_jn(n, mz, derivative=True)
    xi = z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    dxi = (
        spherical_jn(n, z)
        + 1j * spherical_yn(n, z)
        + z
        * (
            spherical_jn(n, z, derivative=True)
            + 1j * spherical_yn(n, z, derivative=True)
        )
    )
    a = (refractive_index * psim * dpsi - psi * dpsim) / (
        refractive_index * psim * dxi - xi * dpsim
    )
    b = (psim * dpsi - refractive_index * psi * dpsim) / (
        psim * dxi - refractive_index * xi * dpsim
    )
    q = 2 / x**2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2), axis=0)
    return q * np.pi * (np.asarray(diameter_nm) / 2) ** 2 * 1e-6


def pm1_transmission(diameter_nm):
    """Declared nominal inlet response; residual mismatch is audited separately."""
    return 1 / (
        1 + np.exp(np.clip((np.log(diameter_nm) - np.log(1000)) / 0.035, -60, 60))
    )


def log_bin_edges(diameter_nm):
    logd = np.log(np.asarray(diameter_nm, dtype=float))
    return np.r_[
        logd[0] - (logd[1] - logd[0]) / 2,
        (logd[:-1] + logd[1:]) / 2,
        logd[-1] + (logd[-1] - logd[-2]) / 2,
    ]


def mean_bin_response(diameter_nm, wavelength_nm, refractive_index):
    # The size instrument reports density per log bin. Use the same uniform
    # within-log-bin representation as the later population quantiles.
    edges = log_bin_edges(diameter_nm)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    d = np.exp(
        (edges[1:] + edges[:-1])[:, None] / 2 + np.diff(edges)[:, None] * nodes / 2
    )
    values = scattering_cross_section(
        d.ravel(), wavelength_nm, refractive_index
    ) * pm1_transmission(d.ravel())
    return values.reshape(d.shape) @ (weights / 2)


def load_series(size_path, cpc_path, neph_path, stop=None):
    end = stop or "2012-07-02"

    def read(path):
        with xr.open_dataset(path) as dataset:
            return dataset.sel(time=slice("2012-01-01", end)).load()

    sizes, cpc, neph = [read(path) for path in (size_path, cpc_path, neph_path)]
    time = pd.DatetimeIndex(sizes.time.values).floor("h")
    d = sizes.D.values.astype(float)
    number = sizes.particle_number_size_distribution_amean.transpose("time", "D").values
    qc = sizes.particle_number_size_distribution_amean_qc.values
    size_valid = np.isfinite(number).all(1) & (number >= 0).all(1)
    # EBAS flags are often masked at their zero fill value. Nonzero finite
    # flags are conservatively excluded. This uses data quality, not fit error.
    size_valid &= ~np.any(np.isfinite(qc) & (qc != 0), axis=tuple(range(qc.ndim - 1)))
    n1 = cpc.particle_number_concentration_aerosol_amean.values
    n2 = cpc.particle_number_concentration_pm10_amean.values
    total = np.where(np.isfinite(n1), n1, n2)

    def unflagged(values):
        # EBAS 100 explicitly means checked/valid and overrides invalid flags.
        invalid = np.isfinite(values) & (values != 0) & (values != 100)
        return ~np.any(invalid, axis=tuple(range(values.ndim - 1)))

    counter_good = np.where(
        np.isfinite(n1),
        unflagged(cpc.particle_number_concentration_aerosol_amean_qc.values),
        unflagged(cpc.particle_number_concentration_pm10_amean_qc.values),
    )
    cpc_series = pd.Series(
        total, index=pd.DatetimeIndex(cpc.time.values).floor("h")
    ).reindex(time)
    counter_good = pd.Series(
        counter_good, index=pd.DatetimeIndex(cpc.time.values).floor("h")
    ).reindex(time, fill_value=False)
    opt = neph.aerosol_light_scattering_coefficient_amean.transpose(
        "time", "Wavelength"
    ).values
    optics = pd.DataFrame(
        opt,
        index=pd.DatetimeIndex(neph.time.values).floor("h"),
        columns=neph.Wavelength.values,
    ).reindex(time)
    optical_good = unflagged(neph.aerosol_light_scattering_coefficient_amean_qc.values)
    optical_good = pd.Series(
        optical_good, index=pd.DatetimeIndex(neph.time.values).floor("h")
    ).reindex(time, fill_value=False)
    rh = pd.Series(
        neph.relative_humidity.values.reshape(-1),
        index=pd.DatetimeIndex(neph.time.values).floor("h"),
    ).reindex(time)
    logd = np.log10(d)
    edges = np.r_[
        logd[0] - (logd[1] - logd[0]) / 2,
        (logd[:-1] + logd[1:]) / 2,
        logd[-1] + (logd[-1] - logd[-2]) / 2,
    ]
    counts = number * np.diff(edges)[None, :]
    size_total = counts.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = counts / size_total[:, None]
    valid = size_valid & np.isfinite(cpc_series.values) & (cpc_series.values > 100)
    valid &= np.isfinite(optics.values).all(1) & (optics.values > 0).all(1)
    valid &= np.isfinite(rh.values) & (rh.values < 40) & (size_total > 50)
    valid &= counter_good.values & optical_good.values
    return {
        "time": time,
        "diameter": d,
        "weights": weights,
        "number": number,
        "size_total": size_total,
        "cpc": cpc_series.values,
        "optics": optics.values,
        "wavelengths": optics.columns.values,
        "valid": valid,
        "rh": rh.values,
        "source_files": [str(p) for p in [size_path, cpc_path, neph_path]],
    }


def histogram_quantile(u, weights, edges):
    cumulative = np.r_[0.0, np.cumsum(weights)]
    cumulative[-1] = 1.0
    indices = np.clip(
        np.searchsorted(cumulative, u, side="right") - 1, 0, len(weights) - 1
    )
    return (
        edges[indices]
        + (u - cumulative[indices]) / weights[indices] * np.diff(edges)[indices]
    )


def endpoint_quadrature(weights0, weights1, nodes_per_segment=4):
    # Adaptive probability intervals use ENDPOINT histograms only. Retaining
    # rare-particle mass matters because optical response grows rapidly in D.
    knots = np.unique(np.r_[0.0, np.cumsum(weights0), np.cumsum(weights1), 1.0])
    knots = np.clip(knots, 0, 1)
    left, right = knots[:-1], knots[1:]
    keep = (right - left) > 1e-12
    left, right = left[keep], right[keep]
    z, w = np.polynomial.legendre.leggauss(nodes_per_segment)
    u = ((left + right)[:, None] / 2 + (right - left)[:, None] * z / 2).ravel()
    mass = ((right - left)[:, None] * w / 2).ravel()
    mass /= mass.sum()
    return u, mass


def prepare(raw, out, download=False):
    provenance = sources("aerosol", raw, download)
    data = load_series(*(raw / source["filename"] for source in provenance))
    calibration = data["valid"] & (data["time"] < pd.Timestamp("2012-03-01"))
    if calibration.sum() == 0:
        raise ValueError("No valid January-February observations for calibration")
    measured = data["optics"][calibration] / data["cpc"][calibration, None]

    def error(parameters):
        kernels = np.stack(
            [
                mean_bin_response(data["diameter"], lam, parameters[0])
                for lam in data["wavelengths"]
            ],
            axis=1,
        )
        predicted = np.exp(parameters[1]) * data["weights"][calibration] @ kernels
        return (np.log(predicted) - np.log(measured)).ravel()

    fitted = least_squares(
        error,
        [1.5, 0.0],
        bounds=([1.33, np.log(0.4)], [1.85, np.log(2.5)]),
        loss="soft_l1",
        f_scale=0.25,
        max_nfev=120,
    )
    if not fitted.success:
        raise RuntimeError(f"Optical calibration failed: {fitted.message}")
    grid = np.geomspace(1.0, 2000.0, 2048)
    response = (
        np.exp(fitted.x[1])
        * scattering_cross_section(grid, 550, fitted.x[0])
        * pm1_transmission(grid)
    )
    index = {stamp: i for i, stamp in enumerate(data["time"])}
    # Fixed date window: select the first complete QC-passing day.
    for day in pd.date_range("2012-06-01", "2012-06-07"):
        times = [day + pd.Timedelta(hours=h) for h in [6, 9, 12, 15, 18]]
        ids = [index.get(t, -1) for t in times]
        if min(ids) >= 0 and data["valid"][ids].all():
            break
    else:
        raise ValueError("No quality-qualified day in the fixed June window")
    edges = log_bin_edges(data["diameter"])
    w0, w1 = data["weights"][ids[0]], data["weights"][ids[-1]]
    u, mass = endpoint_quadrature(w0, w1)
    endpoints = np.stack([histogram_quantile(u, w, edges) for w in [w0, w1]])
    center = float((endpoints * mass).sum() / 2)
    scale = float(np.sqrt(((endpoints - center) ** 2 * mass).sum() / 2))
    physical = np.stack(
        [histogram_quantile(u, data["weights"][i], edges) for i in ids]
    )[:, :, None]
    samples = (physical - center) / scale
    optical_index = int(np.flatnonzero(data["wavelengths"] == 550)[0])
    target = data["optics"][ids[2], optical_index] / data["cpc"][ids[2]] / 0.01
    references = np.concatenate(
        [histogram_quantile((np.arange(256) + 0.5) / 256, w, edges) for w in [w0, w1]]
    )[:, None]
    np.savez_compressed(
        out / "train.npz",
        x0=samples[0],
        x1=samples[-1],
        weights=mass,
        target=np.array([target]),
        tau=0.5,
        center=np.array([center]),
        scale=np.array([scale]),
        land_references=(references - center) / scale,
        quantiles=u,
    )
    np.savez_compressed(
        out / "evaluation.npz",
        samples=samples,
        physical_log_diameter=physical,
        times=np.linspace(0, 1, 5),
        weights=mass,
        bin_probabilities=data["weights"][ids],
        log_bin_edges=edges,
    )
    write_json(
        out / "metadata.json",
        {
            "name": "hyy_test_00",
            "split": "test",
            "domain": "aerosol",
            "endpoint_representation": "shared_quantile_quadrature",
            "endpoint_count": len(u),
            "observation_config": {
                "type": "tabulated_optical",
                "physical_center": [center],
                "physical_scale": [scale],
                "log_diameter_grid": np.log(grid).tolist(),
                "response_um2": response.tolist(),
                "feature_scale_um2": 0.01,
            },
            "feature_type": "tabulated_optical",
            "constraint_time": 0.5,
            "unobserved_times": [0.25, 0.75],
            "timestamps_utc": [str(t + pd.Timedelta(minutes=30)) for t in times],
            "land_reference_pool_policy": "endpoints_only",
            "sources": provenance,
            "calibration_period": "2012-01-01 through 2012-02-29",
            "calibration_count": int(calibration.sum()),
            "refractive_index": float(fitted.x[0]),
            "optical_gain": float(np.exp(fitted.x[1])),
            "information": "Endpoint DMPS histograms; independent midpoint nephelometer/CPC scalar. "
            "Optical response fitted only to January-February measurements.",
            "limitations": [
                "Homogeneous-sphere response and nominal PM1 inlet approximate unknown particle composition."
            ],
        },
    )
