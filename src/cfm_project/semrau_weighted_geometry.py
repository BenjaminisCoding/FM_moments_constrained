"""Canonical LAND energy under a fixed endpoint-derived coordinate map."""
from cfm_project.mfm_core import land_metric_tensor


def land_energy(x,u,references,cfg):
    matrix=cfg.get('geometry_matrix')
    if matrix is not None:
        transform=x.new_tensor(matrix)
        x=x@transform.T
        u=u@transform.T
        references=references@transform.T
    diagonal=land_metric_tensor(x,references,cfg['land_gamma'],cfg['land_rho'])
    return (u.square()*diagonal).sum(1)
