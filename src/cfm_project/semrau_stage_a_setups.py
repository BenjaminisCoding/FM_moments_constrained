"""Semrau endpoint data and aggregate moments at the declared constraint time.

Only endpoint training arrays and frozen aggregate metadata are read here.
Evaluation arrays are intentionally loaded by a separate reporting function.
"""
import json
from pathlib import Path

import numpy as np
import torch

from cfm_project.semrau_benchmark import Data, features
from cfm_project.paths import corrected_path


def load_data(root):
    root = Path(root)
    meta = json.loads((root / 'metadata.json').read_text())
    with np.load(root / 'train.npz', allow_pickle=False) as arrays:
        x0 = torch.tensor(arrays['x0'], dtype=torch.float32)
        x1 = torch.tensor(arrays['x1'], dtype=torch.float32)
        return Data(x0, x1, x0[arrays['pair_i']], x1[arrays['pair_j']],
                    torch.tensor(arrays['pair_weights'], dtype=torch.float32),
                    torch.tensor(meta['matrix'], dtype=torch.float32),
                    torch.tensor(meta['target'], dtype=torch.float32),
                    torch.tensor(meta['normalization'], dtype=torch.float32), {}, meta)


def moment_residual(path, data, normalized=False):
    t = torch.full((len(data.weights), 1), data.metadata['tau'])
    x = (1-t)*data.pair0+t*data.pair1 if path is None else corrected_path(t, data.pair0, data.pair1, path)
    residual = (features(x, data.matrix)*data.weights[:, None]).sum(0)-data.target
    return residual/data.scale/np.sqrt(len(data.target)) if normalized else residual
