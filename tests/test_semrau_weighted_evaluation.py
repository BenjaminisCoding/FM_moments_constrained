import numpy as np
import ot
from scipy.spatial.distance import cdist

from cfm_project.semrau_weighted_evaluation import exact_wasserstein, merge_atoms


def test_duplicate_atoms_preserve_weighted_transport():
    x = np.array([[0., 1.], [2., -1.], [0., 1.], [3., 4.], [2., -1.]])
    a = np.array([.07, .23, .13, .41, .16])
    y = np.array([[1., 0.], [1., 0.], [2., 3.], [-2., 1.]])
    scores = exact_wasserstein(x, y, a)
    squared = cdist(x, y, 'sqeuclidean')
    for name, cost in [('W1', np.sqrt(squared)), ('W2', squared)]:
        expected = ot.emd2(a, np.full(len(y), 1 / len(y)), cost)
        if name == 'W2':
            expected = np.sqrt(expected)
        np.testing.assert_allclose(scores[name], expected, rtol=0, atol=2e-14)
    unique, mass = merge_atoms(x, a)
    assert len(unique) == 3
    np.testing.assert_allclose(mass, [.2, .39, .41], rtol=0, atol=2e-15)
