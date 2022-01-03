import numpy as np
from scipy.spatial.ckdtree import cKDTree


def kl_two_samples(p: np.ndarray, q: np.ndarray, k: int = 1) -> float:
    """
    Based on F. Pérez-Cruz, "Kullback-Leibler divergence estimation of continuous distributions,"
    in 2008 IEEE international symposium on information theory, 2008, pp. 1666–1670.

    The estimator is

    .. math::
        \hat{D}(P || Q) = {d \over n} \sum_{i=1}^{n} log {{r_k(x_i)} \over {s_k(x_i)}} + log {m \over {n-1}}

    Where
    * :math:`n` is the number of samples from p
    * :math:`m` is the number of samples from q
    * :math:`d` is the dimensionality (len(p.shape) - 1)
    * :math:`r_k(x_i)` is the Euclidean distance to the :math:`k^{th}` nearest neighbor of :math:`x_i` in :math:`p \setminus x_i`
    * :math:`s_k(x_i)` is the Euclidean distance to the :math:`k^{th}` nearest neighbor of :math:`x_i` in q

    Parameters
    ----------
    p: np.ndarray
        First set of samples, with the shape [samples, features]
    q: np.ndarray
        Second set of samples, with the shape [samples, features]
    k : int
        Neighbor to use. We default to 2 to a void a situation where the distance to the nearest neighbor happens to be 0
        (i.e. if p is a re-sample of q)

    Returns
    -------
    The Kullback-Leibler divergence estimation.
    """
    assert len(p.shape) > 1, 'Need at least one feature'

    n, m = len(p), len(q)
    d = p.shape[1]

    ptree = cKDTree(p)
    qtree = cKDTree(q)

    rx = ptree.query(p, k=k + 1)[0][:, -1]
    sx = qtree.query(p, k=k)[0]
    if len(sx.shape) > 1:
        sx = sx[:, -1]
    # Note there is an error on formula (14)
    # Either we take the log of $s_k(x_i) / r_k(x_i)$, or there is a missing -1 on the expression
    return (d / n) * np.sum(np.log(sx / rx)) + np.log(m / (n - 1))
