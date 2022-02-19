import numpy as np
from scipy.stats import wishart, multivariate_normal

from soma.generators import Generator


class MultivariateNormalGenerator(Generator):
    """
    Multivariate normal generator

    Parameters
    ----------
    mean : float or an array with `d` values
    cov : [Optional] np.ndarray
        Covariance matrix
    wishart_df : [Optional] int
        If cov is not provided, it can be generated using the Wishart distribution with this many degrees
        of freedom. Note that wishart_df must be greater or equal to d. The higher the degree of freedom,
        the more similar to the scale matrix (identity by default) the generated covariance matrix is.
    wishart_scale: [Optional] np.ndarray
        Scale matrix for the Wishart distribution. Defaults to the identity matrix.
    """

    def __init__(self, mean: np.ndarray, *, cov: np.ndarray = None,
                 wishart_df: int = None, wishart_scale: np.ndarray = None):
        assert (wishart_df is not None) ^ (
                cov is not None), 'Either covariance matrix or degrees of freedom must be specified'
        d = len(mean)
        if cov is None:
            assert wishart_df >= d, 'The number of degrees of freedom must be equal or greater to the dimensionality'
            if wishart_scale is None:
                wishart_scale = np.diag(np.ones(d))
            cov = wishart(df=wishart_df, scale=wishart_scale).rvs() / wishart_df
        self.__dist = multivariate_normal(mean, cov=cov)

    @property
    def dimensions(self) -> int:
        return len(self.__dist.mean)

    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples
        """
        return self.__dist.rvs(n)
