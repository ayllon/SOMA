from typing import List, Iterable

import numpy as np
from sklearn.decomposition import PCA

from soma.generators import Generator


class PCAGenerator(Generator):
    """
    Decorate another generator, performing a dimensionality reduction using PCA.

    Parameters
    ----------
    generator : Generator
        Generator to reduce
    pca: PCA
        Fitted PCA
    """

    @staticmethod
    def fit(generators: Iterable[Generator], *, fit_samples: int = 1000):
        """
        Fit a PCA from one, or several, generators

        Parameters
        ----------
        generators : Iterable[Generator]
            List of generators to use for the fitting
        dimensions : int
            Number of components to pick
        fit_samples : int
            If the generator has no data array, use this number of samples to fit
        Returns
        -------
        out : PCA
            A fitted PCA
        """
        data = []
        for g in generators:
            if hasattr(generators, 'array'):
                data.append(g.array)
            else:
                data.append(g.sample(fit_samples))
        data = np.concatenate(data)
        return PCA(n_components=None).fit(data)

    def __init__(self, generator: Generator, dimensions: int, pca: PCA):
        self.__pca = pca
        self.__d = dimensions
        self.__generator = generator

    @property
    def dimensions(self) -> int:
        return self.__d

    def sample(self, n: int) -> np.ndarray:
        return self.__pca.transform(self.__generator.sample(n))[:, :self.__d]
