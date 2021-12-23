from typing import Iterable

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from soma.generators import Generator


class KBestGenerator(Generator):
    @staticmethod
    def fit(generators: Iterable[Generator], *, fit_samples: int = 1000):
        data = []
        label = []
        for i, g in enumerate(generators):
            if hasattr(generators, 'array'):
                data.append(g.array)
                label.append(np.full(len(g.array), i, dtype=int))
            else:
                data.append(g.sample(fit_samples))
                label.append(np.full(fit_samples, i, dtype=int))
        data = np.concatenate(data)
        label = np.concatenate(label, dtype=int)
        feat_selector = SelectKBest(f_classif, k='all')
        feat_selector.fit(data, label)
        return feat_selector

    def __init__(self, generator: Generator, dimensions: int, feat_selector: SelectKBest):
        self.__generator = generator
        self.__d = dimensions
        self.__selector = feat_selector

    def sample(self, n: int) -> np.ndarray:
        return self.__selector.transform(self.__generator.sample(n))[:, :self.__d]

    @property
    def dimensions(self) -> int:
        return self.__d
