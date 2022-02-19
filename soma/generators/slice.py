from typing import Iterable, Type

import numpy as np

from soma.generators import Generator, Reducer


class SliceGenerator(Reducer):
    @staticmethod
    def fit(generators: Iterable[Generator], *, fit_samples: int = 1000):
        return None

    def __init__(self, generator: Generator, dimensions: int, feat_selector: Type[None]):
        self.__generator = generator
        self.__d = dimensions

    def sample(self, n: int) -> np.ndarray:
        return self.__generator.sample(n)[:, :self.__d]

    @property
    def dimensions(self) -> int:
        return self.__d
