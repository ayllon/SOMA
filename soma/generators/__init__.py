import abc
from typing import Iterable

import numpy as np


class Generator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        pass


class Reducer(Generator):
    @staticmethod
    @abc.abstractmethod
    def fit(generators: Iterable[Generator], *, fit_samples: int = 1000):
        pass
